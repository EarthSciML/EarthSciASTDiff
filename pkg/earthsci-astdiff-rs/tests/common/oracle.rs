//! A test-only value oracle: the band entries evaluated point by point, each
//! coefficient's reads resolved to literals and the closed tree handed to
//! EarthSciAST's `evaluate`. The crate itself evaluates nothing (the values
//! are EarthSciAST's to compute from the emitted documents); this was the
//! binding's `numeric` module and stays here to check the band calculus
//! against the value goldens the Julia reference pins by evaluating its
//! documents, until this binding emits the product documents (spec §9) too.
#![allow(dead_code)]

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::evaluate;
use earthsci_ast::substitute;
use earthsci_ast::{EsmFile, Expr, VariableType};
use earthsci_astdiff::bands::{range_of, Band, BandError, Ctx};
use earthsci_astdiff::helpers::map_expr_children;
use earthsci_astdiff::inline::inline_observed_cached;
use earthsci_astdiff::points::{
    cellname, check_state_column, foreach_point, outside, product_first_fastest,
};
use earthsci_astdiff::reductions::reduce_op;
use earthsci_astdiff::system::{ctx_of, observed_defs, sysview, view_states, SysView};
use earthsci_astdiff::{
    axis_pair, column_names, jacobian_bands2_with_forcing, jacobian_bands_view, jacobian_block2,
    jacobian_columns, JacEntry, JacEntry2, Wrt,
};
use serde_json::Value;

type R<T> = Result<T, BandError>;

/// `for_each_value`'s callback: `(k, entry, row cell, contracted cell,
/// column cell, value)`.
type ValueFn<'a> = dyn FnMut(usize, &JacEntry, &[i64], &[i64], &[i64], f64) + 'a;

#[derive(Clone, Debug)]
pub struct Env {
    /// name → nested array (1-based cells encoded as serde Value) or scalar
    pub states: HashMap<String, Value>,
    pub params: HashMap<String, Value>,
    pub t: f64,
}

fn gather<'a>(v: &'a Value, cells: &[i64]) -> Option<&'a Value> {
    let mut cur = v;
    for c in cells {
        cur = cur.as_array()?.get((*c - 1) as usize)?;
    }
    Some(cur)
}

fn as_lit(e: &Expr) -> R<f64> {
    match e {
        Expr::Integer(i) => Ok(*i as f64),
        Expr::Number(x) => Ok(*x),
        _ => evaluate(e, &HashMap::new()).map_err(|er| BandError(format!("evaluate: {er:?}"))),
    }
}

fn resolve(e: &Expr, env: &Env, ctx: &Ctx) -> R<Expr> {
    match e {
        Expr::Variable(name) => {
            if name == "t" {
                return Ok(Expr::Number(env.t));
            }
            for m in [&env.params, &env.states] {
                if let Some(v) = m.get(name) {
                    if let Some(x) = v.as_f64() {
                        return Ok(Expr::Number(x));
                    }
                }
            }
            Err(BandError(format!("unbound name `{name}` in coefficient")))
        }
        Expr::Operator(n) if n.op == "index" => {
            let mut cells = Vec::new();
            for x in &n.args[1..] {
                cells.push(as_lit(&resolve(x, env, ctx)?)?.round() as i64);
            }
            match n.args.first() {
                Some(Expr::Variable(t)) => {
                    let arr = env
                        .states
                        .get(t)
                        .or_else(|| env.params.get(t))
                        .ok_or_else(|| BandError(format!("index into unbound array `{t}`")))?;
                    let v = gather(arr, &cells)
                        .ok_or_else(|| BandError(format!("out-of-range gather into `{t}`")))?;
                    Ok(Expr::Number(
                        v.as_f64()
                            .ok_or_else(|| BandError("non-numeric gather".into()))?,
                    ))
                }
                Some(Expr::Operator(cn)) if cn.op == "const" => {
                    let v = cn
                        .value
                        .as_ref()
                        .ok_or_else(|| BandError("const without value".into()))?;
                    let v = gather(v, &cells)
                        .ok_or_else(|| BandError("out-of-range const gather".into()))?;
                    Ok(Expr::Number(v.as_f64().ok_or_else(|| {
                        BandError("non-numeric const gather".into())
                    })?))
                }
                _ => Err(BandError("index into unsupported target".into())),
            }
        }
        Expr::Operator(n) if n.op == "faq" => {
            if n.output_idx
                .as_ref()
                .map(|o| !o.is_empty())
                .unwrap_or(false)
            {
                return Err(BandError(
                    "array-producing faq in scalar coefficient".into(),
                ));
            }
            // ⊕ per esm-spec: a filtered-out term is skipped; the empty
            // reduction is the semiring identity (0, 1, -inf, +inf); terms
            // fold in ascending order, first index fastest.
            let red = reduce_op(n).map_err(|m| BandError(format!("{m} in scalar coefficient")))?;
            let empty = HashMap::new();
            let ranges = n.ranges.as_ref().unwrap_or(&empty);
            let mut names: Vec<String> = ranges.keys().cloned().collect();
            names.sort();
            let boxes: Vec<(i64, i64)> = names
                .iter()
                .map(|nm| range_of(&ranges[nm], ctx))
                .collect::<Result<_, _>>()?;
            let body = n
                .expr
                .as_deref()
                .ok_or_else(|| BandError("faq without body".into()))?;
            let mut total = match red {
                "+" => 0.0,
                "*" => 1.0,
                "max" => f64::NEG_INFINITY,
                _ => f64::INFINITY,
            };
            if boxes.iter().any(|(lo, hi)| hi < lo) {
                return Ok(Expr::Number(total));
            }
            for cell in product_first_fastest(&boxes) {
                let binds: HashMap<String, Expr> = names
                    .iter()
                    .cloned()
                    .zip(cell.iter().map(|c| Expr::Integer(*c)))
                    .collect();
                if let Some(f) = &n.filter {
                    if as_lit(&resolve(&substitute(f, &binds), env, ctx)?)? == 0.0 {
                        continue;
                    }
                }
                let x = as_lit(&resolve(&substitute(body, &binds), env, ctx)?)?;
                total = match red {
                    "+" => total + x,
                    "*" => total * x,
                    "max" => {
                        if x > total {
                            x
                        } else {
                            total
                        }
                    }
                    _ => {
                        if x < total {
                            x
                        } else {
                            total
                        }
                    }
                };
            }
            Ok(Expr::Number(total))
        }
        Expr::Operator(_) => {
            let mut err: Option<BandError> = None;
            let out = map_expr_children(e, &mut |x| match resolve(x, env, ctx) {
                Ok(v) => v,
                Err(er) => {
                    if err.is_none() {
                        err = Some(er);
                    }
                    x.clone()
                }
            });
            match err {
                Some(er) => Err(er),
                None => Ok(out),
            }
        }
        _ => Ok(e.clone()),
    }
}

/// The coefficient of every band with the observed it reads spliced in.
/// Pruning leaves target-independent observed as opaque reads in the
/// coefficients; resolve them from their own definitions (what the
/// reference's derived evaluation model does). This splices in only the
/// observed a coefficient actually reads — exactly what evaluating it at a
/// point needs — so it rejects nothing an evaluation could do without.
fn inlined_coefs<'a>(
    obs: &HashMap<String, Expr>,
    bands: impl IntoIterator<Item = &'a Band>,
) -> R<Vec<Expr>> {
    let mut obs_names = None;
    bands
        .into_iter()
        .map(|b| inline_observed_cached(&b.coef, obs, &mut obs_names).map_err(|e| BandError(e.0)))
        .collect()
}

/// One band coefficient (an [`inlined_coefs`] entry) at one point: the
/// point's index binding substituted, every read resolved at `env`.
fn point_value(coef0: &Expr, ibind: &HashMap<String, i64>, env: &Env, ctx: &Ctx) -> R<f64> {
    if ibind.is_empty() {
        return as_lit(&resolve(coef0, env, ctx)?);
    }
    let sub: HashMap<String, Expr> = ibind
        .iter()
        .map(|(k, v)| (k.clone(), Expr::Integer(*v)))
        .collect();
    as_lit(&resolve(&substitute(coef0, &sub), env, ctx)?)
}

/// Every non-ghost point of every band at a point: `f(k, entry, row cell,
/// contracted cell, column cell, value)`, bands in order, cells first index
/// fastest. `coefs` are the bands' [`inlined_coefs`].
///
/// On the forcing axis the ghost skip is an error instead: a column there is
/// a cell of a live buffer, and an unresolvable one would silently drop a
/// gradient (spec §4). The parameters axis follows the state axis: an array
/// parameter read past its extent is a ghost read. On the states axis a
/// column whose variable is not a state at all (`states`) resolves nowhere
/// and is an error rather than a silently dropped derivative.
fn for_each_value(
    ctx: &Ctx,
    entries: &[JacEntry],
    coefs: &[Expr],
    env: &Env,
    wrt: Wrt,
    states: Option<&HashSet<String>>,
    f: &mut ValueFn<'_>,
) -> R<()> {
    for (k, (en, coef0)) in entries.iter().zip(coefs).enumerate() {
        // Checked at the entry's first point, as the reference resolves
        // columns: an entry with no point (its guard excludes every one)
        // has no column to resolve.
        let mut checked = states.is_none();
        let vshape = ctx.shapes.get(&en.v);
        foreach_point(&en.band, &mut |cell, cc, ibind, cidx| {
            if !checked {
                check_state_column(states.expect("checked otherwise"), &en.v, &en.u)?;
                checked = true;
            }
            if let Some(vshape) = vshape {
                if cidx.iter().zip(vshape.iter()).any(|(c, s)| *c < 1 || c > s) {
                    if wrt == Wrt::Forcing {
                        return Err(BandError(format!(
                            "column `{}` (row `{}`) is outside the declared extent of \
                             its variable; on the {} axis every column must resolve \
                             (no ghost skip)",
                            cellname(&en.v, cidx),
                            cellname(&en.u, cell),
                            wrt.as_str()
                        )));
                    }
                    return Ok(()); // ghost read: no column
                }
            }
            f(k, en, cell, cc, cidx, point_value(coef0, ibind, env, ctx)?);
            Ok(())
        })?;
    }
    Ok(())
}

/// A value of `shape` with every cell `x`, as row-major nested arrays (the
/// inline array form).
fn nested_full(shape: &[i64], x: f64) -> Value {
    match shape.split_first() {
        None => serde_json::json!(x),
        Some((n, rest)) => Value::Array((0..*n).map(|_| nested_full(rest, x)).collect()),
    }
}

/// Every parameter's document value, as the Python binding's defaults read
/// it: an inline array `default` as written (row-major nested arrays), a
/// scalar `default` broadcast over an array parameter's shape, and a missing
/// one 0 (every cell of an array parameter included). What the reference's
/// assembly reads when no buffer is bound.
fn parameter_defaults(sv: &SysView, ctx: &Ctx) -> HashMap<String, Value> {
    sv.variables
        .iter()
        .filter(|(_, v)| v.var_type == VariableType::Parameter)
        .map(|(n, v)| {
            let d = v
                .default
                .as_ref()
                .and_then(|d| serde_json::to_value(d).ok());
            let shape = ctx.shapes.get(n).filter(|s| !s.is_empty());
            let val = match (d, shape) {
                (Some(d), _) if d.is_array() => d,
                (d, Some(shape)) => nested_full(shape, d.and_then(|x| x.as_f64()).unwrap_or(0.0)),
                (d, None) => serde_json::json!(d.and_then(|x| x.as_f64()).unwrap_or(0.0)),
            };
            (n.clone(), val)
        })
        .collect()
}

/// `env` with every parameter it leaves unbound filled from `defaults`
/// ([`parameter_defaults`]).
fn with_defaults<'e>(defaults: &HashMap<String, Value>, env: &'e Env) -> Cow<'e, Env> {
    if defaults.keys().all(|n| env.params.contains_key(n)) {
        return Cow::Borrowed(env);
    }
    let mut params = env.params.clone();
    for (n, v) in defaults {
        params.entry(n.clone()).or_insert_with(|| v.clone());
    }
    Cow::Owned(Env {
        states: env.states.clone(),
        params,
        t: env.t,
    })
}

/// A prepared band evaluation for one model and axis, built once by
/// [`prepare_jacobian`]: the band calculus, the observed-inlined
/// coefficients, the column space and the parameter defaults. Each later
/// evaluation ([`JacobianEvaluator::values`], [`JacobianEvaluator::vjp`],
/// [`JacobianEvaluator::vjp_forcing`]) only walks the band points at the
/// given [`Env`] — so an adjoint loop that takes one forcing cotangent per
/// stage does not redo the build-time work per stage.
#[derive(Clone, Debug)]
pub struct JacobianEvaluator {
    pub wrt: Wrt,
    pub entries: Vec<JacEntry>,
    /// The [`earthsci_astdiff::jacobian_columns`] space on the parameters and forcing
    /// axes, `None` otherwise.
    pub columns: Option<Vec<(String, Vec<i64>)>>,
    /// The canonical column names of `columns` (empty without one).
    pub colnames: Vec<String>,
    ctx: Ctx,
    coefs: Vec<Expr>,
    slot: HashMap<String, usize>,
    defaults: HashMap<String, Value>,
    /// The state variables, on the states axis (columns must name one).
    states: Option<HashSet<String>>,
}

/// Build a [`JacobianEvaluator`] once: differentiation, observed inlining,
/// the column space (parameters and forcing axes) and the parameter defaults.
pub fn prepare_jacobian(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> R<JacobianEvaluator> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    let entries = jacobian_bands_view(&sv, wrt, forcing)?;
    let coefs = inlined_coefs(&obs, entries.iter().map(|en| &en.band))?;
    let columns = if wrt.has_columns() {
        Some(jacobian_columns(&sv, wrt, forcing)?)
    } else {
        None
    };
    let colnames = columns.as_deref().map(column_names).unwrap_or_default();
    let slot = colnames
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let defaults = parameter_defaults(&sv, &ctx);
    let states = (wrt == Wrt::States).then(|| view_states(&sv, &obs).into_iter().collect());
    Ok(JacobianEvaluator {
        wrt,
        entries,
        columns,
        colnames,
        ctx,
        coefs,
        slot,
        defaults,
        states,
    })
}

impl JacobianEvaluator {
    fn each_value(&self, env: &Env, f: &mut ValueFn<'_>) -> R<()> {
        let env = with_defaults(&self.defaults, env);
        for_each_value(
            &self.ctx,
            &self.entries,
            &self.coefs,
            &env,
            self.wrt,
            self.states.as_ref(),
            f,
        )
    }

    /// The band entries at a point, as [`assemble_jacobian`] returns them.
    pub fn values(&self, env: &Env) -> R<BTreeMap<(String, String), f64>> {
        let mut out: BTreeMap<(String, String), f64> = BTreeMap::new();
        self.each_value(env, &mut |_, en, cell, _, cidx, val| {
            let key = (cellname(&en.u, cell), cellname(&en.v, cidx));
            *out.entry(key).or_insert(0.0) += val;
        })?;
        Ok(out)
    }

    /// `ḡ += Jᵀ λ` at a point, as [`vjp`] computes it, on this evaluator's
    /// parameters or forcing axis.
    pub fn vjp(&self, env: &Env, lambda: &HashMap<String, f64>, gbar: &mut [f64]) -> R<()> {
        if !self.wrt.has_columns() {
            return Err(BandError(format!(
                "vjp needs wrt = parameters or forcing, got {}",
                self.wrt.as_str()
            )));
        }
        if gbar.len() != self.colnames.len() {
            return Err(BandError(format!(
                "ḡ has length {}, the column space {}",
                gbar.len(),
                self.colnames.len()
            )));
        }
        let mut err: Option<BandError> = None;
        self.each_value(env, &mut |_, en, cell, _, cidx, val| {
            if err.is_some() {
                return;
            }
            let (r, c) = (cellname(&en.u, cell), cellname(&en.v, cidx));
            match (lambda.get(&r), self.slot.get(&c)) {
                (Some(l), Some(&k)) => gbar[k] += val * l,
                (None, _) => err = Some(BandError(format!("λ binds no row `{r}`"))),
                (_, None) => {
                    err = Some(BandError(format!(
                        "column `{c}` is not in the column space"
                    )))
                }
            }
        })?;
        err.map_or(Ok(()), Err)
    }

    /// The forcing cotangent of one stage evaluation into buffer-shaped
    /// accumulators, as [`vjp_forcing`] computes it, on this evaluator's
    /// forcing axis.
    pub fn vjp_forcing(
        &self,
        env: &Env,
        lambda: &HashMap<String, f64>,
        gbar: &mut HashMap<String, Vec<f64>>,
    ) -> R<()> {
        if self.wrt != Wrt::Forcing {
            return Err(BandError(format!(
                "vjp_forcing needs a wrt = forcing evaluator, got wrt = {}",
                self.wrt.as_str()
            )));
        }
        let mut flat = vec![0.0; self.colnames.len()];
        self.vjp(env, lambda, &mut flat)?;
        let mut off = 0;
        for (name, shape) in self.columns.as_deref().unwrap_or_default() {
            let len = shape.iter().product::<i64>() as usize;
            let acc = gbar
                .get_mut(name)
                .ok_or_else(|| BandError(format!("no accumulator for forcing buffer `{name}`")))?;
            if acc.len() != len {
                return Err(BandError(format!(
                    "accumulator `{name}` has length {}, the buffer {len}",
                    acc.len()
                )));
            }
            for (a, g) in acc.iter_mut().zip(&flat[off..off + len]) {
                *a += g;
            }
            off += len;
        }
        Ok(())
    }
}

/// Evaluate the band entries at a point: `{(rowname, colname): value}`,
/// contributions summed (§4 additivity, contracted points included). The
/// forcing set is inferred; see [`assemble_jacobian_with_forcing`].
pub fn assemble_jacobian(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    env: &Env,
) -> R<BTreeMap<(String, String), f64>> {
    assemble_jacobian_with_forcing(file, model_name, wrt, None, env)
}

/// [`assemble_jacobian`] with an explicit forcing set. On the parameters and
/// forcing axes the column names are the canonical column space of
/// [`earthsci_astdiff::jacobian_columns`] (`"E[1]"`, `"F[2,3]"`, scalars bare). A
/// forcing column outside its buffer's extent is an error (no ghost skip); a
/// parameter or state column read past its extent is a ghost read and is
/// skipped. A parameter `env.params` does not bind reads the document's
/// `default`, broadcast over an array parameter's shape (0 without one).
/// For repeated evaluation, [`prepare_jacobian`] once.
pub fn assemble_jacobian_with_forcing(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    env: &Env,
) -> R<BTreeMap<(String, String), f64>> {
    prepare_jacobian(file, model_name, wrt, forcing)?.values(env)
}

/// The transposed product `ḡ += Jᵀ λ` at a point, as a scatter-add over the
/// band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix
/// (and no transpose) formed. `lambda` is keyed by row cell name (`"c[2]"`,
/// `"S"`); a row it does not bind is an error. `gbar` is indexed by the
/// canonical column ordinal of a parameters or forcing axis
/// ([`earthsci_astdiff::jacobian::column_names`] of [`earthsci_astdiff::jacobian_columns`]) and
/// ACCUMULATES — zero it first for a single product. This prepares the
/// evaluator for one call; a loop calls [`JacobianEvaluator::vjp`] on one
/// [`prepare_jacobian`] instead.
pub fn vjp(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    env: &Env,
    lambda: &HashMap<String, f64>,
    gbar: &mut [f64],
) -> R<()> {
    prepare_jacobian(file, model_name, wrt, forcing)?.vjp(env, lambda, gbar)
}

/// [`vjp`] on the forcing axis, into buffer-shaped accumulators: the forcing
/// cotangent of one stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (spec §4.3). `gbar`
/// holds one accumulator per forcing buffer, keyed by buffer name, each the
/// buffer's cells COLUMN-MAJOR (first index fastest). It accumulates; which
/// dataset record a buffer slot holds, and when to flush, is the caller's
/// bookkeeping. This prepares the evaluator for one call; an adjoint loop
/// calls [`JacobianEvaluator::vjp_forcing`] on one [`prepare_jacobian`]
/// instead.
pub fn vjp_forcing(
    file: &EsmFile,
    model_name: Option<&str>,
    forcing: Option<&[String]>,
    env: &Env,
    lambda: &HashMap<String, f64>,
    gbar: &mut HashMap<String, Vec<f64>>,
) -> R<()> {
    prepare_jacobian(file, model_name, Wrt::Forcing, forcing)?.vjp_forcing(env, lambda, gbar)
}

/// The band fields at a point — what the derived evaluation document's
/// `J_k` states hold, and what a [`earthsci_astdiff::ScatterPlan`] scatters: field `k`
/// over the row shape then the contracted extents, column-major. Ghost
/// points (no column) and cells outside the band region stay 0.
pub fn band_fields(file: &EsmFile, model_name: Option<&str>, env: &Env) -> R<Vec<Vec<f64>>> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    let entries = jacobian_bands_view(&sv, Wrt::States, None)?;
    let shapes: Vec<Vec<i64>> = entries
        .iter()
        .map(|en| earthsci_astdiff::scatter::field_shape(en, &ctx))
        .collect();
    let mut fields: Vec<Vec<f64>> = shapes
        .iter()
        .map(|sh| vec![0.0; sh.iter().product::<i64>() as usize])
        .collect();
    let coefs = inlined_coefs(&obs, entries.iter().map(|en| &en.band))?;
    let env = with_defaults(&parameter_defaults(&sv, &ctx), env);
    let states: HashSet<String> = view_states(&sv, &obs).into_iter().collect();
    for_each_value(
        &ctx,
        &entries,
        &coefs,
        &env,
        Wrt::States,
        Some(&states),
        &mut |k, _, cell, cc, _, val| {
            let full: Vec<i64> = cell.iter().chain(cc.iter()).copied().collect();
            fields[k][earthsci_astdiff::scatter::flat(&full, &shapes[k])] = val;
        },
    )?;
    Ok(fields)
}

// ── the second-order contraction (second_order.jl `hvp!`, spec §4.4) ────────

/// One stage pair `(μ, w)` of [`hvp`]: `μ` keyed by row cell name, `w` by
/// first-column cell name (on a time pair, the single column `"t"`).
pub type HvpPair = (HashMap<String, f64>, HashMap<String, f64>);

/// Which side of `∇²(μᵀ f)` [`HvpEvaluator::hvp`] contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HvpForm {
    /// Sum the stages into `M(row, col) = Σᵢ μᵢ[row] wᵢ[col]`, then
    /// `g[col2] += coef2 · M(row, col)` over the points.
    Contract,
    /// On a (states, states) pair only, by the symmetry of `∇²(μᵀ f)`:
    /// `g[col] += μ[row] · coef2 · w[col2]` per stage, with no `M`.
    Symmetric,
}

/// A prepared second-order contraction for one axis pair, built once by
/// [`prepare_hvp`]: the second-order bands, their observed-inlined
/// coefficients, the second axis's column space and the parameter defaults.
/// Each [`HvpEvaluator::hvp`] only walks the band points at the given
/// [`Env`].
#[derive(Clone, Debug)]
pub struct HvpEvaluator {
    pub wrt: (Wrt, Wrt),
    pub entries: Vec<JacEntry2>,
    /// The second axis's [`earthsci_astdiff::jacobian_columns`] space on a parameters
    /// or forcing pair, `None` otherwise.
    pub columns: Option<Vec<(String, Vec<i64>)>>,
    /// The canonical column names of `columns` (empty without one).
    pub colnames: Vec<String>,
    /// `Some("block_diagonal")` when the contraction is lane-local (the
    /// pair block's `structure`).
    pub structure: Option<&'static str>,
    ctx: Ctx,
    coefs: Vec<Expr>,
    defaults: HashMap<String, Value>,
    states: HashSet<String>,
}

/// Build an [`HvpEvaluator`] once for the axis pair `wrt` (first axis states
/// or time, second states, parameters or forcing).
pub fn prepare_hvp(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
) -> R<HvpEvaluator> {
    let wrt = axis_pair(wrt)?;
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    let entries = jacobian_bands2_with_forcing(file, model_name, wrt, forcing)?;
    let coefs = inlined_coefs(&obs, entries.iter().map(|en| &en.band))?;
    let columns = if wrt.1.has_columns() {
        Some(jacobian_columns(&sv, wrt.1, forcing)?)
    } else {
        None
    };
    let colnames = columns.as_deref().map(column_names).unwrap_or_default();
    let structure = jacobian_block2(file, model_name, wrt, forcing, false, 12)?
        .get("structure")
        .and_then(Value::as_str)
        .map(|s| {
            if s == "block_diagonal" {
                "block_diagonal"
            } else {
                "other"
            }
        });
    let defaults = parameter_defaults(&sv, &ctx);
    let states = view_states(&sv, &obs).into_iter().collect();
    Ok(HvpEvaluator {
        wrt,
        entries,
        columns,
        colnames,
        structure,
        ctx,
        coefs,
        defaults,
        states,
    })
}

impl HvpEvaluator {
    /// Every non-ghost point of every second-order band at a point:
    /// `f(row, col, col2, value)` by cell name. A column (first or second)
    /// outside its variable's extent is a ghost read and skipped, except a
    /// second column on the forcing axis, which is an error (spec §4.3).
    fn each_point(&self, env: &Env, f: &mut dyn FnMut(&str, &str, &str, f64) -> R<()>) -> R<()> {
        let env = with_defaults(&self.defaults, env);
        for (en, coef0) in self.entries.iter().zip(&self.coefs) {
            // Checked at the entry's first point (see `for_each_value`).
            let mut checked = false;
            let vshape = self.ctx.shapes.get(&en.v);
            let wshape = self.ctx.shapes.get(&en.w);
            foreach_point(&en.band, &mut |cell, _, ibind, cidx| {
                if !checked {
                    if self.wrt.0 == Wrt::States {
                        check_state_column(&self.states, &en.v, &en.u)?;
                    }
                    if self.wrt.1 == Wrt::States {
                        check_state_column(&self.states, &en.w, &en.u)?;
                    }
                    checked = true;
                }
                let (c1, c2) = cidx.split_at(en.ncol);
                if outside(wshape, c2) {
                    if self.wrt.1 == Wrt::Forcing {
                        return Err(BandError(format!(
                            "second column `{}` (row `{}`) is outside the declared extent \
                             of its variable; on the forcing axis every column must \
                             resolve (no ghost skip)",
                            cellname(&en.w, c2),
                            cellname(&en.u, cell)
                        )));
                    }
                    return Ok(());
                }
                if outside(vshape, c1) {
                    return Ok(()); // ghost read: no column
                }
                let val = point_value(coef0, ibind, &env, &self.ctx)?;
                f(
                    &cellname(&en.u, cell),
                    &cellname(&en.v, c1),
                    &cellname(&en.w, c2),
                    val,
                )
            })?;
        }
        Ok(())
    }

    /// The second-order contraction at a point, as [`hvp`] computes it.
    pub fn hvp(
        &self,
        env: &Env,
        pairs: &[HvpPair],
        form: HvpForm,
        g: &mut BTreeMap<String, f64>,
    ) -> R<()> {
        let get = |m: &HashMap<String, f64>, k: &str, what: &str| {
            m.get(k)
                .copied()
                .ok_or_else(|| BandError(format!("{what} binds no `{k}`")))
        };
        match form {
            HvpForm::Contract => {
                let mut m: HashMap<(String, String), f64> = HashMap::new();
                self.each_point(env, &mut |row, col, col2, val| {
                    let key = (row.to_string(), col.to_string());
                    let mrc = match m.get(&key) {
                        Some(x) => *x,
                        None => {
                            let mut acc = 0.0;
                            for (mu, w) in pairs {
                                acc += get(mu, row, "μ")? * get(w, col, "w")?;
                            }
                            m.insert(key, acc);
                            acc
                        }
                    };
                    *g.entry(col2.to_string()).or_insert(0.0) += val * mrc;
                    Ok(())
                })
            }
            HvpForm::Symmetric => {
                if self.wrt != (Wrt::States, Wrt::States) {
                    return Err(BandError(format!(
                        "the symmetric form needs a (states, states) pair, got ({}, {})",
                        self.wrt.0.as_str(),
                        self.wrt.1.as_str()
                    )));
                }
                self.each_point(env, &mut |row, col, col2, val| {
                    let mut acc = 0.0;
                    for (mu, w) in pairs {
                        acc += get(mu, row, "μ")? * val * get(w, col2, "w")?;
                    }
                    *g.entry(col.to_string()).or_insert(0.0) += acc;
                    Ok(())
                })
            }
        }
    }
}

/// The second-order contraction at a point (spec §4.4), with no tensor
/// formed:
///
/// ```text
/// g += ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ          (pair (states, a2))
/// g += ∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t         (pair (time, a2); wᵢ binds "t")
/// ```
///
/// with `J = ∂f/∂(a1)`. `pairs` holds one `(μ, w)` per Rosenbrock stage,
/// `μ` keyed by row cell name and `w` by first-column cell name; a name a
/// point reads that they do not bind is an error. `g` is keyed by
/// second-column cell name (`"T[2]"`, `"k3"`, `"F[1,4]"`; on
/// [`HvpForm::Symmetric`] by first-column name), gains a key for every
/// column a band point reaches, and ACCUMULATES — clear it first for a
/// single product. Parameters `env` leaves unbound read the document
/// defaults. This prepares the evaluator for one call; a loop calls
/// [`HvpEvaluator::hvp`] on one [`prepare_hvp`] instead.
#[allow(clippy::too_many_arguments)]
pub fn hvp(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
    env: &Env,
    pairs: &[HvpPair],
    form: HvpForm,
    g: &mut BTreeMap<String, f64>,
) -> R<()> {
    prepare_hvp(file, model_name, wrt, forcing)?.hvp(env, pairs, form, g)
}

/// Execute a bound scatter plan (the consumer's step, spec §8): `out`
/// zeroed, then `out[pos] += fields[k][f]` in plan order.
pub fn scatter(b: &earthsci_astdiff::BoundScatter, fields: &[Vec<f64>], out: &mut [f64]) -> R<()> {
    if fields.len() != b.field_lens.len()
        || fields.iter().zip(&b.field_lens).any(|(f, n)| f.len() != *n)
    {
        return Err(BandError(
            "scatter: band fields do not match the plan's bands".into(),
        ));
    }
    if out.len() != b.len {
        return Err(BandError(format!(
            "scatter: output has length {}; the plan needs {}",
            out.len(),
            b.len
        )));
    }
    out.iter_mut().for_each(|x| *x = 0.0);
    for &(k, f, pos) in &b.pairs {
        out[pos] += fields[k][f];
    }
    Ok(())
}
