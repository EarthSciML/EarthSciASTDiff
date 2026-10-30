//! Numeric assembly — a correctness surface (like the Python binding's):
//! per-cell read resolution to literals, then the official evaluator.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

use earthsci_ast::evaluate;
use earthsci_ast::substitute;
use earthsci_ast::{EsmFile, Expr, VariableType};
use serde_json::Value;

use crate::bands::{Band, BandError, Ctx};
use crate::clip::RIdx;
use crate::helpers::map_expr_children;
use crate::inline::inline_observed_cached;
use crate::jacobian::{column_names, jacobian_bands_view, jacobian_columns, JacEntry, Wrt};
use crate::system::{ctx_of, observed_defs, sysview, SysView};

type R<T> = Result<T, BandError>;

/// `foreach_point`'s callback: `(row cell, contracted cell, index binding,
/// column cell)`.
type PointFn<'a> = dyn FnMut(&[i64], &[i64], &HashMap<String, i64>, &[i64]) -> R<()> + 'a;

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
            if !crate::rules::plain_sum(n) {
                return Err(BandError(format!(
                    "{} in scalar coefficient",
                    crate::rules::reduction(n)
                )));
            }
            let empty = HashMap::new();
            let ranges = n.ranges.as_ref().unwrap_or(&empty);
            let mut names: Vec<String> = ranges.keys().cloned().collect();
            names.sort();
            let boxes: Vec<(i64, i64)> = names
                .iter()
                .map(|nm| crate::bands::range_of(&ranges[nm], ctx))
                .collect::<Result<_, _>>()?;
            let body = n
                .expr
                .as_deref()
                .ok_or_else(|| BandError("faq without body".into()))?;
            let mut total = 0.0;
            let mut cell: Vec<i64> = boxes.iter().map(|(lo, _)| *lo).collect();
            'outer: loop {
                let binds: HashMap<String, Expr> = names
                    .iter()
                    .cloned()
                    .zip(cell.iter().map(|c| Expr::Integer(*c)))
                    .collect();
                total += as_lit(&resolve(&substitute(body, &binds), env, ctx)?)?;
                let mut k = 0;
                loop {
                    if k == cell.len() {
                        break 'outer;
                    }
                    cell[k] += 1;
                    if cell[k] <= boxes[k].1 {
                        break;
                    }
                    cell[k] = boxes[k].0;
                    k += 1;
                }
                if cell.is_empty() {
                    break;
                }
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

pub(crate) fn cellname(v: &str, cell: &[i64]) -> String {
    if cell.is_empty() {
        v.to_string()
    } else {
        format!(
            "{v}[{}]",
            cell.iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

fn fold_const_gathers(e: &Expr, bind: &HashMap<String, f64>) -> R<Expr> {
    let Expr::Operator(n) = e else {
        return Ok(e.clone());
    };
    if n.op == "index" {
        if let Some(Expr::Operator(cn)) = n.args.first() {
            if cn.op == "const" {
                let mut v = cn
                    .value
                    .as_ref()
                    .ok_or_else(|| BandError("const without value".into()))?;
                for x in &n.args[1..] {
                    let i = eval_cidx(x, bind)?;
                    v = v
                        .as_array()
                        .and_then(|a| a.get((i - 1) as usize))
                        .ok_or_else(|| BandError("out-of-range const gather".into()))?;
                }
                return Ok(match v {
                    Value::Number(num) if num.is_i64() => Expr::Integer(num.as_i64().unwrap()),
                    _ => Expr::Number(
                        v.as_f64()
                            .ok_or_else(|| BandError("non-numeric const gather".into()))?,
                    ),
                });
            }
        }
    }
    let mut err: Option<BandError> = None;
    let out = map_expr_children(e, &mut |x| match fold_const_gathers(x, bind) {
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

pub(crate) fn eval_cidx(c: &Expr, bind: &HashMap<String, f64>) -> R<i64> {
    let folded = fold_const_gathers(c, bind)?;
    evaluate(&folded, bind)
        .map(|v| v.round() as i64)
        .map_err(|er| BandError(format!("evaluate cidx: {er:?}")))
}

pub(crate) fn product_first_fastest(ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let mut cell: Vec<i64> = ranges.iter().map(|(lo, _)| *lo).collect();
    if ranges.is_empty() {
        return vec![vec![]];
    }
    loop {
        out.push(cell.clone());
        let mut k = 0;
        loop {
            if k == cell.len() {
                return out;
            }
            cell[k] += 1;
            if cell[k] <= ranges[k].1 {
                break;
            }
            cell[k] = ranges[k].0;
            k += 1;
        }
    }
}

/// Every point of one band: `f(row cell, contracted cell, index binding,
/// column cell)` per (row cell, contracted cell), first index fastest, with
/// the column index evaluated there (ghost reads included). The one point
/// walk shared by numeric assembly and the CSR scatter plan.
pub(crate) fn foreach_point(b: &Band, f: &mut PointFn<'_>) -> R<()> {
    let cranges: Vec<(i64, i64)> = b.contracted.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();
    for cell in product_first_fastest(&b.rows) {
        for cc in product_first_fastest(&cranges) {
            let mut ibind: HashMap<String, i64> = HashMap::new();
            for (k, r) in b.ridx.iter().enumerate() {
                if let RIdx::Name(nm) = r {
                    ibind.insert(nm.clone(), cell[k]);
                }
            }
            for (k, (nm, _, _)) in b.contracted.iter().enumerate() {
                ibind.insert(nm.clone(), cc[k]);
            }
            let fbind: HashMap<String, f64> =
                ibind.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
            let cidx: Vec<i64> = b
                .cidx
                .iter()
                .map(|c| eval_cidx(c, &fbind))
                .collect::<Result<_, _>>()?;
            f(&cell, &cc, &ibind, &cidx)?;
        }
    }
    Ok(())
}

/// Every non-ghost point of every band at a point: `f(k, entry, row cell,
/// contracted cell, column cell, value)`, bands in order, cells first index
/// fastest.
///
/// On the parameters and forcing axes (`wrt.has_columns()`, spec §4) the
/// ghost skip is
/// an error instead: a column there is a buffer cell, and an unresolvable
/// one would silently drop a gradient.
fn for_each_value(
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    entries: &[JacEntry],
    env: &Env,
    wrt: Wrt,
    f: &mut ValueFn<'_>,
) -> R<()> {
    // Pruning leaves target-independent observed as opaque reads in the
    // coefficients; resolve them from their own definitions (what the
    // reference's derived evaluation model does). This splices in only the
    // observed a coefficient actually reads — exactly what evaluating it at
    // a point needs — so it rejects nothing an evaluation could do without.
    let mut obs_names = None;
    for (k, en) in entries.iter().enumerate() {
        let coef0 = inline_observed_cached(&en.band.coef, obs, &mut obs_names)
            .map_err(|e| BandError(e.0))?;
        let vshape = ctx.shapes.get(&en.v);
        foreach_point(&en.band, &mut |cell, cc, ibind, cidx| {
            if let Some(vshape) = vshape {
                if cidx.iter().zip(vshape.iter()).any(|(c, s)| *c < 1 || c > s) {
                    if wrt.has_columns() {
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
            let coef = if ibind.is_empty() {
                coef0.clone()
            } else {
                let sub: HashMap<String, Expr> = ibind
                    .iter()
                    .map(|(k, v)| (k.clone(), Expr::Integer(*v)))
                    .collect();
                substitute(&coef0, &sub)
            };
            let val = as_lit(&resolve(&coef, env, ctx)?)?;
            f(k, en, cell, cc, cidx, val);
            Ok(())
        })?;
    }
    Ok(())
}

/// The environment with every parameter `env.params` leaves unbound filled
/// from the document's `default` (inline row-major nested arrays for an array
/// parameter or forcing buffer, spec §4.3): what the reference's assembly
/// reads when no buffer is bound.
fn with_defaults<'e>(sv: &SysView, env: &'e Env) -> Cow<'e, Env> {
    let missing: Vec<(&String, Value)> = sv
        .variables
        .iter()
        .filter(|(n, v)| v.var_type == VariableType::Parameter && !env.params.contains_key(*n))
        .filter_map(|(n, v)| {
            let d = v.default.as_ref()?;
            Some((n, serde_json::to_value(d).ok()?))
        })
        .collect();
    if missing.is_empty() {
        return Cow::Borrowed(env);
    }
    let mut params = env.params.clone();
    for (n, v) in missing {
        params.insert(n.clone(), v);
    }
    Cow::Owned(Env {
        states: env.states.clone(),
        params,
        t: env.t,
    })
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
/// [`crate::jacobian_columns`] (`"E[1]"`, `"F[2,3]"`, scalars bare), a
/// column outside its variable's extent is an error (no ghost skip), and an
/// array parameter or forcing buffer `env.params` does not bind reads the
/// document's `default`.
pub fn assemble_jacobian_with_forcing(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    env: &Env,
) -> R<BTreeMap<(String, String), f64>> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    let entries = jacobian_bands_view(&sv, &ctx, &obs, wrt, forcing)?;
    let env = with_defaults(&sv, env);
    let mut out: BTreeMap<(String, String), f64> = BTreeMap::new();
    for_each_value(
        &ctx,
        &obs,
        &entries,
        &env,
        wrt,
        &mut |_, en, cell, _, cidx, val| {
            let key = (cellname(&en.u, cell), cellname(&en.v, cidx));
            *out.entry(key).or_insert(0.0) += val;
        },
    )?;
    Ok(out)
}

/// The transposed product `ḡ += Jᵀ λ` at a point, as a scatter-add over the
/// band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix
/// (and no transpose) formed. `lambda` is keyed by row cell name (`"c[2]"`,
/// `"S"`); a row it does not bind is an error. `gbar` is indexed by the
/// canonical column ordinal of a parameters or forcing axis
/// ([`crate::jacobian::column_names`] of [`crate::jacobian_columns`]) and
/// ACCUMULATES — zero it first for a single product.
pub fn vjp(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    env: &Env,
    lambda: &HashMap<String, f64>,
    gbar: &mut [f64],
) -> R<()> {
    let sv = sysview(file, model_name)?;
    let cols = jacobian_columns(&sv, wrt, forcing)?;
    let names = column_names(&cols);
    if gbar.len() != names.len() {
        return Err(BandError(format!(
            "ḡ has length {}, the column space {}",
            gbar.len(),
            names.len()
        )));
    }
    let slot: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    let entries = jacobian_bands_view(&sv, &ctx, &obs, wrt, forcing)?;
    let env = with_defaults(&sv, env);
    let mut err: Option<BandError> = None;
    for_each_value(
        &ctx,
        &obs,
        &entries,
        &env,
        wrt,
        &mut |_, en, cell, _, cidx, val| {
            if err.is_some() {
                return;
            }
            let (r, c) = (cellname(&en.u, cell), cellname(&en.v, cidx));
            match (lambda.get(&r), slot.get(c.as_str())) {
                (Some(l), Some(&k)) => gbar[k] += val * l,
                (None, _) => err = Some(BandError(format!("λ binds no row `{r}`"))),
                (_, None) => {
                    err = Some(BandError(format!(
                        "column `{c}` is not in the column space"
                    )))
                }
            }
        },
    )?;
    err.map_or(Ok(()), Err)
}

/// [`vjp`] on the forcing axis, into buffer-shaped accumulators: the forcing
/// cotangent of one stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (spec §4.3). `gbar`
/// holds one accumulator per forcing buffer, keyed by buffer name, each the
/// buffer's cells COLUMN-MAJOR (first index fastest). It accumulates; which
/// dataset record a buffer slot holds, and when to flush, is the caller's
/// bookkeeping.
pub fn vjp_forcing(
    file: &EsmFile,
    model_name: Option<&str>,
    forcing: Option<&[String]>,
    env: &Env,
    lambda: &HashMap<String, f64>,
    gbar: &mut HashMap<String, Vec<f64>>,
) -> R<()> {
    let sv = sysview(file, model_name)?;
    let cols = jacobian_columns(&sv, Wrt::Forcing, forcing)?;
    let mut flat = vec![0.0; column_names(&cols).len()];
    vjp(
        file,
        model_name,
        Wrt::Forcing,
        forcing,
        env,
        lambda,
        &mut flat,
    )?;
    let mut off = 0;
    for (name, shape) in &cols {
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

/// The band fields at a point — what the derived evaluation document's
/// `J_k` states hold, and what a [`crate::ScatterPlan`] scatters: field `k`
/// over the row shape then the contracted extents, column-major. Ghost
/// points (no column) and cells outside the band region stay 0.
pub fn band_fields(file: &EsmFile, model_name: Option<&str>, env: &Env) -> R<Vec<Vec<f64>>> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    let entries = jacobian_bands_view(&sv, &ctx, &obs, Wrt::States, None)?;
    let shapes: Vec<Vec<i64>> = entries
        .iter()
        .map(|en| crate::scatter::field_shape(en, &ctx))
        .collect();
    let mut fields: Vec<Vec<f64>> = shapes
        .iter()
        .map(|sh| vec![0.0; sh.iter().product::<i64>() as usize])
        .collect();
    let env = with_defaults(&sv, env);
    for_each_value(
        &ctx,
        &obs,
        &entries,
        &env,
        Wrt::States,
        &mut |k, _, cell, cc, _, val| {
            let full: Vec<i64> = cell.iter().chain(cc.iter()).copied().collect();
            fields[k][crate::scatter::flat(&full, &shapes[k])] = val;
        },
    )?;
    Ok(fields)
}
