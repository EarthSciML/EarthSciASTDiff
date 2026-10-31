//! Values from documents: the crate evaluates nothing, so the tests get every
//! number by running one of its documents through EarthSciAST's interpreter —
//! `J` column by column from the `JvpEval` product document (spec §9.2),
//! `Jᵀ λ` from `VjpEval`, the second-order contraction from `HvpEval` and
//! the band fields from the evaluation document. Mirrors the reference's
//! test/docvals.jl. Test code may execute a scatter plan (the consumer's
//! job); [`scatter`] does.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::{esm_problem, rhs_program, Compiler, EsmFile, ProblemOptions, RhsProgram};
use earthsci_astdiff::bands::BandError;
use earthsci_astdiff::points::{cellname, check_state_column, foreach_point, outside};
use earthsci_astdiff::system::{ctx_of, observed_defs, sysview, view_states, SysView};
use earthsci_astdiff::{
    column_names, evaluation_document, extended_bands, hvp_document, jacobian_bands2_with_forcing,
    jacobian_bands_view, jacobian_columns, jvp_document, vjp_document, BoundScatter, HvpForm,
    IoMap, JacEntry, ProductOptions, Wrt,
};
use serde_json::Value;

type R<T> = Result<T, BandError>;

/// A point: state values (name → nested array, row-major with the first
/// index outermost, or a scalar), scalar parameter overrides (a parameter
/// not bound reads its document default) and the time.
#[derive(Clone, Debug)]
pub struct Env {
    pub states: HashMap<String, Value>,
    pub params: HashMap<String, Value>,
    pub t: f64,
}

/// One stage pair `(μ, w)` of [`doc_hvp`]: `μ` keyed by row cell name, `w`
/// by first-column cell name (on a time pair, the single column `"t"`).
pub type HvpPair = (HashMap<String, f64>, HashMap<String, f64>);

/// A cell name (`"c[2]"`, `"S"`) split into its variable and 1-based index.
pub fn split_cell(cell: &str) -> (&str, Vec<usize>) {
    match cell.find('[') {
        None => (cell, Vec::new()),
        Some(k) => (
            &cell[..k],
            cell[k + 1..cell.len() - 1]
                .split(',')
                .map(|x| x.parse().unwrap())
                .collect(),
        ),
    }
}

/// The document cell carrying `cell` of a variable `names` maps.
pub fn doc_cell(names: &IoMap, cell: &str) -> String {
    let (v, _) = split_cell(cell);
    format!("{}{}", names[v], &cell[v.len()..])
}

/// Every cell of a nested-array value, as `(1-based index, value)`.
fn cells_of(v: &Value, idx: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, f64)>) {
    match v {
        Value::Array(a) => {
            for (k, x) in a.iter().enumerate() {
                idx.push(k + 1);
                cells_of(x, idx, out);
                idx.pop();
            }
        }
        x => out.push((idx.clone(), x.as_f64().expect("a numeric state value"))),
    }
}

/// An EarthSciAST evaluator of a derived document, at a point.
pub struct DocEval {
    prog: RhsProgram,
    /// state name (model prefix stripped) → (offset, column-major shape)
    blocks: HashMap<String, (usize, Vec<usize>)>,
    x0: Vec<f64>,
    aux: Vec<f64>,
    p: Vec<f64>,
    t: f64,
}

impl DocEval {
    /// Build `doc`'s right-hand side with the interpreter (the native
    /// compiler refuses an equality-filtered `faq` until it implements
    /// §9.3's scatter, and some coefficient forms) and bind `env`.
    pub fn new(doc: &Value, model: &str, env: &Env) -> Self {
        let file = earthsci_ast::load_document(doc).expect("a derived document loads");
        let opts = ProblemOptions {
            compiler: Some(Compiler::Interpreter),
            ..Default::default()
        };
        let prog = rhs_program(&esm_problem(&file, (0.0, 1.0), opts).unwrap()).unwrap();
        let blocks = prog
            .state_layout()
            .iter()
            .map(|b| {
                let nm = b.name.strip_prefix(&format!("{model}.")).unwrap_or(&b.name);
                (nm.to_string(), (b.flat_offset, b.shape.clone()))
            })
            .collect();
        let (x0, aux) = prog.default_state();
        let declared = &doc["models"][model]["variables"];
        let mut overrides = HashMap::new();
        for (k, v) in &env.params {
            let x = v.as_f64().expect("a scalar parameter override");
            if let Some(s) = prog
                .parameters()
                .iter()
                .find(|s| s.name == format!("{model}.{k}") || s.name == *k)
            {
                overrides.insert(s.name.clone(), x);
            } else {
                // a parameter the derived document declares but does not
                // read is not in `p`; a name it does not declare at all is a
                // misspelled or stale override, never silently dropped
                assert!(
                    declared.get(k).is_some(),
                    "parameter override `{k}` names no variable of the derived document"
                );
            }
        }
        let p = prog.parameter_vector(&overrides).unwrap();
        let mut ev = DocEval {
            prog,
            blocks,
            x0,
            aux,
            p,
            t: env.t,
        };
        for (name, v) in &env.states {
            let mut cells = Vec::new();
            cells_of(v, &mut Vec::new(), &mut cells);
            for (idx, x) in cells {
                let k = ev.slot(name, &idx);
                ev.x0[k] = x;
            }
        }
        ev
    }

    /// The slot of cell `idx` of state `v` (column-major).
    fn slot(&self, v: &str, idx: &[usize]) -> usize {
        let (off, shape) = self
            .blocks
            .get(v)
            .unwrap_or_else(|| panic!("no state `{v}` in the document"));
        let (mut k, mut stride) = (0, 1);
        for (i, n) in idx.iter().zip(shape) {
            k += (i - 1) * stride;
            stride *= n;
        }
        off + k
    }

    fn cell_slot(&self, cell: &str) -> usize {
        let (v, idx) = split_cell(cell);
        self.slot(v, &idx)
    }

    /// One right-hand-side call with `inputs` (document cell → value) written
    /// into their slots; `du` by document cell name.
    pub fn call(&self, inputs: &HashMap<String, f64>) -> Du<'_> {
        let mut x = self.x0.clone();
        for (c, v) in inputs {
            x[self.cell_slot(c)] = *v;
        }
        let mut du = vec![0.0; x.len()];
        let mut ws = self.prog.new_workspace();
        self.prog
            .eval_into(
                &x,
                &self.aux,
                &self.p,
                self.t,
                &self.prog.default_forcing(),
                &mut du,
                &mut ws,
            )
            .unwrap();
        Du { ev: self, du }
    }
}

/// The derivatives of one [`DocEval::call`].
pub struct Du<'a> {
    ev: &'a DocEval,
    du: Vec<f64>,
}

impl Du<'_> {
    pub fn at(&self, cell: &str) -> f64 {
        self.du[self.ev.cell_slot(cell)]
    }

    /// The whole block of state `v`, column-major.
    pub fn block(&self, v: &str) -> &[f64] {
        let (off, shape) = &self.ev.blocks[v];
        &self.du[*off..off + shape.iter().product::<usize>().max(1)]
    }
}

/// Every cell name of variable `v` of shape `shape`, first index fastest.
fn cells(v: &str, shape: &[i64]) -> Vec<String> {
    column_names(&[(v.to_string(), shape.to_vec())])
}

/// The structural pattern of `entries` on axis `wrt`: every (row cell,
/// column cell) a band point reaches, ghost reads dropped. A state column
/// naming no state is refused.
pub fn pattern(sv: &SysView, entries: &[JacEntry], wrt: Wrt) -> R<HashSet<(String, String)>> {
    let ctx = ctx_of(sv)?;
    let states: HashSet<String> = view_states(sv, &observed_defs(sv)?).into_iter().collect();
    let mut out = HashSet::new();
    for en in entries {
        if wrt == Wrt::States {
            check_state_column(&states, &en.v, &en.u)?;
        }
        let shape = ctx.shapes.get(&en.v);
        foreach_point(&en.band, &mut |cell, _, _, cidx| {
            if !outside(shape, cidx) {
                out.insert((cellname(&en.u, cell), cellname(&en.v, cidx)));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// `J` at `env` on axis `wrt`, keyed `(row cell, column cell)` over exactly
/// the structural [`pattern`]: column `c` read off one `JvpEval` call at
/// the unit direction `e_c`, for every column cell of the axis. A nonzero
/// off the pattern panics, as does a pattern pair the axis has no row or
/// column for.
pub fn doc_jacobian(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    env: &Env,
) -> R<BTreeMap<(String, String), f64>> {
    let sv = sysview(file, model_name)?;
    let entries = jacobian_bands_view(&sv, wrt, forcing)?;
    doc_jacobian_entries(&sv, &entries, wrt, forcing, env)
}

/// [`doc_jacobian`] of given entries.
pub fn doc_jacobian_entries(
    sv: &SysView,
    entries: &[JacEntry],
    wrt: Wrt,
    forcing: Option<&[String]>,
    env: &Env,
) -> R<BTreeMap<(String, String), f64>> {
    let pat = pattern(sv, entries, wrt)?;
    let opts = ProductOptions {
        forcing,
        ..Default::default()
    };
    let (doc, io) = jvp_document(sv, entries, wrt, &opts)?;
    let ev = DocEval::new(&doc, &io.model, env);
    let ctx = ctx_of(sv)?;
    let rows: Vec<String> = io
        .jvp
        .keys()
        .flat_map(|r| cells(r, ctx.shapes.get(r).map_or(&[], |s| s.as_slice())))
        .collect();
    let mut out = BTreeMap::new();
    for c in &column_cells(sv, wrt, forcing)? {
        let du = ev.call(&HashMap::from([(doc_cell(&io.v, c), 1.0)]));
        for r in &rows {
            let x = du.at(&doc_cell(&io.jvp, r));
            let key = (r.clone(), c.clone());
            if pat.contains(&key) {
                out.insert(key, x);
            } else {
                assert_eq!(x, 0.0, "J[{r}, {c}] = {x} lies off the structural pattern");
            }
        }
    }
    assert_eq!(
        out.len(),
        pat.len(),
        "a pattern pair names no row or column of the axis"
    );
    Ok(out)
}

/// `d` binds exactly `names`: a missing or misspelled cell is an error.
fn bound(d: &HashMap<String, f64>, names: &[String], what: &str) -> R<()> {
    let want: HashSet<&str> = names.iter().map(String::as_str).collect();
    let got: HashSet<&str> = d.keys().map(String::as_str).collect();
    if want != got {
        let mut miss: Vec<&&str> = want.difference(&got).collect();
        let mut extra: Vec<&&str> = got.difference(&want).collect();
        miss.sort();
        extra.sort();
        return Err(BandError(format!(
            "{what} must bind exactly the axis's cells: missing {miss:?}, unknown {extra:?}"
        )));
    }
    Ok(())
}

/// The column cells of axis `wrt`: every state cell, `["t"]`, or the
/// canonical column space.
pub fn column_cells(sv: &SysView, wrt: Wrt, forcing: Option<&[String]>) -> R<Vec<String>> {
    Ok(match wrt {
        Wrt::Time => vec!["t".into()],
        Wrt::States => {
            let ctx = ctx_of(sv)?;
            view_states(sv, &observed_defs(sv)?)
                .iter()
                .flat_map(|v| cells(v, ctx.shapes.get(v).map_or(&[], |s| s.as_slice())))
                .collect()
        }
        _ => column_names(&jacobian_columns(sv, wrt, forcing)?),
    })
}

/// `Jᵀ λ` from one `VjpEval` call, over every column cell of the axis;
/// `lambda` binds exactly every row (state) cell: a missing or misspelled
/// cell is an error, never a zero.
pub fn doc_vjp(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    env: &Env,
    lambda: &HashMap<String, f64>,
) -> R<BTreeMap<String, f64>> {
    let sv = sysview(file, model_name)?;
    let entries = jacobian_bands_view(&sv, wrt, forcing)?;
    let opts = ProductOptions {
        forcing,
        ..Default::default()
    };
    let (doc, io) = vjp_document(&sv, &entries, wrt, &opts)?;
    bound(lambda, &column_cells(&sv, Wrt::States, None)?, "λ")?;
    let ev = DocEval::new(&doc, &io.model, env);
    let inputs = lambda
        .iter()
        .map(|(r, x)| (doc_cell(&io.lam, r), *x))
        .collect();
    let du = ev.call(&inputs);
    Ok(column_cells(&sv, wrt, forcing)?
        .into_iter()
        .map(|c| {
            let x = du.at(&doc_cell(&io.vjp, &c));
            (c, x)
        })
        .collect())
}

/// The §4.4 contraction from one `HvpEval` (`HvpEvalSymmetric`) call, keyed
/// by the second-axis column cells a band point reaches (the structural key
/// set); a nonzero elsewhere panics. Each stage's `μ` binds exactly every
/// state cell and its `w` every first-axis column cell (`"t"` on a time
/// pair): a missing or misspelled cell is an error, never a zero.
pub fn doc_hvp(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
    env: &Env,
    pairs: &[HvpPair],
    form: HvpForm,
) -> R<BTreeMap<String, f64>> {
    let sv = sysview(file, model_name)?;
    let entries = jacobian_bands2_with_forcing(file, model_name, wrt, forcing)?;
    let ctx = ctx_of(&sv)?;
    // the second columns some point reads with both columns in extent
    let mut reach = HashSet::new();
    for en in &entries {
        let (s1, s2) = (ctx.shapes.get(&en.v), ctx.shapes.get(&en.w));
        foreach_point(&en.band, &mut |_, _, _, cidx| {
            let (c1, c2) = cidx.split_at(en.ncol);
            if !outside(s1, c1) && !outside(s2, c2) {
                reach.insert(cellname(&en.w, c2));
            }
            Ok(())
        })?;
    }
    // through a recurrence (spec §4.5) the second axis's recurrence rows
    // carry the contraction on to their columns
    let e2 = extended_bands(&sv, wrt.1, forcing, Some(wrt.0))?;
    for en in e2.rec_rows.iter().filter(|en| !e2.is_recurrence(&en.v)) {
        let s = ctx.shapes.get(&en.v);
        foreach_point(&en.band, &mut |_, _, _, cidx| {
            if !outside(s, cidx) {
                reach.insert(cellname(&en.v, cidx));
            }
            Ok(())
        })?;
    }
    let opts = ProductOptions {
        forcing,
        ..Default::default()
    };
    let (doc, io) = hvp_document(&sv, &entries, wrt, pairs.len(), form, &opts)?;
    let rows = column_cells(&sv, Wrt::States, None)?;
    let cols1 = column_cells(&sv, wrt.0, forcing)?;
    for (mu, w) in pairs {
        bound(mu, &rows, "μ")?;
        bound(w, &cols1, "w")?;
    }
    let ev = DocEval::new(&doc, &io.model, env);
    let mut inputs = HashMap::new();
    for (i, (mu, w)) in pairs.iter().enumerate() {
        for (r, x) in mu {
            inputs.insert(doc_cell(&io.mu[i], r), *x);
        }
        for (c, x) in w {
            inputs.insert(doc_cell(&io.w[i], c), *x);
        }
    }
    let du = ev.call(&inputs);
    let mut out = BTreeMap::new();
    for c in column_cells(&sv, wrt.1, forcing)? {
        let x = du.at(&doc_cell(&io.hvp, &c));
        if reach.contains(&c) {
            out.insert(c, x);
        } else {
            assert_eq!(x, 0.0, "g[{c}] = {x} lies off the structural key set");
        }
    }
    Ok(out)
}

/// The band fields at a point — the derived evaluation document's `J_k`
/// derivatives (each over the row shape then the contracted extents,
/// column-major), what a [`earthsci_astdiff::ScatterPlan`] scatters.
pub fn doc_band_fields(file: &EsmFile, model_name: Option<&str>, env: &Env) -> R<Vec<Vec<f64>>> {
    let sv = sysview(file, model_name)?;
    let entries = jacobian_bands_view(&sv, Wrt::States, None)?;
    let (doc, names) = evaluation_document(&sv, &entries, true, 12)?;
    let ev = DocEval::new(&doc, "JacobianEval", env);
    let du = ev.call(&HashMap::new());
    Ok(names.iter().map(|n| du.block(n).to_vec()).collect())
}

/// Execute a bound scatter plan: `out[pos] = Σ fields[k][cell]` over its
/// `(k, cell, pos)` pairs, every other slot zero. The consumer's job, done here to check a
/// plan by value.
pub fn scatter(b: &BoundScatter, fields: &[Vec<f64>], out: &mut [f64]) -> R<()> {
    if fields.len() != b.field_lens.len()
        || fields.iter().zip(&b.field_lens).any(|(f, n)| f.len() != *n)
    {
        return Err(BandError(
            "scatter: band fields do not match the plan's bands".into(),
        ));
    }
    if out.len() != b.len {
        return Err(BandError(format!(
            "scatter: output has length {}, the plan {}",
            out.len(),
            b.len
        )));
    }
    out.fill(0.0);
    for &(k, f, pos) in &b.pairs {
        out[pos] += fields[k][f];
    }
    Ok(())
}
