//! Production-rate assembly (#4 W2, W4): the derived evaluation document
//! compiled once through earthsci-ast-rs's own programs, its band fields
//! scattered by a [`ScatterPlan`] straight into a consumer's storage, and the
//! matrix-free products over the same band slab.
//!
//! * [`JacobianProgram::build`] differentiates once and compiles the trimmed
//!   evaluation document ([`crate::evaluation_document`], the `*.evaldoc.json`
//!   golden) with the `native` compiler, `select`ed to write only the band
//!   fields `J_k`: the source states become read-only `aux` inputs and no
//!   source `D` equation is evaluated.
//! * [`JacobianProgram::plan`] pairs a storage layout with an executor —
//!   [`Exec::Lanes`] (a lane program over run-time cells or columns, RFC U3
//!   §6) or [`Exec::Domain`] (the whole-domain program) — and checks every
//!   name the call will read against the integrator's own program for the
//!   same operator: states, parameters and forcing slots must all resolve.
//! * [`JacobianProgram::eval_into`] evaluates the band slab into a
//!   [`JacWorkspace`] and scatters it, `+=` in plan order into zeroed storage;
//!   [`JacobianProgram::apply`] and [`JacobianProgram::apply_transpose`] walk
//!   the slab the last `eval_into` left.
//!
//! `JacobianProgram` and `JacPlan` are immutable, `Send + Sync` and cheap to
//! clone; a `JacWorkspace` is `Send`, not `Sync`, one per thread. A lane's
//! result depends only on its own inputs and the plan order (no atomics, no
//! thread-local partial sums), so assembly is bitwise invariant to thread
//! count and lane position. The oracle ([`crate::assemble_jacobian`]) stays
//! the reference this path is tested against.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use earthsci_ast::{
    esm_problem, rhs_program, Compiler, EsmFile, ForcingSet, LaneKind, LaneProgram,
    ProblemOptions, RhsError, RhsProgram, RhsSelection, RhsWorkspace,
};
use serde_json::{json, Map, Value};

use crate::bands::BandError;
use crate::clip::RIdx;
use crate::emit::evaluation_document;
use crate::jacobian::{jacobian_bands_view, JacEntry, Wrt};
use crate::scatter::{ncells, Layout, PlanKind, ScatterPlan};
use crate::system::{ctx_of, observed_defs, sysview};

/// Why a program, plan or call was refused.
#[derive(Debug)]
pub enum JacError {
    /// The band calculus or the scatter plan refused the document.
    Band(BandError),
    /// The evaluation document did not compile.
    Build(String),
    /// A plan's layout, executor or source program does not fit.
    Plan(String),
    /// A call's arguments do not fit its plan or workspace.
    Call(String),
    /// The compiled program refused the call (an evaluation fault, a forcing
    /// set for other slots, …).
    Rhs(RhsError),
}

impl fmt::Display for JacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JacError::Band(e) => write!(f, "{}", e.0),
            JacError::Build(m) => write!(f, "build: {m}"),
            JacError::Plan(m) => write!(f, "plan: {m}"),
            JacError::Call(m) => write!(f, "call: {m}"),
            JacError::Rhs(e) => write!(f, "evaluation: {e}"),
        }
    }
}

impl std::error::Error for JacError {}

impl From<BandError> for JacError {
    fn from(e: BandError) -> Self {
        JacError::Band(e)
    }
}

impl From<RhsError> for JacError {
    fn from(e: RhsError) -> Self {
        JacError::Rhs(e)
    }
}

type R<T> = Result<T, JacError>;

/// Options for [`JacobianProgram::build`].
#[derive(Clone, Debug)]
pub struct BuildOptions {
    /// Hoist repeated coefficient subtrees into observed of the evaluation
    /// document ([`crate::hoist_observed`]).
    pub cse: bool,
    /// The smallest subtree (in nodes) hoisting considers.
    pub cse_min_nodes: usize,
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions { cse: true, cse_min_nodes: 12 }
    }
}

/// How a plan evaluates the band fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exec {
    /// The whole-domain program: `u`/`aux` in the source program's
    /// [`RhsProgram::layout`]. Every layout; lane layouts get one lane per
    /// lane-domain cell (column-major), their capacity the lane count.
    Domain,
    /// A lane program of `capacity` lanes over run-time `cells` (RFC U3 §6):
    /// `u`/`aux` in the lane layout of the source program laned the same way.
    /// `LaneBlocks` and `LaneSparseBlocks` lane over cells, `ColumnBanded`
    /// over columns; `Csr` has no lane form.
    Lanes { capacity: usize },
}

/// The scalar a plan stores into (`f64`, or `f32` Jacobian storage). Band
/// fields are evaluated in `f64`; each output element is summed in `f64`
/// and rounded once.
pub trait StorageScalar: Copy + Send + Sync + 'static {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl StorageScalar for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl StorageScalar for f32 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// A model's state Jacobian, differentiated and compiled once.
#[derive(Clone)]
pub struct JacobianProgram {
    inner: Arc<ProgInner>,
}

struct ProgInner {
    id: u64,
    file: EsmFile,
    model_name: Option<String>,
    /// The model key: the flattened names' prefix (`<model>.<name>`).
    model: String,
    entries: Vec<JacEntry>,
    evaldoc: Value,
    /// The evaluation document's program, selected to the band fields.
    rhs: RhsProgram,
    /// The document as emitted, and its program, compiled on the first lane
    /// plan (see [`fused_band`]).
    lane_doc: Value,
    lane_rhs: OnceLock<Result<RhsProgram, String>>,
    names: Vec<String>,
    /// The index-set names of each state's shape.
    dims: HashMap<String, Vec<String>>,
}

impl fmt::Debug for JacobianProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JacobianProgram")
            .field("model", &self.inner.model)
            .field("bands", &self.inner.entries.len())
            .finish()
    }
}

fn the_model(file: &EsmFile, model_name: Option<&str>) -> R<String> {
    if let Some(n) = model_name {
        return Ok(n.to_string());
    }
    match &file.models {
        Some(m) if m.len() == 1 => Ok(m.keys().next().expect("one model").clone()),
        _ => Err(JacError::Build("the document has several models (or none); name one".into())),
    }
}

/// The document that is compiled: the source document (so data loaders and
/// forcing declarations stay what the integrator's own program sees) with
/// `model`'s variables and equations replaced by the evaluation document's,
/// and the zeroed source `D` equations dropped (the states stay read-only
/// inputs; a `D(u) = 0` would make `select` keep writing `u`). Source
/// variables keep their source declarations, so a forcing parameter is a
/// forcing slot here too. The model keeps its name, so the flattened names
/// equal the source program's.
fn compiled_document(file: &EsmFile, model: &str, evaldoc: &Value, names: &[String],
                     entries: &[JacEntry], fuse: bool)
                     -> R<Value> {
    let text = earthsci_ast::to_json(file)
        .map_err(|e| JacError::Build(format!("source document does not serialize: {e}")))?;
    let mut src: Value = serde_json::from_str(&text)
        .map_err(|e| JacError::Build(format!("source document does not serialize: {e}")))?;
    let derived = &evaldoc["models"]["JacobianEval"];
    let src_vars = src["models"][model]["variables"].as_object().cloned().unwrap_or_default();
    let mut vars = Map::new();
    for (n, d) in derived["variables"].as_object().into_iter().flatten() {
        vars.insert(n.clone(), src_vars.get(n).cloned().unwrap_or_else(|| d.clone()));
    }
    let bands: HashMap<&str, &JacEntry> =
        names.iter().map(String::as_str).zip(entries).collect();
    let eqs: Vec<Value> = derived["equations"].as_array().into_iter().flatten()
        .filter(|eq| {
            let lhs = &eq["lhs"];
            lhs.get("op").and_then(Value::as_str) != Some("D")
                || lhs["args"][0].as_str().is_some_and(|u| bands.contains_key(u))
        })
        .flat_map(|eq| match lhs_band(eq, &bands) {
            Some((u, en)) if fuse => fused_band(u, en, eq),
            _ => vec![eq.clone()],
        })
        .collect();
    let Value::Object(top) = &mut src else {
        return Err(JacError::Build("source document is not an object".into()));
    };
    let mut isets = top.get("index_sets").and_then(Value::as_object).cloned().unwrap_or_default();
    for (n, s) in evaldoc["index_sets"].as_object().into_iter().flatten() {
        isets.entry(n.clone()).or_insert_with(|| s.clone());
    }
    top.insert("index_sets".into(), Value::Object(isets));
    top.insert("models".into(), json!({model: {"variables": vars, "equations": eqs}}));
    top.remove("coupling");
    top.remove("reaction_systems");
    // The band coefficients may carry `faq` nodes (esm 1.1.0).
    let v = top.get("esm").and_then(Value::as_str).unwrap_or("1.1.0");
    let mut parts = v.split('.').map(|x| x.parse::<u64>().unwrap_or(0));
    if (parts.next().unwrap_or(0), parts.next().unwrap_or(0)) < (1, 1) {
        top.insert("esm".into(), json!("1.1.0"));
    }
    Ok(src)
}

/// The band entry a `D(J_k)` equation defines.
fn lhs_band<'a>(eq: &'a Value, bands: &HashMap<&str, &'a JacEntry>)
                -> Option<(&'a str, &'a JacEntry)> {
    let lhs = &eq["lhs"];
    if lhs.get("op").and_then(Value::as_str) != Some("D") {
        return None;
    }
    let u = lhs["args"][0].as_str()?;
    bands.get(u).map(|en| (u, *en))
}

/// A band field's `D(J) = makearray(0, region ← faq(coef))` in the form
/// the compiler fuses like any whole-array right-hand side: `D(J) =
/// faq(coef)` for a region spanning the field (the band program 1.5× faster
/// on `bd_chem`), else range-restricted equations (~12% on a column
/// operator) —
/// `faq{D(index(J, …))} = faq(coef)` over the region and `= 0` over the
/// boxes tiling the rest. The same values. A lane program restricts neither
/// form, so lane plans compile the document as emitted; a region fixed along
/// a row dimension stays as emitted too.
fn fused_band(name: &str, en: &JacEntry, eq: &Value) -> Vec<Value> {
    let rhs = &eq["rhs"];
    let value = &rhs["values"][1];
    let bounds = |r: &Value| -> Option<Vec<(i64, i64)>> {
        r.as_array()?.iter().map(|b| Some((b[0].as_i64()?, b[1].as_i64()?))).collect()
    };
    let (Some(full), Some(region)) =
        (bounds(&rhs["regions"][0]), bounds(&rhs["regions"][1])) else {
        return vec![eq.clone()];
    };
    if rhs.get("op").and_then(Value::as_str) != Some("makearray")
        || value.get("op").and_then(Value::as_str) != Some("faq")
    {
        return vec![eq.clone()];
    }
    if full == region {
        return vec![json!({"lhs": eq["lhs"], "rhs": value})];
    }
    if en.band.ridx.iter().any(|r| matches!(r, RIdx::Fixed(_))) {
        return vec![eq.clone()];
    }
    let idx: Vec<Value> = en.band.ridx.iter().filter_map(|r| match r {
        RIdx::Name(s) => Some(json!(s)),
        RIdx::Fixed(_) => None,                      // refused above
    }).chain(en.band.contracted.iter().map(|(n, _, _)| json!(n))).collect();
    let lhs = |ranges: &Value| json!({
        "op": "faq", "args": [], "output_idx": idx, "ranges": ranges,
        "expr": {"op": "D", "wrt": "t", "args": [{"op": "index", "args":
                 std::iter::once(json!(name)).chain(idx.iter().cloned()).collect::<Vec<_>>()}]}});
    let mut out = vec![json!({"lhs": lhs(&value["ranges"]), "rhs": value})];
    // The complement, dimension by dimension: inside the region before `m`,
    // below or above it along `m`, the whole field after.
    for m in 0..full.len() {
        let (lo, hi) = region[m];
        for side in [(full[m].0, lo - 1), (hi + 1, full[m].1)] {
            if side.0 > side.1 {
                continue;
            }
            let ranges: Map<String, Value> = idx.iter().enumerate().map(|(d, n)| {
                let r = if d < m { region[d] } else if d == m { side } else { full[d] };
                (n.as_str().expect("index names").to_string(), json!([r.0, r.1]))
            }).collect();
            let ranges = Value::Object(ranges);
            out.push(json!({"lhs": lhs(&ranges), "rhs": {
                "op": "faq", "args": [], "output_idx": idx, "ranges": ranges, "expr": 0.0}}));
        }
    }
    out
}

/// The band-field program of a compiled document: `native`, selected to
/// write the band fields `<model>.J_k` only.
fn compile(doc: &Value, model: &str, names: &[String]) -> R<RhsProgram> {
    let popts = ProblemOptions { compiler: Some(Compiler::Native), ..Default::default() };
    let prob = esm_problem(doc, (0.0, 1.0), popts)
        .map_err(|e| JacError::Build(format!("evaluation document: {e}")))?;
    let whole = rhs_program(&prob)
        .map_err(|e| JacError::Build(format!("evaluation document: {e}")))?;
    let writes = names.iter().map(|n| format!("{model}.{n}")).collect();
    whole.select(&RhsSelection { writes: Some(writes), terms: None })
        .map_err(|e| JacError::Build(format!("band-field selection: {e}")))
}

impl JacobianProgram {
    /// Differentiate `model_name` (the only model when `None`) with respect
    /// to its states and compile the band fields. Refuses an empty band list:
    /// the calculus found no coupling at all, which for an operator with
    /// array states is the silent all-zero failure the plans also refuse.
    pub fn build(file: &EsmFile, model_name: Option<&str>, opts: &BuildOptions)
                 -> R<JacobianProgram> {
        let model = the_model(file, model_name)?;
        let sv = sysview(file, Some(&model))?;
        let ctx = ctx_of(&sv)?;
        let obs = observed_defs(&sv)?;
        let entries = jacobian_bands_view(&sv, &ctx, &obs, Wrt::States)?;
        if entries.is_empty() {
            return Err(JacError::Build(
                "empty band list (an all-zero Jacobian is never a valid program — check the \
                 band calculus against this document)".into()));
        }
        let (evaldoc, names) =
            evaluation_document(&sv, &entries, opts.cse, opts.cse_min_nodes)?;
        let doc = compiled_document(file, &model, &evaldoc, &names, &entries, true)?;
        let rhs = compile(&doc, &model, &names)?;
        let lane_doc = compiled_document(file, &model, &evaldoc, &names, &entries, false)?;
        let lane_rhs = OnceLock::new();
        if lane_doc == doc {
            let _ = lane_rhs.set(Ok(rhs.clone()));
        }
        let dims = sv.variables.iter()
            .map(|(n, v)| (n.clone(), v.shape.clone().unwrap_or_default())).collect();
        Ok(JacobianProgram {
            inner: Arc::new(ProgInner {
                id: next_id(),
                file: file.clone(),
                model_name: model_name.map(str::to_string),
                model,
                entries,
                evaldoc,
                rhs,
                lane_doc,
                lane_rhs,
                names,
                dims,
            }),
        })
    }

    /// The band entries, in `J_k` order.
    pub fn bands(&self) -> &[JacEntry] {
        &self.inner.entries
    }

    /// The derived evaluation document (byte-equal to `*.evaldoc.json` under
    /// `stable_json` with the default options).
    pub fn evaluation_document(&self) -> &Value {
        &self.inner.evaldoc
    }

    /// The compiled band-field program: writes `<model>.J_k`, reads the
    /// source states it needs as `aux`.
    pub fn rhs(&self) -> &RhsProgram {
        &self.inner.rhs
    }

    fn prefixed(&self, name: &str) -> String {
        format!("{}.{name}", self.inner.model)
    }

    /// A plan: `layout`'s [`ScatterPlan`] over this program's bands, bound to
    /// executor `exec` and to the integrator's program for the same operator
    /// (`source`, which `u`, `aux`, `p` and the forcing set of every call are
    /// laid out for). Strict: every band row must be a state `source` writes,
    /// every state a band field reads must be in `source` with the same
    /// extent, and every parameter and forcing slot of the band program must
    /// be one of `source`'s — so a source from another build is refused.
    pub fn plan(&self, layout: &Layout, source: &RhsProgram, exec: Exec) -> R<JacPlan> {
        let inner = &*self.inner;
        let sv = sysview(&inner.file, Some(&inner.model))?;
        let scatter = ScatterPlan::build(&sv, &inner.entries, layout)?;
        let perr = |m: String| -> R<JacPlan> { Err(JacError::Plan(m)) };

        // The executor, and where each variable lives in its inputs and in
        // the source's (`Slots`: name → start in the concatenated `u ++ aux`,
        // length, written?), the lanes an item covers, and the lane count.
        let (pexec, ours, theirs, width, l) = match exec {
            Exec::Domain => {
                let nl = scatter.lanes();
                (PlanExec::Domain(inner.rhs.clone(), inner.rhs.clone()),
                 Slots::tile(inner.rhs.layout()), Slots::tile(source.layout()), 1, nl)
            }
            Exec::Lanes { capacity } => {
                if capacity == 0 {
                    return perr("a lane capacity of 0".into());
                }
                let kind = match &scatter.kind {
                    PlanKind::LaneBlocks { .. } | PlanKind::LaneSparseBlocks { .. } =>
                        LaneKind::Cells { sets: self.lane_sets(&scatter, None) },
                    PlanKind::ColumnBanded { axis, .. } =>
                        LaneKind::Columns { sets: self.lane_sets(&scatter, Some(axis)) },
                    PlanKind::Csr { .. } =>
                        return perr("Csr has no lane form; use Exec::Domain".into()),
                };
                if kind.sets().is_empty() {
                    return perr(format!("{kind:?} has no lane set (one lane would be the whole \
                                         domain); use Exec::Domain"));
                }
                let base = inner.lane_rhs.get_or_init(|| {
                    compile(&inner.lane_doc, &inner.model, &inner.names).map_err(|e| e.to_string())
                }).as_ref().map_err(|e| JacError::Plan(e.clone()))?;
                let lp = base.lanes(kind.clone(), capacity).map_err(|e| JacError::Plan(
                    format!("the band fields do not lane as {kind:?}: {e}")))?;
                let o = Slots::lane(lp.layout());
                // The source's lane layout follows from its tile layout (RFC
                // U3 §6), so the source need not lane itself; the same rule
                // applied to the band program must reproduce its lane layout.
                if Slots::lanes_of(base.layout(), kind.sets(), capacity)
                    .ok().as_ref() != Some(&o)
                {
                    return perr("the lane layout rule disagrees with earthsci-ast's lane \
                                 program".into());
                }
                let t = Slots::lanes_of(source.layout(), kind.sets(), capacity)
                    .map_err(JacError::Plan)?;
                (PlanExec::Lanes(lp, base.clone()), o, t, capacity, capacity)
            }
        };

        // The band program's own inputs, gathered from the source's.
        let mut gather = Vec::new();
        for (name, s) in &ours.vars {
            if s.written {
                continue;
            }
            let Some(t) = theirs.vars.get(name) else {
                return perr(format!("the band fields read `{name}`, which the source program \
                                     neither writes nor reads"));
            };
            if t.len != s.len {
                return perr(format!("`{name}` holds {} values in the source program and {} in \
                                     the band program (a source from another build?)",
                                    t.len, s.len));
            }
            gather.push((t.start, s.start - ours.u_len, s.len));
        }
        gather.sort_by_key(|g| g.1);
        // Reading the source's `u` as is: every input at its own offset.
        let identity = ours.aux_len == theirs.u_len
            && gather.iter().map(|g| g.2).sum::<usize>() == ours.aux_len
            && gather.iter().all(|g| g.0 == g.1);
        let aux_map = if identity { AuxMap::SourceU } else { AuxMap::Copy(gather) };

        // Parameters by name.
        let bprog = match &pexec {
            PlanExec::Domain(_, b) | PlanExec::Lanes(_, b) => b,
        };
        let ours_p = bprog.parameters();
        let theirs_p = source.parameters();
        let p_map = if ours_p.iter().map(|p| &p.name).eq(theirs_p.iter().map(|p| &p.name)) {
            None
        } else {
            let at: HashMap<&String, usize> =
                theirs_p.iter().enumerate().map(|(i, p)| (&p.name, i)).collect();
            let mut m = Vec::with_capacity(ours_p.len());
            for p in ours_p {
                match at.get(&p.name) {
                    Some(i) => m.push(*i),
                    None => return perr(format!("parameter `{}` of the band program is not a \
                                                 parameter of the source program", p.name)),
                }
            }
            Some(m)
        };

        // Forcing slots by name.
        let ours_f = bprog.forcing_slots();
        let theirs_f = source.forcing_slots();
        let forcing_map = if ours_f == theirs_f {
            None
        } else {
            let mut m = Vec::with_capacity(ours_f.len());
            for f in ours_f {
                match theirs_f.iter().position(|g| g.name == f.name && g.shape == f.shape) {
                    Some(i) => m.push(i),
                    None => return perr(format!("forcing slot `{}` {:?} of the band program is \
                                                 not a forcing slot of the source program",
                                                f.name, f.shape)),
                }
            }
            Some(m)
        };

        // Contributions: (slab start, output start, row start in y, column
        // start in v), each covering `width` consecutive lanes.
        let mut items = Vec::new();
        let field = |k: usize| -> R<&SlotInfo> {
            let nm = self.prefixed(&format!("J_{}", k + 1));
            ours.vars.get(&nm).ok_or_else(|| JacError::Plan(format!("no band field `{nm}`")))
        };
        let state = |u: &str, row: bool| -> R<&SlotInfo> {
            let nm = self.prefixed(u);
            match theirs.vars.get(&nm) {
                Some(s) if s.written || !row => Ok(s),
                Some(_) => Err(JacError::Plan(format!(
                    "band row `{nm}` is a state the source program reads but does not write"))),
                None => Err(JacError::Plan(format!(
                    "state `{nm}` is not in the source program (a source from another build?)"))),
            }
        };
        // A lane layout is one template per item, `l` lanes wide: lane `ℓ` is
        // the `ℓ`-th lane-domain cell in a domain plan too, and every field
        // and state is contiguous over the lanes. So is a column layout
        // whose axis is the last dimension (columns enumerate the others).
        let template = match &scatter.kind {
            PlanKind::LaneBlocks { .. } | PlanKind::LaneSparseBlocks { .. } => true,
            PlanKind::ColumnBanded { cells, axis_dim, .. } =>
                matches!(exec, Exec::Lanes { .. }) || *axis_dim + 1 == cells.len(),
            PlanKind::Csr { .. } => false,
        };
        let width = if template { width.max(l) } else { width };
        let fields: Vec<usize> = (0..scatter.bands.len())
            .map(|k| field(k).map(|s| s.start)).collect::<R<_>>()?;
        let field = |k: usize| -> R<usize> { Ok(fields[k]) };
        match &scatter.kind {
            PlanKind::LaneBlocks { block_vars, items: it, .. } if template => {
                let n = block_vars.len();
                for &(k, off) in it {
                    let (r, c) = (off % n, off / n);
                    items.push(Item {
                        slab: field(k)?,
                        out: off * l,
                        row: state(&block_vars[r], true)?.start,
                        col: state(&block_vars[c], false)?.start,
                    });
                }
            }
            PlanKind::LaneSparseBlocks { block_vars, lu_pattern, items: it, .. } if template => {
                for &(k, q) in it {
                    let (r, c) = lu_pattern[q];
                    items.push(Item {
                        slab: field(k)?,
                        out: q * l,
                        row: state(&block_vars[r], true)?.start,
                        col: state(&block_vars[c], false)?.start,
                    });
                }
            }
            PlanKind::ColumnBanded { cells, axis_dim, vars, kl, items: it, .. } if template => {
                let nn = cells[*axis_dim] as usize;
                for &(k, lev, off) in it {
                    let (r, c, d) = column_decode(off, nn, vars.len(), *kl, scatter_w(&scatter));
                    let colev = (lev as i64 + d) as usize;
                    items.push(Item {
                        slab: field(k)? + lev * l,
                        out: off * l,
                        row: state(&vars[r], true)?.start + lev * l,
                        col: state(&vars[c], false)?.start + colev * l,
                    });
                }
            }
            _ => {
                // One item per contribution, decoded to its (row, column)
                // state cells through the layout's own positions.
                let canon = Canon::new(&scatter.states);
                for (k, f, pos) in scatter.contributions(l) {
                    let (rc, cc) = position_cells(&scatter, &canon, pos, l);
                    let (ru, rf) = rc;
                    let (cu, cf) = cc;
                    items.push(Item {
                        slab: field(k)? + f,
                        out: pos,
                        row: state(ru, true)?.start + rf,
                        col: state(cu, false)?.start + cf,
                    });
                }
            }
        }
        // Runs of items sharing an output start. A stable sort keeps plan
        // order within each element (a per-contribution walk visits the
        // lanes of one item before the next item's).
        items.sort_by_key(|it| it.out);
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, it) in items.iter().enumerate() {
            match runs.last_mut() {
                Some(r) if items[r.0].out == it.out => r.1 = i + 1,
                _ => runs.push((i, i + 1)),
            }
        }
        // The output positions no run writes (block slots no band reaches,
        // LU fill-in): zeroed by every call; every other one is written once.
        let out_len = scatter.out_len(l);
        let mut zero = Vec::new();
        let mut at = 0;
        for &(lo, _) in &runs {
            let o = items[lo].out;
            if o > at {
                zero.push((at, o));
            }
            at = o + width;
        }
        if out_len > at {
            zero.push((at, out_len));
        }
        let pairs = if width == 1 {
            if ours.u_len.max(out_len) > u32::MAX as usize {
                return perr("more than 2³² band-field or output slots".into());
            }
            items.iter().map(|it| (it.slab as u32, it.out as u32)).collect()
        } else {
            Vec::new()
        };
        let single: Vec<bool> = (0..pairs.len()).map(|i| {
            let o = pairs[i].1;
            (i == 0 || pairs[i - 1].1 != o) && pairs.get(i + 1).is_none_or(|n| n.1 != o)
        }).collect();
        let (singles, pairs): (Vec<_>, Vec<_>) = pairs.into_iter().zip(single)
            .partition(|(_, one)| *one);
        let pairs: Vec<(u32, u32)> = pairs.into_iter().map(|(q, _)| q).collect();
        let mut singles: Vec<(u32, u32)> = singles.into_iter().map(|(q, _)| q).collect();
        singles.sort_unstable();
        Ok(JacPlan {
            inner: Arc::new(PlanInner {
                id: next_id(),
                owner: inner.id,
                out_len,
                zero,
                singles,
                pairs,
                scatter,
                exec: pexec,
                width,
                capacity: l,
                items,
                runs,
                ours_u_len: ours.u_len,
                ours_aux_len: ours.aux_len,
                src_u_len: theirs.u_len,
                src_aux_len: theirs.aux_len,
                n_src_params: theirs_p.len(),
                aux_map,
                p_map,
                forcing_map,
            }),
        })
    }

    /// The lane sets of a lane plan: every dimension of the states' common
    /// shape, less `axis` for a column plan.
    fn lane_sets(&self, scatter: &ScatterPlan, axis: Option<&String>) -> Vec<String> {
        let dims = scatter.states.iter().find(|(_, s)| !s.is_empty())
            .and_then(|(u, _)| self.inner.dims.get(u)).cloned().unwrap_or_default();
        dims.into_iter().filter(|d| Some(d) != axis).collect()
    }

    /// Buffers for calls with `plan`.
    pub fn new_workspace(&self, plan: &JacPlan) -> JacWorkspace {
        let p = &*plan.inner;
        let rhs = match &p.exec {
            PlanExec::Domain(prog, _) => prog.new_workspace(),
            PlanExec::Lanes(lp, _) => lp.new_workspace(),
        };
        JacWorkspace {
            plan: p.id,
            rhs,
            slab: vec![0.0; p.ours_u_len],
            zeros: vec![0.0; p.ours_u_len],
            aux: vec![0.0; p.ours_aux_len],
            p: vec![0.0; p.p_map.as_ref().map_or(0, Vec::len)],
            acc: vec![0.0; p.width],
            forcing: None,
            evaluated: false,
        }
    }

    fn check(&self, plan: &JacPlan, ws: &JacWorkspace) -> R<()> {
        if plan.inner.owner != self.inner.id {
            return Err(JacError::Call("the plan was made by another program".into()));
        }
        if ws.plan != plan.inner.id {
            return Err(JacError::Call("the workspace was made for another plan".into()));
        }
        Ok(())
    }

    /// Evaluate the band fields and scatter them into `out` per `plan`:
    /// zeroed, then each element the `f64` sum of its contributions in plan
    /// order, rounded once to `T`.
    ///
    /// `u`, `aux`, `p` and `forcing` are what the source program (see
    /// [`JacobianProgram::plan`]) takes for the same call; for
    /// [`Exec::Lanes`], `cells` (`[D, capacity]`, column-major, 1-based) and
    /// `n` name the lanes' cells or columns as for
    /// [`LaneProgram::eval_into`]. A domain plan takes `&[]` and 0. The band
    /// slab stays in `ws` for [`Self::apply`] and [`Self::apply_transpose`].
    #[allow(clippy::too_many_arguments)]
    pub fn eval_into<T: StorageScalar>(&self, plan: &JacPlan, u: &[f64], aux: &[f64],
                                       p: &[f64], t: f64, forcing: &ForcingSet,
                                       cells: &[i32], n: usize, out: &mut [T],
                                       ws: &mut JacWorkspace) -> R<()> {
        self.check(plan, ws)?;
        let pl = &*plan.inner;
        ws.evaluated = false;
        let len = |what: &str, want: usize, got: usize| -> R<()> {
            if want != got {
                return Err(JacError::Call(format!("{what} has length {got}; the plan needs \
                                                   {want}")));
            }
            Ok(())
        };
        len("u", pl.src_u_len, u.len())?;
        len("aux", pl.src_aux_len, aux.len())?;
        len("p", pl.n_src_params, p.len())?;
        len("out", pl.out_len, out.len())?;
        let JacWorkspace { rhs, slab, zeros, aux: our_aux, p: our_p, acc, forcing: fcache, .. } =
            ws;
        let aux_in: &[f64] = match &pl.aux_map {
            AuxMap::SourceU => u,
            AuxMap::Copy(m) => {
                let su = u.len();
                for &(src, dst, n) in m {
                    let from = if src < su { &u[src..src + n] } else { &aux[src - su..src - su + n] };
                    our_aux[dst..dst + n].copy_from_slice(from);
                }
                our_aux
            }
        };
        let p_in: &[f64] = match &pl.p_map {
            None => p,
            Some(m) => {
                for (d, &s) in our_p.iter_mut().zip(m) {
                    *d = p[s];
                }
                our_p
            }
        };
        let f_in: &ForcingSet = match &pl.forcing_map {
            None => forcing,
            Some(m) => {
                if fcache.as_ref().map(|c| c.0) != Some(forcing.epoch()) {
                    let bufs = m.iter().map(|&j| Arc::from(forcing.buffer(j))).collect();
                    let base = match &pl.exec {
                        PlanExec::Domain(_, b) | PlanExec::Lanes(_, b) => b,
                    };
                    *fcache = Some((forcing.epoch(), ForcingSet::new(base, bufs)?));
                }
                &fcache.as_ref().expect("filled above").1
            }
        };
        match &pl.exec {
            PlanExec::Domain(prog, _) => {
                if !cells.is_empty() || n != 0 {
                    return Err(JacError::Call("a domain plan takes no cells".into()));
                }
                prog.eval_into(zeros, aux_in, p_in, t, f_in, slab, rhs)?;
            }
            PlanExec::Lanes(lp, _) => lp.eval_into(zeros, aux_in, p_in, t, f_in, cells, n, slab, rhs)?,
        }
        let w = pl.width;
        for &(lo, hi) in &pl.zero {
            out[lo..hi].iter_mut().for_each(|x| *x = T::from_f64(0.0));
        }
        if w == 1 {
            // Elements with one contribution, in slab order (the reads
            // stream; the scattered writes do not stall a sum)...
            for &(sl, o) in &pl.singles {
                out[o as usize] = T::from_f64(0.0 + slab[sl as usize]);
            }
            // ...then the shared ones: (slab, output) pairs in output order,
            // a run being a stretch of equal outputs.
            let pairs = &pl.pairs;
            let mut i = 0;
            while i < pairs.len() {
                let o = pairs[i].1;
                let mut a = 0.0;
                while i < pairs.len() && pairs[i].1 == o {
                    a += slab[pairs[i].0 as usize];
                    i += 1;
                }
                out[o as usize] = T::from_f64(a);
            }
        } else {
            for &(lo, hi) in &pl.runs {
                let first = &pl.items[lo];
                for (a, s) in acc.iter_mut().zip(&slab[first.slab..first.slab + w]) {
                    *a = 0.0 + *s;
                }
                for it in &pl.items[lo + 1..hi] {
                    for (a, s) in acc.iter_mut().zip(&slab[it.slab..it.slab + w]) {
                        *a += *s;
                    }
                }
                for (x, a) in out[first.out..first.out + w].iter_mut().zip(acc.iter()) {
                    *x = T::from_f64(*a);
                }
            }
        }
        ws.evaluated = true;
        Ok(())
    }

    fn check_evaluated(&self, plan: &JacPlan, ws: &JacWorkspace) -> R<()> {
        self.check(plan, ws)?;
        if !ws.evaluated {
            return Err(JacError::Call("no band slab: call eval_into first (a failed \
                                       evaluation leaves none)".into()));
        }
        Ok(())
    }

    /// `y += J·v` over the band slab the last [`Self::eval_into`] left in
    /// `ws`, nothing assembled. `v` is the source's `u ++ aux` (every state a
    /// column may name), `y` its `u`. Each element is updated in `f64` in
    /// plan order and rounded once per contribution.
    pub fn apply<T: StorageScalar>(&self, plan: &JacPlan, ws: &JacWorkspace, v: &[T],
                                   y: &mut [T]) -> R<()> {
        self.check_evaluated(plan, ws)?;
        let pl = &*plan.inner;
        self.mf_lengths(pl, v.len(), y.len())?;
        let w = pl.width;
        if w == 1 {
            for it in &pl.items {
                let yy = &mut y[it.row];
                *yy = T::from_f64(yy.to_f64() + ws.slab[it.slab] * v[it.col].to_f64());
            }
            return Ok(());
        }
        for it in &pl.items {
            let ys = &mut y[it.row..it.row + w];
            let js = &ws.slab[it.slab..it.slab + w];
            let vs = &v[it.col..it.col + w];
            for ((yy, jj), vv) in ys.iter_mut().zip(js).zip(vs) {
                *yy = T::from_f64(yy.to_f64() + jj * vv.to_f64());
            }
        }
        Ok(())
    }

    /// `ḡ += Jᵀ·λ` over the band slab the last [`Self::eval_into`] left in
    /// `ws` (the `vjp_u` of an explicit or Rosenbrock stage). `λ` is the
    /// source's `u`, `ḡ` its `u ++ aux`; accumulated in `f64` in plan order.
    pub fn apply_transpose(&self, plan: &JacPlan, ws: &JacWorkspace, lambda: &[f64],
                           gbar: &mut [f64]) -> R<()> {
        self.check_evaluated(plan, ws)?;
        let pl = &*plan.inner;
        self.mf_lengths(pl, gbar.len(), lambda.len())?;
        let w = pl.width;
        if w == 1 {
            for it in &pl.items {
                gbar[it.col] += ws.slab[it.slab] * lambda[it.row];
            }
            return Ok(());
        }
        for it in &pl.items {
            let gs = &mut gbar[it.col..it.col + w];
            let js = &ws.slab[it.slab..it.slab + w];
            let ls = &lambda[it.row..it.row + w];
            for ((gg, jj), ll) in gs.iter_mut().zip(js).zip(ls) {
                *gg += jj * ll;
            }
        }
        Ok(())
    }

    fn mf_lengths(&self, pl: &PlanInner, cols: usize, rows: usize) -> R<()> {
        if cols != pl.src_u_len + pl.src_aux_len || rows != pl.src_u_len {
            return Err(JacError::Call(format!(
                "matrix-free vectors have lengths (columns {cols}, rows {rows}); the plan needs \
                 ({}, {})", pl.src_u_len + pl.src_aux_len, pl.src_u_len)));
        }
        Ok(())
    }

    /// The model this program differentiates, as passed to `build`.
    pub fn model_name(&self) -> Option<&str> {
        self.inner.model_name.as_deref()
    }
}

/// `ColumnBanded` band count `kl + ku + 1`.
fn scatter_w(s: &ScatterPlan) -> usize {
    match &s.kind {
        PlanKind::ColumnBanded { kl, ku, .. } => (kl + ku + 1) as usize,
        _ => 1,
    }
}

/// `(r, c, d)` of a `ColumnBanded` offset `((c·nvar + r)·w + (d+kl))·N + lev`.
fn column_decode(off: usize, nn: usize, nvar: usize, kl: i64, w: usize) -> (usize, usize, i64) {
    let t = off / nn;
    let j = t % w;
    let rc = t / w;
    (rc % nvar, rc / nvar, j as i64 - kl)
}

/// The canonical state order (spec §8): each state's first canonical index.
struct Canon<'a> {
    states: &'a [(String, Vec<i64>)],
    starts: Vec<usize>,
}

impl<'a> Canon<'a> {
    fn new(states: &'a [(String, Vec<i64>)]) -> Self {
        let mut starts = Vec::with_capacity(states.len() + 1);
        let mut n = 0;
        for (_, s) in states {
            starts.push(n);
            n += ncells(s);
        }
        starts.push(n);
        Canon { states, starts }
    }

    /// (state, flat cell) of canonical index `i`.
    fn cell(&self, i: usize) -> (&'a str, usize) {
        let s = self.starts.partition_point(|&x| x <= i) - 1;
        (&self.states[s].0, i - self.starts[s])
    }
}

/// The (row, column) state cells output position `pos` holds in a domain
/// plan at lane capacity `l`, as (state, column-major flat cell).
fn position_cells<'a>(s: &'a ScatterPlan, canon: &Canon<'a>, pos: usize, l: usize)
                      -> ((&'a str, usize), (&'a str, usize)) {
    match &s.kind {
        PlanKind::LaneBlocks { block_vars, .. } => {
            let n = block_vars.len();
            let (off, ln) = (pos / l, pos % l);
            ((&block_vars[off % n], ln), (&block_vars[off / n], ln))
        }
        PlanKind::LaneSparseBlocks { block_vars, lu_pattern, .. } => {
            let (r, c) = lu_pattern[pos / l];
            ((&block_vars[r], pos % l), (&block_vars[c], pos % l))
        }
        PlanKind::ColumnBanded { cells, axis_dim, vars, kl, .. } => {
            let nn = cells[*axis_dim] as usize;
            let (off, ln) = (pos / l, pos % l);
            let lev = off % nn;
            let (r, c, d) = column_decode(off, nn, vars.len(), *kl, scatter_w(s));
            let colev = (lev as i64 + d) as usize;
            ((&vars[r], ScatterPlan::column_cell(cells, *axis_dim, lev, ln)),
             (&vars[c], ScatterPlan::column_cell(cells, *axis_dim, colev, ln)))
        }
        PlanKind::Csr { rowptr, colind, .. } => {
            let row = rowptr.partition_point(|&x| x <= pos) - 1;
            (canon.cell(row), canon.cell(colind[pos]))
        }
    }
}

/// Where one variable lives in a program's concatenated `u ++ aux`.
#[derive(Debug, PartialEq, Eq)]
struct SlotInfo {
    start: usize,
    len: usize,
    written: bool,
}

/// A program layout by flattened name.
#[derive(Debug, PartialEq, Eq)]
struct Slots {
    vars: HashMap<String, SlotInfo>,
    u_len: usize,
    aux_len: usize,
}

impl Slots {
    fn tile(t: &earthsci_ast::TileLayout) -> Slots {
        let mut vars = HashMap::new();
        for (blocks, written, base) in [(&t.u, true, 0), (&t.aux, false, t.u_len)] {
            for b in blocks {
                vars.insert(b.name.clone(), SlotInfo {
                    start: base + b.base, len: b.len(), written,
                });
            }
        }
        Slots { vars, u_len: t.u_len, aux_len: t.aux_len }
    }

    /// The lane layout of a program with tile layout `t` laned over `sets`
    /// at `capacity` (RFC U3 §6): its blocks in order, each a column-major
    /// `[lane, other axes…]` block over the variable's axes outside `sets`.
    fn lanes_of(t: &earthsci_ast::TileLayout, sets: &[String], capacity: usize)
                -> Result<Slots, String> {
        let mut vars = HashMap::new();
        let mut lens = [0usize; 2];
        for (blocks, written, which) in [(&t.u, true, 0), (&t.aux, false, 1)] {
            for b in blocks {
                if let Some(s) = sets.iter().find(|s| !b.sets.contains(s)) {
                    return Err(format!("the source state `{}` does not carry the lane set \
                                        `{s}`", b.name));
                }
                let len = capacity * b.sets.iter().zip(&b.storage)
                    .filter(|(s, _)| !sets.contains(s))
                    .map(|(_, (lo, hi))| (hi - lo + 1).max(0) as usize)
                    .product::<usize>();
                vars.insert(b.name.clone(), SlotInfo { start: lens[which], len, written });
                lens[which] += len;
            }
        }
        // aux blocks follow u in the concatenated buffer
        for s in vars.values_mut() {
            if !s.written {
                s.start += lens[0];
            }
        }
        Ok(Slots { vars, u_len: lens[0], aux_len: lens[1] })
    }

    fn lane(t: &earthsci_ast::LaneLayout) -> Slots {
        let l = t.capacity;
        let mut vars = HashMap::new();
        for (blocks, written, base) in [(&t.u, true, 0), (&t.aux, false, t.u_len)] {
            for b in blocks {
                vars.insert(b.name.clone(), SlotInfo {
                    start: base + b.offset * l,
                    len: l * b.extents.iter().product::<usize>(),
                    written,
                });
            }
        }
        Slots { vars, u_len: t.u_len, aux_len: t.aux_len }
    }
}

/// The executor, with the band program it was made from (whose parameters
/// and forcing slots a call fills).
#[derive(Clone)]
enum PlanExec {
    Domain(RhsProgram, RhsProgram),
    Lanes(LaneProgram, RhsProgram),
}

/// How the band program's `aux` is read from the source's inputs.
#[derive(Clone, Debug)]
enum AuxMap {
    /// The source's `u` is the band program's `aux`, offset for offset.
    SourceU,
    /// Copies `(start in source u ++ aux, start in our aux, length)`.
    Copy(Vec<(usize, usize, usize)>),
}

/// One contribution template: slab start, output start, row start (source
/// `u`) and column start (source `u ++ aux`), each covering `width` lanes.
#[derive(Clone, Copy, Debug)]
struct Item {
    slab: usize,
    out: usize,
    row: usize,
    col: usize,
}

/// A [`ScatterPlan`] bound to an executor and to the source program's
/// layout (see [`JacobianProgram::plan`]). Immutable, `Send + Sync`, cheap to
/// clone: one plan serves every thread and every leaf using the operator.
#[derive(Clone)]
pub struct JacPlan {
    inner: Arc<PlanInner>,
}

struct PlanInner {
    id: u64,
    owner: u64,
    scatter: ScatterPlan,
    exec: PlanExec,
    /// Lanes each item covers: the capacity (lanes) or 1 (domain).
    width: usize,
    capacity: usize,
    out_len: usize,
    items: Vec<Item>,
    /// `[lo, hi)` runs of `items` sharing an output start.
    runs: Vec<(usize, usize)>,
    /// `[lo, hi)` output ranges no run writes.
    zero: Vec<(usize, usize)>,
    /// When each item is one lane wide: the outputs with one contribution as
    /// `(slab, output)` in slab order, and the rest in output order.
    singles: Vec<(u32, u32)>,
    pairs: Vec<(u32, u32)>,
    ours_u_len: usize,
    ours_aux_len: usize,
    src_u_len: usize,
    src_aux_len: usize,
    n_src_params: usize,
    aux_map: AuxMap,
    /// Source parameter index of each band-program parameter (`None`: the
    /// same list).
    p_map: Option<Vec<usize>>,
    /// Source forcing slot of each band-program slot (`None`: the same).
    forcing_map: Option<Vec<usize>>,
}

impl fmt::Debug for JacPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JacPlan")
            .field("capacity", &self.inner.capacity)
            .field("out_len", &self.inner.out_len)
            .field("items", &self.inner.items.len())
            .finish()
    }
}

impl JacPlan {
    /// The storage map (the `*.plan.json` golden's content).
    pub fn scatter(&self) -> &ScatterPlan {
        &self.inner.scatter
    }

    /// The lane capacity `L` the output is laid out for (the lane count of a
    /// domain plan's lane domain; 1 for `Csr`).
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// The output length `eval_into` fills.
    pub fn out_len(&self) -> usize {
        self.inner.out_len
    }

    /// The lengths of the source `u` and `aux` every call takes.
    pub fn source_lengths(&self) -> (usize, usize) {
        (self.inner.src_u_len, self.inner.src_aux_len)
    }
}

/// The per-call buffers of one plan: the band program's workspace and the
/// band slab. Reused across calls, so a warm call with an unchanged forcing
/// set allocates nothing. `Send`, not `Sync`: one per thread.
pub struct JacWorkspace {
    plan: u64,
    rhs: RhsWorkspace,
    slab: Vec<f64>,
    zeros: Vec<f64>,
    aux: Vec<f64>,
    p: Vec<f64>,
    acc: Vec<f64>,
    /// The band program's forcing set, rebuilt from the source's when its
    /// epoch moves (only when the two programs' slot lists differ).
    forcing: Option<(u64, ForcingSet)>,
    evaluated: bool,
}

impl JacWorkspace {
    /// The band slab of the last successful `eval_into`, in the band
    /// program's `u` layout (each `J_k` block; lanes fastest for a lane plan).
    pub fn slab(&self) -> &[f64] {
        &self.slab
    }
}

impl fmt::Debug for JacWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JacWorkspace").field("slab", &self.slab.len()).finish()
    }
}
