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
//! * [`HvpProgram`] does the same for the second-order bands of an axis pair
//!   (spec §4.4, [`crate::jacobian_bands2`]): each entry rides through the
//!   first-order compile as a band with concatenated column indices, an
//!   [`HvpPlan`] resolves every band point to the source program's rows,
//!   columns and run-time inputs by name, and [`HvpProgram::contract`] runs
//!   the Rosenbrock-adjoint contraction over the slab (whole-domain plans
//!   only; [`crate::numeric::HvpEvaluator`] is its oracle).
//!
//! `JacobianProgram`, `JacPlan`, `HvpProgram` and `HvpPlan` are immutable,
//! `Send + Sync` and cheap to clone; a `JacWorkspace` or `HvpWorkspace` is
//! `Send`, not `Sync`, one per thread. A lane's
//! result depends only on its own inputs and the plan order (no atomics, no
//! thread-local partial sums), so assembly is bitwise invariant to thread
//! count and lane position. The oracle ([`crate::assemble_jacobian`]) stays
//! the reference this path is tested against.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use earthsci_ast::{
    esm_problem, rhs_program, Compiler, EsmFile, ForcingSet, ForcingSlot, LaneKind, LaneProgram,
    ProblemOptions, RhsError, RhsProgram, RhsSelection, RhsWorkspace, SlotKind,
};
use serde_json::{json, Map, Value};

use crate::bands::BandError;
use crate::clip::RIdx;
use crate::emit::{evaluation_document, evaluation_document_with};
use crate::jacobian::{jacobian_bands_view, JacEntry, Wrt};
use crate::numeric::{foreach_point, outside, HvpForm};
use crate::scatter::{flat, ncells, Layout, PlanKind, ScatterPlan};
use crate::second_order::{axis_pair, bands2_view, JacEntry2};
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
        BuildOptions {
            cse: true,
            cse_min_nodes: 12,
        }
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
    opts: BuildOptions,
    /// The `*.evaldoc.json` form, emitted on first request (the compiled
    /// document is built from the form without the zeroed `D` equations).
    evaldoc: OnceLock<Value>,
    /// The compiled document (fused band form) and its programs, selected
    /// to the band fields; the build-time-data form is compiled at build.
    doc: Value,
    rhs: Compiled,
    /// The document as emitted, and its programs, compiled on the first lane
    /// plan (see [`fused_band`]).
    lane_doc: Value,
    lane_rhs: Compiled,
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
        _ => Err(JacError::Build(
            "the document has several models (or none); name one".into(),
        )),
    }
}

/// The source document as JSON, serialized once per build.
fn source_value(file: &EsmFile) -> R<Value> {
    let text = earthsci_ast::to_json(file)
        .map_err(|e| JacError::Build(format!("source document does not serialize: {e}")))?;
    serde_json::from_str(&text)
        .map_err(|e| JacError::Build(format!("source document does not serialize: {e}")))
}

/// The document that is compiled: the source document `src` (so data
/// loaders and forcing declarations stay what the integrator's own program
/// sees) with `model`'s variables and equations replaced by those of
/// `evaldoc`, the evaluation document emitted without the zeroed source `D`
/// equations (the states stay read-only inputs). Source variables keep their
/// source declarations, so a forcing parameter is a forcing slot here too;
/// the band fields (whose names `build` checked are not source variables)
/// keep the derived ones. The model keeps its name, so the flattened names
/// equal the source program's.
fn compiled_document(
    src: &Value,
    model: &str,
    evaldoc: &Value,
    names: &[String],
    entries: &[JacEntry],
    fuse: bool,
) -> R<Value> {
    let mut src = src.clone();
    let derived = &evaldoc["models"]["JacobianEval"];
    let src_vars = src["models"][model]["variables"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let bands: HashMap<&str, &JacEntry> = names.iter().map(String::as_str).zip(entries).collect();
    let mut vars = Map::new();
    for (n, d) in derived["variables"].as_object().into_iter().flatten() {
        let keep = src_vars.get(n).filter(|_| !bands.contains_key(n.as_str()));
        vars.insert(n.clone(), keep.unwrap_or(d).clone());
    }
    let eqs: Vec<Value> = derived["equations"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|eq| match lhs_band(eq, &bands) {
            Some((u, en)) if fuse => fused_band(u, en, eq),
            _ => vec![eq.clone()],
        })
        .collect();
    let Value::Object(top) = &mut src else {
        return Err(JacError::Build("source document is not an object".into()));
    };
    let mut isets = top
        .get("index_sets")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (n, s) in evaldoc["index_sets"].as_object().into_iter().flatten() {
        isets.entry(n.clone()).or_insert_with(|| s.clone());
    }
    top.insert("index_sets".into(), Value::Object(isets));
    top.insert(
        "models".into(),
        json!({model: {"variables": vars, "equations": eqs}}),
    );
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
fn lhs_band<'a>(
    eq: &'a Value,
    bands: &HashMap<&str, &'a JacEntry>,
) -> Option<(&'a str, &'a JacEntry)> {
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
/// a row dimension stays as emitted too, even when it spans the field (an
/// extent-1 dimension): the faq leaves fixed dimensions out of its output.
fn fused_band(name: &str, en: &JacEntry, eq: &Value) -> Vec<Value> {
    let rhs = &eq["rhs"];
    let value = &rhs["values"][1];
    let bounds = |r: &Value| -> Option<Vec<(i64, i64)>> {
        r.as_array()?
            .iter()
            .map(|b| Some((b[0].as_i64()?, b[1].as_i64()?)))
            .collect()
    };
    let (Some(full), Some(region)) = (bounds(&rhs["regions"][0]), bounds(&rhs["regions"][1]))
    else {
        return vec![eq.clone()];
    };
    if rhs.get("op").and_then(Value::as_str) != Some("makearray")
        || value.get("op").and_then(Value::as_str) != Some("faq")
    {
        return vec![eq.clone()];
    }
    if en.band.ridx.iter().any(|r| matches!(r, RIdx::Fixed(_))) {
        return vec![eq.clone()];
    }
    if full == region {
        return vec![json!({"lhs": eq["lhs"], "rhs": value})];
    }
    let idx: Vec<Value> = en
        .band
        .ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(s) => Some(json!(s)),
            RIdx::Fixed(_) => None, // refused above
        })
        .chain(en.band.contracted.iter().map(|(n, _, _)| json!(n)))
        .collect();
    let lhs = |ranges: &Value| {
        json!({
        "op": "faq", "args": [], "output_idx": idx, "ranges": ranges,
        "expr": {"op": "D", "wrt": "t", "args": [{"op": "index", "args":
                 std::iter::once(json!(name)).chain(idx.iter().cloned()).collect::<Vec<_>>()}]}})
    };
    let mut out = vec![json!({"lhs": lhs(&value["ranges"]), "rhs": value})];
    // The complement, dimension by dimension: inside the region before `m`,
    // below or above it along `m`, the whole field after.
    for m in 0..full.len() {
        let (lo, hi) = region[m];
        for side in [(full[m].0, lo - 1), (hi + 1, full[m].1)] {
            if side.0 > side.1 {
                continue;
            }
            let ranges: Map<String, Value> = idx
                .iter()
                .enumerate()
                .map(|(d, n)| {
                    let r = if d < m {
                        region[d]
                    } else if d == m {
                        side
                    } else {
                        full[d]
                    };
                    (
                        n.as_str().expect("index names").to_string(),
                        json!([r.0, r.1]),
                    )
                })
                .collect();
            let ranges = Value::Object(ranges);
            out.push(json!({"lhs": lhs(&ranges), "rhs": {
                "op": "faq", "args": [], "output_idx": idx, "ranges": ranges, "expr": 0.0}}));
        }
    }
    out
}

/// The band-field program of a compiled document: `native`, selected to
/// write the band fields `<model>.J_k` only.
///
/// With `runtime_arrays`, every shaped parameter without an `update` stays a
/// run-time [`earthsci_ast::SlotKind::ArrayParameter`] forcing slot
/// (`ProblemOptions::runtime_array_parameters`), as in a source program built
/// that way, so the band fields read the source's buffer rather than a
/// build-time copy of the document's value.
fn compile(doc: &Value, model: &str, names: &[String], runtime_arrays: bool) -> R<RhsProgram> {
    let popts = ProblemOptions {
        compiler: Some(Compiler::Native),
        runtime_array_parameters: runtime_arrays,
        ..Default::default()
    };
    let prob = esm_problem(doc, (0.0, 1.0), popts)
        .map_err(|e| JacError::Build(format!("evaluation document: {e}")))?;
    let whole =
        rhs_program(&prob).map_err(|e| JacError::Build(format!("evaluation document: {e}")))?;
    let writes = names.iter().map(|n| format!("{model}.{n}")).collect();
    whole
        .select(&RhsSelection {
            writes: Some(writes),
            terms: None,
        })
        .map_err(|e| JacError::Build(format!("band-field selection: {e}")))
}

/// Whether `source` holds shaped parameters as run-time slots
/// (`ProblemOptions::runtime_array_parameters`). A band program planned
/// against it must read those slots too: compiled with the shaped
/// parameters as build-time data, its fields would see the document's
/// values while the source's buffers move, and nothing would say so.
fn runtime_arrays(source: &RhsProgram) -> bool {
    source
        .forcing_slots()
        .iter()
        .any(|s| s.kind == SlotKind::ArrayParameter)
}

/// One band-field document's programs, each compiled on first use: shaped
/// parameters as build-time data, and as run-time slots ([`compile`]).
#[derive(Default)]
struct Compiled([OnceLock<Result<RhsProgram, String>>; 2]);

impl Compiled {
    /// With the build-time-data form already compiled.
    fn with(rhs: RhsProgram) -> Compiled {
        let c = Compiled::default();
        let _ = c.0[0].set(Ok(rhs));
        c
    }

    /// The form that matches `source` ([`runtime_arrays`]).
    fn matching(
        &self,
        doc: &Value,
        model: &str,
        names: &[String],
        source: &RhsProgram,
    ) -> R<&RhsProgram> {
        let arrays = runtime_arrays(source);
        self.0[usize::from(arrays)]
            .get_or_init(|| compile(doc, model, names, arrays).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| JacError::Plan(e.clone()))
    }

    /// The build-time-data form, when compiled.
    fn data(&self) -> Option<&RhsProgram> {
        self.0[0].get().and_then(|r| r.as_ref().ok())
    }
}

impl JacobianProgram {
    /// Differentiate `model_name` (the only model when `None`) with respect
    /// to its states and compile the band fields. Refuses an empty band list:
    /// the calculus found no coupling at all, which for an operator with
    /// array states is the silent all-zero failure the plans also refuse.
    pub fn build(
        file: &EsmFile,
        model_name: Option<&str>,
        opts: &BuildOptions,
    ) -> R<JacobianProgram> {
        let model = the_model(file, model_name)?;
        let sv = sysview(file, Some(&model))?;
        let ctx = ctx_of(&sv)?;
        let obs = observed_defs(&sv)?;
        let entries = jacobian_bands_view(&sv, &ctx, &obs, Wrt::States, None)?;
        if entries.is_empty() {
            return Err(JacError::Build(
                "empty band list (an all-zero Jacobian is never a valid program — check the \
                 band calculus against this document)"
                    .into(),
            ));
        }
        let (evaldoc, names) =
            evaluation_document_with(&sv, &entries, opts.cse, opts.cse_min_nodes, false)?;
        // The evaluation document names its band fields `J_k` whatever the
        // source declares: a source `J_k` would be both.
        if let Some(n) = names.iter().find(|n| sv.variables.contains_key(*n)) {
            return Err(JacError::Build(format!(
                "the source model declares `{n}`, the name of a band field of its Jacobian"
            )));
        }
        let src = source_value(file)?;
        let doc = compiled_document(&src, &model, &evaldoc, &names, &entries, true)?;
        let rhs = compile(&doc, &model, &names, false)?;
        let lane_doc = compiled_document(&src, &model, &evaldoc, &names, &entries, false)?;
        let lane_rhs = if lane_doc == doc {
            Compiled::with(rhs.clone())
        } else {
            Compiled::default()
        };
        let dims = sv
            .variables
            .iter()
            .map(|(n, v)| (n.clone(), v.shape.clone().unwrap_or_default()))
            .collect();
        Ok(JacobianProgram {
            inner: Arc::new(ProgInner {
                id: next_id(),
                file: file.clone(),
                model_name: model_name.map(str::to_string),
                model,
                entries,
                opts: opts.clone(),
                evaldoc: OnceLock::new(),
                doc,
                rhs: Compiled::with(rhs),
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
        let inner = &*self.inner;
        inner.evaldoc.get_or_init(|| {
            let sv = sysview(&inner.file, Some(&inner.model)).expect("viewed at build");
            evaluation_document(
                &sv,
                &inner.entries,
                inner.opts.cse,
                inner.opts.cse_min_nodes,
            )
            .expect("emitted at build")
            .0
        })
    }

    /// The compiled band-field program: writes `<model>.J_k`, reads the
    /// source states it needs as `aux`. Shaped parameters are build-time
    /// data here; a plan against a source built with
    /// `ProblemOptions::runtime_array_parameters` uses a form that reads
    /// them as run-time slots instead.
    pub fn rhs(&self) -> &RhsProgram {
        self.inner.rhs.data().expect("compiled at build")
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
                let rhs = inner
                    .rhs
                    .matching(&inner.doc, &inner.model, &inner.names, source)?;
                (
                    PlanExec::Domain(rhs.clone(), rhs.clone()),
                    Slots::tile(rhs.layout()),
                    Slots::tile(source.layout()),
                    1,
                    nl,
                )
            }
            Exec::Lanes { capacity } => {
                if capacity == 0 {
                    return perr("a lane capacity of 0".into());
                }
                let kind = match &scatter.kind {
                    PlanKind::LaneBlocks { .. } | PlanKind::LaneSparseBlocks { .. } => {
                        LaneKind::Cells {
                            sets: self.lane_sets(&scatter, None),
                        }
                    }
                    PlanKind::ColumnBanded { axis, .. } => LaneKind::Columns {
                        sets: self.lane_sets(&scatter, Some(axis)),
                    },
                    PlanKind::Csr { .. } => {
                        return perr("Csr has no lane form; use Exec::Domain".into())
                    }
                };
                if kind.sets().is_empty() {
                    return perr(format!(
                        "{kind:?} has no lane set (one lane would be the whole \
                                         domain); use Exec::Domain"
                    ));
                }
                let base =
                    inner
                        .lane_rhs
                        .matching(&inner.lane_doc, &inner.model, &inner.names, source)?;
                let lp = base.lanes(kind.clone(), capacity).map_err(|e| {
                    JacError::Plan(format!("the band fields do not lane as {kind:?}: {e}"))
                })?;
                let o = Slots::lane(lp.layout());
                // The source's lane layout follows from its tile layout (RFC
                // U3 §6), so the source need not lane itself; the same rule
                // applied to the band program must reproduce its lane layout.
                if Slots::lanes_of(base.layout(), kind.sets(), capacity)
                    .ok()
                    .as_ref()
                    != Some(&o)
                {
                    return perr(
                        "the lane layout rule disagrees with earthsci-ast's lane \
                                 program"
                            .into(),
                    );
                }
                let t = Slots::lanes_of(source.layout(), kind.sets(), capacity)
                    .map_err(JacError::Plan)?;
                (PlanExec::Lanes(lp, base.clone()), o, t, capacity, capacity)
            }
        };

        let bprog = match &pexec {
            PlanExec::Domain(_, b) | PlanExec::Lanes(_, b) => b,
        };
        let io = bind_inputs(bprog, &ours, &theirs, source)?;

        // Contributions: (slab start, output start, row start in y, column
        // start in v), each covering `width` consecutive lanes.
        let mut items = Vec::new();
        let field = |k: usize| -> R<&SlotInfo> {
            let nm = self.prefixed(&format!("J_{}", k + 1));
            ours.vars
                .get(&nm)
                .ok_or_else(|| JacError::Plan(format!("no band field `{nm}`")))
        };
        // A domain plan addresses a state cell as its block start plus its
        // column-major flat index over the declared shape, so the source's
        // block must be dense over exactly that shape (no halo, offset box
        // or strides).
        let shapes: HashMap<&str, Vec<usize>> = scatter
            .states
            .iter()
            .map(|(u, sh)| (u.as_str(), sh.iter().map(|&n| n.max(0) as usize).collect()))
            .collect();
        let domain = matches!(exec, Exec::Domain);
        let state = |u: &str, row: bool| -> R<&SlotInfo> {
            let nm = self.prefixed(u);
            match theirs.vars.get(&nm) {
                Some(s) if domain && s.dense.as_ref() != shapes.get(u) => {
                    Err(JacError::Plan(format!(
                        "`{nm}`'s block in the source program is not dense column-major over \
                         its declared shape {:?} (storage {:?})",
                        shapes.get(u),
                        s.dense
                    )))
                }
                Some(s) if s.written || !row => Ok(s),
                Some(_) => Err(JacError::Plan(format!(
                    "band row `{nm}` is a state the source program reads but does not write"
                ))),
                None => Err(JacError::Plan(format!(
                    "state `{nm}` is not in the source program (a source from another build?)"
                ))),
            }
        };
        // A lane layout is one template per item, `l` lanes wide: lane `ℓ` is
        // the `ℓ`-th lane-domain cell in a domain plan too, and every field
        // and state is contiguous over the lanes. So is a column layout
        // whose axis is the last dimension (columns enumerate the others).
        let template = match &scatter.kind {
            PlanKind::LaneBlocks { .. } | PlanKind::LaneSparseBlocks { .. } => true,
            PlanKind::ColumnBanded {
                cells, axis_dim, ..
            } => matches!(exec, Exec::Lanes { .. }) || *axis_dim + 1 == cells.len(),
            PlanKind::Csr { .. } => false,
        };
        let width = if template { width.max(l) } else { width };
        let fields: Vec<usize> = (0..scatter.bands.len())
            .map(|k| field(k).map(|s| s.start))
            .collect::<R<_>>()?;
        let field = |k: usize| -> R<usize> { Ok(fields[k]) };
        match &scatter.kind {
            PlanKind::LaneBlocks {
                block_vars,
                items: it,
                ..
            } if template => {
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
            PlanKind::LaneSparseBlocks {
                block_vars,
                lu_pattern,
                items: it,
                ..
            } if template => {
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
            PlanKind::ColumnBanded {
                cells,
                axis_dim,
                vars,
                kl,
                items: it,
                ..
            } if template => {
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
            items
                .iter()
                .map(|it| (it.slab as u32, it.out as u32))
                .collect()
        } else {
            Vec::new()
        };
        let single: Vec<bool> = (0..pairs.len())
            .map(|i| {
                let o = pairs[i].1;
                (i == 0 || pairs[i - 1].1 != o) && pairs.get(i + 1).is_none_or(|n| n.1 != o)
            })
            .collect();
        let (singles, pairs): (Vec<_>, Vec<_>) =
            pairs.into_iter().zip(single).partition(|(_, one)| *one);
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
                io,
            }),
        })
    }

    /// The lane sets of a lane plan: every dimension of the states' common
    /// shape, less `axis` for a column plan.
    fn lane_sets(&self, scatter: &ScatterPlan, axis: Option<&String>) -> Vec<String> {
        let dims = scatter
            .states
            .iter()
            .find(|(_, s)| !s.is_empty())
            .and_then(|(u, _)| self.inner.dims.get(u))
            .cloned()
            .unwrap_or_default();
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
            slab: vec![0.0; p.io.ours_u_len],
            aux: vec![0.0; p.io.ours_aux_len],
            p: vec![0.0; p.io.p_map.as_ref().map_or(0, Vec::len)],
            acc: vec![0.0; p.width],
            forcing: None,
            evaluated: false,
        }
    }

    fn check(&self, plan: &JacPlan, ws: &JacWorkspace) -> R<()> {
        if plan.inner.owner != self.inner.id {
            return Err(JacError::Call(
                "the plan was made by another program".into(),
            ));
        }
        if ws.plan != plan.inner.id {
            return Err(JacError::Call(
                "the workspace was made for another plan".into(),
            ));
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
    pub fn eval_into<T: StorageScalar>(
        &self,
        plan: &JacPlan,
        u: &[f64],
        aux: &[f64],
        p: &[f64],
        t: f64,
        forcing: &ForcingSet,
        cells: &[i32],
        n: usize,
        out: &mut [T],
        ws: &mut JacWorkspace,
    ) -> R<()> {
        self.check(plan, ws)?;
        let pl = &*plan.inner;
        ws.evaluated = false;
        if out.len() != pl.out_len {
            return Err(JacError::Call(format!(
                "out has length {}; the plan needs {}",
                out.len(),
                pl.out_len
            )));
        }
        eval_slab(&pl.io, &pl.exec, u, aux, p, t, forcing, cells, n, ws)?;
        let JacWorkspace { slab, acc, .. } = ws;
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
            return Err(JacError::Call(
                "no band slab: call eval_into first (a failed \
                                       evaluation leaves none)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// `y += J·v` over the band slab the last [`Self::eval_into`] left in
    /// `ws`, nothing assembled. `v` is the source's `u ++ aux` (every state a
    /// column may name), `y` its `u`. Each element is updated in `f64` in
    /// plan order and rounded once per contribution.
    pub fn apply<T: StorageScalar>(
        &self,
        plan: &JacPlan,
        ws: &JacWorkspace,
        v: &[T],
        y: &mut [T],
    ) -> R<()> {
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
    pub fn apply_transpose(
        &self,
        plan: &JacPlan,
        ws: &JacWorkspace,
        lambda: &[f64],
        gbar: &mut [f64],
    ) -> R<()> {
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
        let (nu, na) = (pl.io.src_u_len, pl.io.src_aux_len);
        if cols != nu + na || rows != nu {
            return Err(JacError::Call(format!(
                "matrix-free vectors have lengths (columns {cols}, rows {rows}); the plan needs \
                 ({}, {})",
                nu + na,
                nu
            )));
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
fn position_cells<'a>(
    s: &'a ScatterPlan,
    canon: &Canon<'a>,
    pos: usize,
    l: usize,
) -> ((&'a str, usize), (&'a str, usize)) {
    match &s.kind {
        PlanKind::LaneBlocks { block_vars, .. } => {
            let n = block_vars.len();
            let (off, ln) = (pos / l, pos % l);
            ((&block_vars[off % n], ln), (&block_vars[off / n], ln))
        }
        PlanKind::LaneSparseBlocks {
            block_vars,
            lu_pattern,
            ..
        } => {
            let (r, c) = lu_pattern[pos / l];
            ((&block_vars[r], pos % l), (&block_vars[c], pos % l))
        }
        PlanKind::ColumnBanded {
            cells,
            axis_dim,
            vars,
            kl,
            ..
        } => {
            let nn = cells[*axis_dim] as usize;
            let (off, ln) = (pos / l, pos % l);
            let lev = off % nn;
            let (r, c, d) = column_decode(off, nn, vars.len(), *kl, scatter_w(s));
            let colev = (lev as i64 + d) as usize;
            (
                (
                    &vars[r],
                    ScatterPlan::column_cell(cells, *axis_dim, lev, ln),
                ),
                (
                    &vars[c],
                    ScatterPlan::column_cell(cells, *axis_dim, colev, ln),
                ),
            )
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
    /// The block's extents when it is dense column-major over them (a cell's
    /// slot is `start` + its column-major flat index); `None` for a strided
    /// block.
    dense: Option<Vec<usize>>,
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
                let ext: Vec<usize> = b
                    .storage
                    .iter()
                    .map(|(lo, hi)| (hi - lo + 1).max(0) as usize)
                    .collect();
                let dense = b
                    .strides
                    .iter()
                    .zip(&ext)
                    .try_fold(1usize, |st, (&s, &n)| (s == st || n <= 1).then_some(st * n))
                    .map(|_| ext);
                vars.insert(
                    b.name.clone(),
                    SlotInfo {
                        start: base + b.base,
                        len: b.len(),
                        written,
                        dense,
                    },
                );
            }
        }
        Slots {
            vars,
            u_len: t.u_len,
            aux_len: t.aux_len,
        }
    }

    /// The lane layout of a program with tile layout `t` laned over `sets`
    /// at `capacity` (RFC U3 §6): its blocks in order, each a column-major
    /// `[lane, other axes…]` block over the variable's axes outside `sets`.
    fn lanes_of(
        t: &earthsci_ast::TileLayout,
        sets: &[String],
        capacity: usize,
    ) -> Result<Slots, String> {
        let mut vars = HashMap::new();
        let mut lens = [0usize; 2];
        for (blocks, written, which) in [(&t.u, true, 0), (&t.aux, false, 1)] {
            for b in blocks {
                if let Some(s) = sets.iter().find(|s| !b.sets.contains(s)) {
                    return Err(format!(
                        "the source state `{}` does not carry the lane set \
                                        `{s}`",
                        b.name
                    ));
                }
                let ext: Vec<usize> = std::iter::once(capacity)
                    .chain(
                        b.sets
                            .iter()
                            .zip(&b.storage)
                            .filter(|(s, _)| !sets.contains(s))
                            .map(|(_, (lo, hi))| (hi - lo + 1).max(0) as usize),
                    )
                    .collect();
                let len = ext.iter().product::<usize>();
                vars.insert(
                    b.name.clone(),
                    SlotInfo {
                        start: lens[which],
                        len,
                        written,
                        dense: Some(ext),
                    },
                );
                lens[which] += len;
            }
        }
        // aux blocks follow u in the concatenated buffer
        for s in vars.values_mut() {
            if !s.written {
                s.start += lens[0];
            }
        }
        Ok(Slots {
            vars,
            u_len: lens[0],
            aux_len: lens[1],
        })
    }

    fn lane(t: &earthsci_ast::LaneLayout) -> Slots {
        let l = t.capacity;
        let mut vars = HashMap::new();
        for (blocks, written, base) in [(&t.u, true, 0), (&t.aux, false, t.u_len)] {
            for b in blocks {
                vars.insert(
                    b.name.clone(),
                    SlotInfo {
                        start: base + b.offset * l,
                        len: l * b.extents.iter().product::<usize>(),
                        written,
                        dense: Some(
                            std::iter::once(l)
                                .chain(b.extents.iter().copied())
                                .collect(),
                        ),
                    },
                );
            }
        }
        Slots {
            vars,
            u_len: t.u_len,
            aux_len: t.aux_len,
        }
    }
}

/// Evaluate a band program's fields at a source program's point into
/// `ws.slab`: the source's `u`, `aux`, `p` and forcing set read through `io`.
/// Leaves `ws.evaluated` false; the caller sets it once its own step is done.
#[allow(clippy::too_many_arguments)]
fn eval_slab(
    io: &Inputs,
    exec: &PlanExec,
    u: &[f64],
    aux: &[f64],
    p: &[f64],
    t: f64,
    forcing: &ForcingSet,
    cells: &[i32],
    n: usize,
    ws: &mut JacWorkspace,
) -> R<()> {
    ws.evaluated = false;
    let len = |what: &str, want: usize, got: usize| -> R<()> {
        if want != got {
            return Err(JacError::Call(format!(
                "{what} has length {got}; the plan needs \
                                               {want}"
            )));
        }
        Ok(())
    };
    len("u", io.src_u_len, u.len())?;
    len("aux", io.src_aux_len, aux.len())?;
    len("p", io.n_src_params, p.len())?;
    let JacWorkspace {
        rhs,
        slab,
        aux: our_aux,
        p: our_p,
        forcing: fcache,
        ..
    } = ws;
    let zeros = &io.zeros[..];
    let aux_in: &[f64] = match &io.aux_map {
        AuxMap::SourceU => u,
        AuxMap::Copy(m) => {
            let su = u.len();
            for &(src, dst, n) in m {
                let from = if src < su {
                    &u[src..src + n]
                } else {
                    &aux[src - su..src - su + n]
                };
                our_aux[dst..dst + n].copy_from_slice(from);
            }
            our_aux
        }
    };
    let p_in: &[f64] = match &io.p_map {
        None => p,
        Some(m) => {
            for (d, &s) in our_p.iter_mut().zip(m) {
                *d = p[s];
            }
            our_p
        }
    };
    let f_in: &ForcingSet = match &io.forcing_map {
        None => forcing,
        Some(m) => {
            if fcache.as_ref().map(|c| c.0) != Some(forcing.epoch()) {
                if forcing.slots() != &io.src_forcing[..] {
                    return Err(JacError::Call(
                        "the forcing set is not one for the source \
                                               program's slots"
                            .into(),
                    ));
                }
                let bufs = m.iter().map(|&j| Arc::from(forcing.buffer(j))).collect();
                let base = match &exec {
                    PlanExec::Domain(_, b) | PlanExec::Lanes(_, b) => b,
                };
                *fcache = Some((forcing.epoch(), ForcingSet::new(base, bufs)?));
            }
            &fcache.as_ref().expect("filled above").1
        }
    };
    match &exec {
        PlanExec::Domain(prog, _) => {
            if !cells.is_empty() || n != 0 {
                return Err(JacError::Call("a domain plan takes no cells".into()));
            }
            prog.eval_into(zeros, aux_in, p_in, t, f_in, slab, rhs)?;
        }
        PlanExec::Lanes(lp, _) => {
            lp.eval_into(zeros, aux_in, p_in, t, f_in, cells, n, slab, rhs)?
        }
    }
    Ok(())
}

/// A band program's inputs bound to a source program: where each call's
/// `u`, `aux`, `p` and forcing set are read from (see
/// [`JacobianProgram::plan`]).
struct Inputs {
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
    /// The source program's forcing slots, when `forcing_map` reads them.
    src_forcing: Vec<ForcingSlot>,
    /// The band program's `u` (its own band fields, never read), shared by
    /// every workspace of the plan.
    zeros: Arc<[f64]>,
}

/// Bind the band program `bprog` (laid out as `ours`) to `source` (laid out
/// as `theirs`), by name: every state it reads must be in `source` with the
/// same extent, and every parameter and forcing slot one of `source`'s.
fn bind_inputs(bprog: &RhsProgram, ours: &Slots, theirs: &Slots, source: &RhsProgram) -> R<Inputs> {
    let perr = |m: String| -> R<Inputs> { Err(JacError::Plan(m)) };
    // The band program's own inputs, gathered from the source's.
    let mut gather = Vec::new();
    for (name, s) in &ours.vars {
        if s.written {
            continue;
        }
        let Some(t) = theirs.vars.get(name) else {
            return perr(format!(
                "the band fields read `{name}`, which the source program \
                                 neither writes nor reads"
            ));
        };
        if t.len != s.len {
            return perr(format!(
                "`{name}` holds {} values in the source program and {} in \
                                 the band program (a source from another build?)",
                t.len, s.len
            ));
        }
        gather.push((t.start, s.start - ours.u_len, s.len));
    }
    gather.sort_by_key(|g| g.1);
    // Reading the source's `u` as is: every input at its own offset.
    let identity = ours.aux_len == theirs.u_len
        && gather.iter().map(|g| g.2).sum::<usize>() == ours.aux_len
        && gather.iter().all(|g| g.0 == g.1);
    let aux_map = if identity {
        AuxMap::SourceU
    } else {
        AuxMap::Copy(gather)
    };

    // Parameters by name.
    let ours_p = bprog.parameters();
    let theirs_p = source.parameters();
    let p_map = if ours_p
        .iter()
        .map(|p| &p.name)
        .eq(theirs_p.iter().map(|p| &p.name))
    {
        None
    } else {
        let at: HashMap<&String, usize> = theirs_p
            .iter()
            .enumerate()
            .map(|(i, p)| (&p.name, i))
            .collect();
        let mut m = Vec::with_capacity(ours_p.len());
        for p in ours_p {
            match at.get(&p.name) {
                Some(i) => m.push(*i),
                None => {
                    return perr(format!(
                        "parameter `{}` of the band program is not a \
                                             parameter of the source program",
                        p.name
                    ))
                }
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
            match theirs_f
                .iter()
                .position(|g| g.name == f.name && g.shape == f.shape && g.kind == f.kind)
            {
                Some(i) => m.push(i),
                None => {
                    return perr(format!(
                        "forcing slot `{}` {:?} ({:?}) of the band \
                                             program is not a forcing slot of the source \
                                             program",
                        f.name, f.shape, f.kind
                    ))
                }
            }
        }
        Some(m)
    };
    Ok(Inputs {
        ours_u_len: ours.u_len,
        ours_aux_len: ours.aux_len,
        src_u_len: theirs.u_len,
        src_aux_len: theirs.aux_len,
        n_src_params: theirs_p.len(),
        aux_map,
        p_map,
        src_forcing: if forcing_map.is_some() {
            theirs_f.to_vec()
        } else {
            Vec::new()
        },
        forcing_map,
        zeros: vec![0.0; ours.u_len].into(),
    })
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
    io: Inputs,
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
        (self.inner.io.src_u_len, self.inner.io.src_aux_len)
    }
}

/// The per-call buffers of one plan: the band program's workspace and the
/// band slab. Reused across calls, so a warm call with an unchanged forcing
/// set allocates nothing. `Send`, not `Sync`: one per thread.
pub struct JacWorkspace {
    plan: u64,
    rhs: RhsWorkspace,
    slab: Vec<f64>,
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
        f.debug_struct("JacWorkspace")
            .field("slab", &self.slab.len())
            .finish()
    }
}

// ── the second-order contraction (spec §4.4) ────────────────────────────────

/// A model's second-order bands for one axis pair ([`crate::jacobian_bands2`]),
/// their coefficient fields compiled once — the production form of
/// [`crate::numeric::HvpEvaluator`], which stays the oracle it is tested
/// against.
///
/// Not a new compile target: each second-order entry rides through the
/// first-order build as a band whose column indices are `v`'s followed by
/// `w`'s (the evaluation document reads only rows, row and contracted
/// indices and the coefficient), so the band fields `J_k` are compiled
/// exactly as [`JacobianProgram::build`] compiles them. What differs is the
/// plan: [`HvpProgram::plan`] resolves each band point to a (slab, row,
/// column, second column) item of the source program's layout instead of a
/// storage layout, and [`HvpProgram::contract`] walks those items.
///
/// Separate from [`JacobianProgram`] because its plans are: a
/// [`ScatterPlan`] reads every column index as a state column, which a
/// concatenated `cidx` is not, and a second-order band list may be empty
/// (a piecewise-linear operator), which the first-order build refuses.
#[derive(Clone)]
pub struct HvpProgram {
    inner: Arc<HvpInner>,
}

struct HvpInner {
    id: u64,
    model_name: Option<String>,
    model: String,
    wrt: (Wrt, Wrt),
    entries: Vec<JacEntry2>,
    /// Every variable's declared shape (scalars absent), all a plan needs
    /// of the document.
    shapes: HashMap<String, Vec<i64>>,
    /// The compiled document (fused band form), its band-field names and
    /// programs; `None` for an empty band list, which compiles nothing.
    doc: Option<(Value, Vec<String>, Compiled)>,
}

impl fmt::Debug for HvpProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HvpProgram")
            .field("model", &self.inner.model)
            .field("wrt", &self.inner.wrt)
            .field("bands", &self.inner.entries.len())
            .finish()
    }
}

/// One contraction item: the band slab slot, the row (source `u`), the first
/// column (source `u ++ aux`, or 0 on a time pair) and the second column
/// (`g`'s index).
#[derive(Clone, Copy, Debug)]
struct HvpItem {
    slab: usize,
    row: usize,
    col: usize,
    col2: usize,
}

/// An [`HvpProgram`] bound to the integrator's program for the same
/// operator (see [`HvpProgram::plan`]). Immutable, `Send + Sync`, cheap to
/// clone.
#[derive(Clone)]
pub struct HvpPlan {
    inner: Arc<HvpPlanInner>,
}

struct HvpPlanInner {
    id: u64,
    owner: u64,
    wrt: (Wrt, Wrt),
    /// The executor and its input binding; `None` for an empty band list.
    run: Option<(PlanExec, Inputs)>,
    src_u_len: usize,
    src_aux_len: usize,
    n_src_params: usize,
    w_len: usize,
    g_len: usize,
    /// In plan order: entries in order, each one's points first index
    /// fastest (rows, then contracted cells).
    items: Vec<HvpItem>,
    /// The distinct (row, col) the items read `M` at, and item `k`'s index
    /// into them.
    rc: Vec<(usize, usize)>,
    rcidx: Vec<usize>,
}

impl fmt::Debug for HvpPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HvpPlan")
            .field("wrt", &self.inner.wrt)
            .field("items", &self.inner.items.len())
            .field("g_len", &self.inner.g_len)
            .finish()
    }
}

impl HvpPlan {
    /// The lengths of the source `u` and `aux` every call takes.
    pub fn source_lengths(&self) -> (usize, usize) {
        (self.inner.src_u_len, self.inner.src_aux_len)
    }

    /// The length of each stage's `w`: the source's `u ++ aux` on a
    /// (states, ·) pair, 1 (the column `t`) on a (time, ·) pair.
    pub fn w_len(&self) -> usize {
        self.inner.w_len
    }

    /// The length of `g` (see [`HvpProgram::plan`] for its indexing).
    pub fn g_len(&self) -> usize {
        self.inner.g_len
    }

    /// The `g` positions [`HvpProgram::contract`] writes in `form`, sorted:
    /// every second column a band point reaches ([`HvpForm::Contract`]), or
    /// every first column ([`HvpForm::Symmetric`]).
    pub fn touched(&self, form: HvpForm) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .inner
            .items
            .iter()
            .map(|it| match form {
                HvpForm::Contract => it.col2,
                HvpForm::Symmetric => it.col,
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// The per-call buffers of one [`HvpPlan`]: the band program's workspace
/// and slab, and the contraction's `M` scratch. `Send`, not `Sync`: one per
/// thread.
pub struct HvpWorkspace {
    plan: u64,
    ws: Option<JacWorkspace>,
    m: Vec<f64>,
    evaluated: bool,
}

impl HvpWorkspace {
    /// The band slab of the last successful [`HvpProgram::eval`] (empty for
    /// an empty band list).
    pub fn slab(&self) -> &[f64] {
        self.ws.as_ref().map_or(&[], |w| &w.slab[..])
    }
}

impl fmt::Debug for HvpWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HvpWorkspace")
            .field("slab", &self.slab().len())
            .finish()
    }
}

impl HvpProgram {
    /// Differentiate `model_name` (the only model when `None`) twice, for
    /// the axis pair `wrt` (first axis states or time, second states,
    /// parameters or forcing; `forcing` as for
    /// [`crate::jacobian_bands2_with_forcing`]), and compile the band fields.
    /// An empty second-order band list is legitimate (a piecewise-linear
    /// operator): the program compiles nothing and its contraction is a
    /// no-op.
    pub fn build(
        file: &EsmFile,
        model_name: Option<&str>,
        wrt: (Wrt, Wrt),
        forcing: Option<&[String]>,
        opts: &BuildOptions,
    ) -> R<HvpProgram> {
        let wrt = axis_pair(wrt)?;
        let model = the_model(file, model_name)?;
        let sv = sysview(file, Some(&model))?;
        let ctx = ctx_of(&sv)?;
        let obs = observed_defs(&sv)?;
        let (_, entries) = bands2_view(&sv, &ctx, &obs, wrt, forcing)?;
        let doc = if entries.is_empty() {
            None
        } else {
            let first = first_order_form(&entries);
            let (evaldoc, names) =
                evaluation_document_with(&sv, &first, opts.cse, opts.cse_min_nodes, false)?;
            if let Some(n) = names.iter().find(|n| sv.variables.contains_key(*n)) {
                return Err(JacError::Build(format!(
                    "the source model declares `{n}`, the name of a band field of its \
                     second-order bands"
                )));
            }
            let src = source_value(file)?;
            let doc = compiled_document(&src, &model, &evaldoc, &names, &first, true)?;
            let rhs = compile(&doc, &model, &names, false)?;
            Some((doc, names, Compiled::with(rhs)))
        };
        Ok(HvpProgram {
            inner: Arc::new(HvpInner {
                id: next_id(),
                model_name: model_name.map(str::to_string),
                model,
                wrt,
                entries,
                shapes: ctx.shapes.clone(),
                doc,
            }),
        })
    }

    /// The second-order bands, in `J_k` order.
    pub fn bands(&self) -> &[JacEntry2] {
        &self.inner.entries
    }

    /// The axis pair.
    pub fn wrt(&self) -> (Wrt, Wrt) {
        self.inner.wrt
    }

    /// The model this program differentiates, as passed to `build`.
    pub fn model_name(&self) -> Option<&str> {
        self.inner.model_name.as_deref()
    }

    /// The compiled band-field program (`None` for an empty band list).
    pub fn rhs(&self) -> Option<&RhsProgram> {
        self.inner.doc.as_ref().and_then(|(_, _, c)| c.data())
    }

    fn prefixed(&self, name: &str) -> String {
        format!("{}.{name}", self.inner.model)
    }

    /// A plan: every band point resolved, by name, against `source` — the
    /// integrator's program for the same operator, which `u`, `aux`, `p`
    /// and the forcing set of every call are laid out for — with the same
    /// strictness as [`JacobianProgram::plan`] (a source from another build
    /// is refused). Only [`Exec::Domain`]: a lane form is not built yet.
    ///
    /// Indexing, for a source with `u` of length `nu`, `aux` of `na` and `np`
    /// parameters:
    ///
    /// * `μ` (one per stage) is a source `u` (the rows);
    /// * `w` is a source `u ++ aux` (the first columns, as `v` of
    ///   [`JacobianProgram::apply`]), or one value (the column `t`) on a time
    ///   pair;
    /// * `g` is a source `u ++ aux` on a (·, states) pair (as `ḡ` of
    ///   [`JacobianProgram::apply_transpose`]), else the source's run-time
    ///   inputs: `p` (length `np`, a scalar parameter's cotangent at its
    ///   index) followed by every forcing slot's buffer in slot order, each
    ///   column-major (an array parameter or forcing buffer cell's cotangent
    ///   at its slot's offset). A column the source holds in neither is
    ///   refused: an array parameter without an `update` is a slot only when
    ///   the source is built with `ProblemOptions::runtime_array_parameters`
    ///   (the band fields then read the slot too).
    ///
    /// A column outside its variable's extent is a ghost read and skipped,
    /// except a second column on the forcing axis, which is refused (as in
    /// [`crate::numeric::HvpEvaluator`]).
    pub fn plan(&self, source: &RhsProgram, exec: Exec) -> R<HvpPlan> {
        let inner = &*self.inner;
        if let Exec::Lanes { .. } = exec {
            return Err(JacError::Plan(
                "a second-order contraction has no lane form yet; use Exec::Domain".into(),
            ));
        }
        let (a1, a2) = inner.wrt;
        let theirs = Slots::tile(source.layout());
        let theirs_p = source.parameters();
        let theirs_f = source.forcing_slots();
        let n_src_params = theirs_p.len();
        let run = match &inner.doc {
            Some((doc, names, rhs)) => {
                let bprog = rhs.matching(doc, &inner.model, names, source)?;
                let ours = Slots::tile(bprog.layout());
                let io = bind_inputs(bprog, &ours, &theirs, source)?;
                Some((PlanExec::Domain(bprog.clone(), bprog.clone()), io, ours))
            }
            None => None,
        };

        let perr = |m: String| JacError::Plan(m);
        let shape_of = |n: &str| inner.shapes.get(n).cloned().unwrap_or_default();
        let as_usize = |s: &[i64]| s.iter().map(|&n| n.max(0) as usize).collect::<Vec<_>>();
        // A state's block in the source's `u ++ aux`, dense column-major
        // over its declared shape.
        let state = |n: &str, row: bool| -> R<(&SlotInfo, Vec<i64>)> {
            let nm = self.prefixed(n);
            let shape = shape_of(n);
            match theirs.vars.get(&nm) {
                Some(s) if s.dense.as_deref() != Some(&as_usize(&shape)[..]) => Err(perr(format!(
                    "`{nm}`'s block in the source program is not dense column-major over its \
                     declared shape {shape:?} (storage {:?})",
                    s.dense
                ))),
                Some(s) if s.written || !row => Ok((s, shape)),
                Some(_) => Err(perr(format!(
                    "band row `{nm}` is a state the source program reads but does not write"
                ))),
                None => Err(perr(format!(
                    "state `{nm}` is not in the source program (a source from another build?)"
                ))),
            }
        };
        // Where a non-state column lives in `p ++ forcing buffers`.
        let mut slot_off = Vec::with_capacity(theirs_f.len());
        let mut off = n_src_params;
        for s in theirs_f {
            slot_off.push(off);
            off += s.len();
        }
        let input_len = off;
        let input = |n: &str, cell: &[i64]| -> R<usize> {
            let nm = self.prefixed(n);
            if let Some(i) = theirs_p.iter().position(|q| q.name == nm) {
                return Ok(i);
            }
            let Some(j) = theirs_f.iter().position(|s| s.name == nm) else {
                return Err(perr(format!(
                    "`{nm}` is neither a parameter nor a forcing slot of the source program, \
                     so its cotangent has no place in `g`: the source holds it as build-time \
                     data (a shaped parameter without an `update` is a slot only when the \
                     source is built with ProblemOptions::runtime_array_parameters), or the \
                     source is from another build"
                )));
            };
            let shape = shape_of(n);
            if theirs_f[j].shape != as_usize(&shape) {
                return Err(perr(format!(
                    "forcing slot `{nm}` has shape {:?} in the source program, {shape:?} here \
                     (a source from another build?)",
                    theirs_f[j].shape
                )));
            }
            Ok(slot_off[j] + flat(cell, &shape))
        };
        let ghost = |n: &str, cell: &[i64]| outside(inner.shapes.get(n), cell);

        let mut items = Vec::new();
        for (k, en) in inner.entries.iter().enumerate() {
            let ours = &run.as_ref().expect("a non-empty band list compiles").2;
            let nm = self.prefixed(&format!("J_{}", k + 1));
            let fshape: Vec<i64> = shape_of(&en.u)
                .into_iter()
                .chain(en.band.contracted.iter().map(|(_, _, hi)| *hi))
                .collect();
            let field = match ours.vars.get(&nm) {
                Some(f) if f.dense.as_deref() == Some(&as_usize(&fshape)[..]) => f.start,
                Some(f) => {
                    return Err(perr(format!(
                        "band field `{nm}` is not dense column-major over {fshape:?} \
                         (storage {:?})",
                        f.dense
                    )))
                }
                None => return Err(perr(format!("no band field `{nm}`"))),
            };
            let (urow, ushape) = state(&en.u, true)?;
            let mut pts: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = Vec::new();
            foreach_point(&en.band, &mut |cell, cc, _, cidx| {
                pts.push((cell.to_vec(), cc.to_vec(), cidx.to_vec()));
                Ok(())
            })?;
            for (cell, cc, cidx) in pts {
                let (c1, c2) = cidx.split_at(en.ncol);
                if ghost(&en.w, c2) {
                    if a2 == Wrt::Forcing {
                        return Err(perr(format!(
                            "second column `{}` (row `{}`) is outside the declared extent of \
                             its variable; on the forcing axis every column must resolve (no \
                             ghost skip)",
                            crate::numeric::cellname(&en.w, c2),
                            crate::numeric::cellname(&en.u, &cell)
                        )));
                    }
                    continue;
                }
                if a1 == Wrt::States && ghost(&en.v, c1) {
                    continue; // ghost read: no column
                }
                let col = match a1 {
                    Wrt::Time => 0,
                    _ => {
                        let (s, sh) = state(&en.v, false)?;
                        s.start + flat(c1, &sh)
                    }
                };
                let col2 = match a2 {
                    Wrt::States => {
                        let (s, sh) = state(&en.w, false)?;
                        s.start + flat(c2, &sh)
                    }
                    _ => input(&en.w, c2)?,
                };
                let full: Vec<i64> = cell.iter().chain(&cc).copied().collect();
                items.push(HvpItem {
                    slab: field + flat(&full, &fshape),
                    row: urow.start + flat(&cell, &ushape),
                    col,
                    col2,
                });
            }
        }
        let mut lut: HashMap<(usize, usize), usize> = HashMap::new();
        let mut rc = Vec::new();
        let rcidx = items
            .iter()
            .map(|it| {
                *lut.entry((it.row, it.col)).or_insert_with(|| {
                    rc.push((it.row, it.col));
                    rc.len() - 1
                })
            })
            .collect();
        let cols = theirs.u_len + theirs.aux_len;
        Ok(HvpPlan {
            inner: Arc::new(HvpPlanInner {
                id: next_id(),
                owner: inner.id,
                wrt: inner.wrt,
                run: run.map(|(e, io, _)| (e, io)),
                src_u_len: theirs.u_len,
                src_aux_len: theirs.aux_len,
                n_src_params,
                w_len: if a1 == Wrt::Time { 1 } else { cols },
                g_len: if a2 == Wrt::States { cols } else { input_len },
                items,
                rc,
                rcidx,
            }),
        })
    }

    /// Buffers for calls with `plan`.
    pub fn new_workspace(&self, plan: &HvpPlan) -> HvpWorkspace {
        let p = &*plan.inner;
        let ws = p.run.as_ref().map(|(exec, io)| {
            let PlanExec::Domain(prog, _) = exec else {
                unreachable!("domain plans only")
            };
            JacWorkspace {
                plan: p.id,
                rhs: prog.new_workspace(),
                slab: vec![0.0; io.ours_u_len],
                aux: vec![0.0; io.ours_aux_len],
                p: vec![0.0; io.p_map.as_ref().map_or(0, Vec::len)],
                acc: Vec::new(),
                forcing: None,
                evaluated: false,
            }
        });
        HvpWorkspace {
            plan: p.id,
            ws,
            m: vec![0.0; p.rc.len()],
            evaluated: false,
        }
    }

    fn check(&self, plan: &HvpPlan, ws: &HvpWorkspace) -> R<()> {
        if plan.inner.owner != self.inner.id {
            return Err(JacError::Call(
                "the plan was made by another program".into(),
            ));
        }
        if ws.plan != plan.inner.id {
            return Err(JacError::Call(
                "the workspace was made for another plan".into(),
            ));
        }
        Ok(())
    }

    /// Evaluate the second-order band fields at a point of the source
    /// program (`u`, `aux`, `p` and `forcing` as it takes them) into `ws`,
    /// for [`Self::contract`].
    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &self,
        plan: &HvpPlan,
        u: &[f64],
        aux: &[f64],
        p: &[f64],
        t: f64,
        forcing: &ForcingSet,
        ws: &mut HvpWorkspace,
    ) -> R<()> {
        self.check(plan, ws)?;
        let pl = &*plan.inner;
        ws.evaluated = false;
        match (&pl.run, &mut ws.ws) {
            (Some((exec, io)), Some(jw)) => {
                eval_slab(io, exec, u, aux, p, t, forcing, &[], 0, jw)?;
            }
            _ => {
                for (what, want, got) in [
                    ("u", pl.src_u_len, u.len()),
                    ("aux", pl.src_aux_len, aux.len()),
                    ("p", pl.n_src_params, p.len()),
                ] {
                    if want != got {
                        return Err(JacError::Call(format!(
                            "{what} has length {got}; the plan needs {want}"
                        )));
                    }
                }
            }
        }
        ws.evaluated = true;
        Ok(())
    }

    /// The second-order contraction over the band slab the last
    /// [`Self::eval`] left in `ws`, accumulated into `g` in `f64` in plan
    /// order (zero it first for a single product):
    ///
    /// ```text
    /// g += ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ          (pair (states, a2))
    /// g += ∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t         (pair (time, a2))
    /// ```
    ///
    /// `pairs` holds one `(μ, w)` per stage, indexed as [`Self::plan`]
    /// documents. [`HvpForm::Contract`] sums the stages into `M(row, col)`
    /// once and runs `g[col2] += J₂ · M(row, col)`; [`HvpForm::Symmetric`]
    /// (a (states, states) pair only) runs `g[col] += μ[row] · J₂ · w[col2]`
    /// per stage, with no `M`.
    pub fn contract(
        &self,
        plan: &HvpPlan,
        ws: &mut HvpWorkspace,
        pairs: &[(&[f64], &[f64])],
        form: HvpForm,
        g: &mut [f64],
    ) -> R<()> {
        self.check(plan, ws)?;
        let pl = &*plan.inner;
        if !ws.evaluated {
            return Err(JacError::Call(
                "no band slab: call eval first (a failed evaluation leaves none)".into(),
            ));
        }
        if form == HvpForm::Symmetric && pl.wrt != (Wrt::States, Wrt::States) {
            return Err(JacError::Call(format!(
                "the symmetric form needs a (states, states) pair, got ({}, {})",
                pl.wrt.0.as_str(),
                pl.wrt.1.as_str()
            )));
        }
        if g.len() != pl.g_len {
            return Err(JacError::Call(format!(
                "g has length {}; the plan needs {}",
                g.len(),
                pl.g_len
            )));
        }
        for (mu, w) in pairs {
            if mu.len() != pl.src_u_len || w.len() != pl.w_len {
                return Err(JacError::Call(format!(
                    "a stage pair has lengths (μ {}, w {}); the plan needs ({}, {})",
                    mu.len(),
                    w.len(),
                    pl.src_u_len,
                    pl.w_len
                )));
            }
        }
        let HvpWorkspace { ws: jw, m, .. } = ws;
        let slab: &[f64] = jw.as_ref().map_or(&[], |w| &w.slab[..]);
        match form {
            HvpForm::Contract => {
                m.iter_mut().for_each(|x| *x = 0.0);
                for (mu, w) in pairs {
                    for (x, &(r, c)) in m.iter_mut().zip(&pl.rc) {
                        *x += mu[r] * w[c];
                    }
                }
                for (it, &q) in pl.items.iter().zip(&pl.rcidx) {
                    g[it.col2] += slab[it.slab] * m[q];
                }
            }
            HvpForm::Symmetric => {
                for (mu, w) in pairs {
                    for it in &pl.items {
                        g[it.col] += mu[it.row] * slab[it.slab] * w[it.col2];
                    }
                }
            }
        }
        Ok(())
    }
}

/// Second-order entries as first-order ones whose column indices are `v`'s
/// followed by `w`'s: the form the evaluation document and the compiled
/// document read (rows, row and contracted indices, the coefficient).
fn first_order_form(entries: &[JacEntry2]) -> Vec<JacEntry> {
    entries
        .iter()
        .map(|en| JacEntry {
            u: en.u.clone(),
            v: en.v.clone(),
            band: en.band.clone(),
        })
        .collect()
}
