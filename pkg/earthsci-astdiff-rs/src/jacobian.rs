//! Model-level driver (the reference's jacobian.jl).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use earthsci_ast::free_variables;
use earthsci_ast::{EsmFile, Expr, ParameterUpdate, VariableType};

use crate::bands::{bands, merge_bands, normalize_band, Band, BandError, Ctx};
use crate::clip::RIdx;
use crate::inline::inline_observed_cached;
use crate::recurrence::{active, materialize, recurrences, ExtendedBands};
use crate::system::{
    ctx_of, inlinable_observed, is_derivative, lhs_state, observed_defs, sysview, view_states,
    SysView,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wrt {
    States,
    Parameters,
    /// The cells of the forcing buffers (spec §4.3), record axis included.
    Forcing,
    Time,
}

impl Wrt {
    /// Parse a `wrt` string (`"states"`, `"parameters"`, `"forcing"`,
    /// `"time"`).
    pub fn parse(s: &str) -> Option<Wrt> {
        match s {
            "states" => Some(Wrt::States),
            "parameters" => Some(Wrt::Parameters),
            "forcing" => Some(Wrt::Forcing),
            "time" => Some(Wrt::Time),
            _ => None,
        }
    }

    /// Whether this axis's columns are buffer cells indexed by the canonical
    /// column space ([`jacobian_columns`]): the parameters and forcing axes.
    pub fn has_columns(&self) -> bool {
        matches!(self, Wrt::Parameters | Wrt::Forcing)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Wrt::States => "states",
            Wrt::Parameters => "parameters",
            Wrt::Forcing => "forcing",
            Wrt::Time => "time",
        }
    }
}

/// One block of the Jacobian: row variable `u`, column variable `v`, and the
/// [`Band`] giving the rows it covers, the column-index expressions and the
/// coefficient.
#[derive(Clone, Debug)]
pub struct JacEntry {
    pub u: String,
    pub v: String,
    pub band: Band,
}

/// The forcing set of a system, sorted by name: the parameters whose cells
/// are the columns of the [`Wrt::Forcing`] axis (spec §4.3). With `forcing =
/// None` the set is inferred — every ARRAY-valued parameter with an `update`
/// of kind `"data"` (a live buffer a data provider refills). An explicit list
/// is what a planner that builds the providers itself passes, and every name
/// in it must be a parameter of the system: a name that resolves to no buffer
/// is an error, never an empty axis.
pub fn forcing_names(sv: &SysView, forcing: Option<&[String]>) -> Result<Vec<String>, BandError> {
    let mut names: Vec<String> = match forcing {
        None => sv
            .variables
            .iter()
            .filter(|(_, v)| {
                v.var_type == VariableType::Parameter
                    && v.shape.as_ref().is_some_and(|s| !s.is_empty())
                    && v.update.as_ref().is_some_and(|u| {
                        u.rules()
                            .iter()
                            .any(|r| matches!(r, ParameterUpdate::Data { .. }))
                    })
            })
            .map(|(n, _)| n.clone())
            .collect(),
        Some(list) => {
            for n in list {
                if !sv
                    .variables
                    .get(n)
                    .is_some_and(|v| v.var_type == VariableType::Parameter)
                {
                    return Err(BandError(format!(
                        "forcing name `{n}` resolves to no buffer: it is not a \
                         parameter of the system"
                    )));
                }
            }
            list.to_vec()
        }
    };
    names.sort();
    names.dedup();
    Ok(names)
}

/// The column variables of one axis, in canonical (name-sorted) order: every
/// parameter outside the forcing set (parameters), the forcing set
/// (forcing), the solver-slot states (states) or the single site `t`.
pub(crate) fn targets(
    sv: &SysView,
    wrt: Wrt,
    forcing: Option<&[String]>,
    obs: &HashMap<String, Expr>,
) -> Result<Vec<String>, BandError> {
    Ok(match wrt {
        Wrt::Time => vec!["t".to_string()],
        Wrt::States => view_states(sv, obs),
        Wrt::Forcing => forcing_names(sv, forcing)?,
        Wrt::Parameters => {
            let fset: HashSet<String> = forcing_names(sv, forcing)?.into_iter().collect();
            let mut t: Vec<String> = sv
                .variables
                .iter()
                .filter(|(n, v)| v.var_type == VariableType::Parameter && !fset.contains(*n))
                .map(|(n, _)| n.clone())
                .collect();
            t.sort();
            t
        }
    })
}

/// The column space of a parameters or forcing block: each column variable
/// with its integer shape (empty for a scalar), sorted by name — the
/// `columns` field of the emitted block (spec §4). The canonical column
/// ordinal enumerates these variables in order, each one's cells
/// column-major (first index fastest).
pub fn jacobian_columns(
    sv: &SysView,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Vec<(String, Vec<i64>)>, BandError> {
    if !wrt.has_columns() {
        return Err(BandError(format!(
            "columns are defined for wrt = parameters or forcing, got {}",
            wrt.as_str()
        )));
    }
    let ctx = ctx_of(sv)?;
    Ok(targets(sv, wrt, forcing, &HashMap::new())?
        .into_iter()
        .map(|n| {
            let shape = ctx.shapes.get(&n).cloned().unwrap_or_default();
            (n, shape)
        })
        .collect())
}

/// The canonical column names of a [`jacobian_columns`] space, in ordinal
/// order: `"k"` for a scalar, `"F[1,1]"`, `"F[2,1]"`, … (column-major) for
/// an array.
pub fn column_names(columns: &[(String, Vec<i64>)]) -> Vec<String> {
    let mut out = Vec::new();
    for (v, shape) in columns {
        if shape.is_empty() {
            out.push(v.clone());
        } else {
            let ranges: Vec<(i64, i64)> = shape.iter().map(|n| (1, *n)).collect();
            for cell in crate::points::product_first_fastest(&ranges) {
                out.push(crate::points::cellname(v, &cell));
            }
        }
    }
    out
}

/// Differentiate a model's right-hand side into Jacobian bands with respect
/// to `wrt` (states, parameters, forcing or time). `model_name` may be
/// `None` when the file has exactly one model. The forcing set is inferred
/// (see [`jacobian_bands_with_forcing`] to pass it explicitly).
///
/// One [`JacEntry`] is returned for each (row variable, column variable) pair
/// that occurs in the model; the derivative is symbolic, so it does not depend
/// on the point where it is later evaluated.
pub fn jacobian_bands(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
) -> Result<Vec<JacEntry>, BandError> {
    jacobian_bands_with_forcing(file, model_name, wrt, None)
}

/// [`jacobian_bands`] with an explicit forcing set (`None` infers it: every
/// array parameter with a `data` update). The set also decides which
/// parameters the parameters axis leaves to the forcing axis.
///
/// On the parameters and forcing axes every column index must be closed at
/// build time (literals, folded metaparameters, `const` gathers, and the
/// entry's row and contracted index names); an unclosed one is an error
/// (spec §4.3). A model that reads the forcing set only where the derivative
/// is identically zero (an `ifelse` condition, a piecewise-constant op) has
/// an empty forcing list.
pub fn jacobian_bands_with_forcing(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Vec<JacEntry>, BandError> {
    jacobian_bands_view(&sysview(file, model_name)?, wrt, forcing)
}

/// [`jacobian_bands_with_forcing`] for a view already built: the reference's
/// `jacobian_bands(sv::SysView; wrt, forcing)`. With [`crate::sysview_flat`]
/// this is how a flattened system (a coupled document) is differentiated:
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let file = earthsci_ast::load_path("coupled.esm")?;
/// let flat = earthsci_ast::flatten(&file)?;
/// let sv = earthsci_astdiff::sysview_flat(&flat);
/// let entries = earthsci_astdiff::jacobian_bands_view(&sv, earthsci_astdiff::Wrt::States, None)?;
/// # Ok(()) }
/// ```
pub fn jacobian_bands_view(
    sv: &SysView,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Vec<JacEntry>, BandError> {
    let ctx = ctx_of(sv)?;
    let obs = observed_defs(sv)?;
    bands_of_view(sv, &ctx, &obs, wrt, forcing)
}

/// The body of [`jacobian_bands`], for callers that
/// already hold the view, its context and its observed definitions.
pub(crate) fn bands_of_view(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Vec<JacEntry>, BandError> {
    jacobian_bands_seeded(sv, ctx, obs, wrt, forcing, &[])
}

/// [`bands_of_view`] with `seed` added to the inlining seed without
/// making those names targets: the second-order driver
/// ([`crate::second_order`]) differentiates these coefficients again with
/// respect to its second axis, so an observed that moves only with that axis
/// must be inlined too, or its sites stay hidden behind its name.
pub(crate) fn jacobian_bands_seeded(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    seed: &[String],
) -> Result<Vec<JacEntry>, BandError> {
    let ext = extended_pass(sv, ctx, obs, wrt, forcing, seed)?;
    if ext.recurrences.is_empty() {
        return Ok(ext.rows);
    }
    materialize(sv, ctx, &ext)
}

/// The extended first-order system of axis `wrt` (spec §4.5): every state
/// equation's bands with respect to the axis's targets and the active
/// recurrences, and each active recurrence definition's own bands. With
/// `second` the pass is a pair's first pass: seeded with the second axis's
/// targets, so the recurrences that move with either axis are active and
/// the observed that move with either are inlined (spec §4.4). Without a
/// recurrence `rows` is [`jacobian_bands_view`]'s list and the rest is
/// empty.
pub fn extended_bands(
    sv: &SysView,
    wrt: Wrt,
    forcing: Option<&[String]>,
    second: Option<Wrt>,
) -> Result<ExtendedBands, BandError> {
    let ctx = ctx_of(sv)?;
    let obs = observed_defs(sv)?;
    let seed = match second {
        Some(a2) if a2 != wrt => targets(sv, a2, forcing, &obs)?,
        _ => Vec::new(),
    };
    extended_pass(sv, &ctx, &obs, wrt, forcing, &seed)
}

/// The extended first-order system of one pass (spec §4.5): the bands of
/// every state equation with respect to the axis's targets AND the active
/// recurrences, and the bands of each active recurrence's own definition
/// (rows `V`). A recurrence is active when it moves with the targets or
/// `seed` (a pair's second axis); recurrences are never inlined (no
/// fixpoint exists), so an inactive one stays an opaque read, constant with
/// respect to every target. Without an active recurrence `rows` is exactly
/// [`bands_of_view`]'s list.
pub(crate) fn extended_pass(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    seed: &[String],
) -> Result<ExtendedBands, BandError> {
    let src_obs = obs;
    let targets = targets(sv, wrt, forcing, obs)?;
    let tset: HashSet<String> = targets.into_iter().collect();
    let recs = recurrences(sv, obs, ctx)?;
    let act = if recs.is_empty() {
        Vec::new()
    } else {
        let mut s: HashSet<String> = tset.clone();
        s.extend(seed.iter().cloned());
        active(&recs, obs, &s)
    };
    // Inline only the observed that can move with THIS `wrt`'s targets (and
    // the seed's); the rest have an identically-zero derivative and stay
    // opaque reads. Time is not pruned: it also enters implicitly through
    // time-varying data.
    let pruned;
    let mut obs = if wrt == Wrt::Time {
        obs
    } else {
        pruned = if seed.is_empty() {
            inlinable_observed(obs, sv.equations, &tset)
        } else {
            let mut seeded = tset.clone();
            seeded.extend(seed.iter().cloned());
            inlinable_observed(obs, sv.equations, &seeded)
        };
        &pruned
    };
    let unrec;
    if !recs.is_empty() {
        let mut m = obs.clone();
        for r in &recs {
            m.remove(&r.name);
        }
        unrec = m;
        obs = &unrec;
    }
    let mut cols = tset.clone();
    cols.extend(act.iter().map(|r| r.name.clone()));
    let mut entries = Vec::new();
    let mut obs_names = None;
    for eq in sv.equations {
        let Some(u) = lhs_state(eq) else {
            if is_derivative(eq) {
                return Err(unsupported_derivative(eq));
            }
            continue;
        };
        row_bands(&mut entries, &u, &eq.rhs, obs, &mut obs_names, &cols, ctx)?;
    }
    let mut rec_rows = Vec::new();
    for r in &act {
        let def = &src_obs[&r.name];
        row_bands(&mut rec_rows, &r.name, def, obs, &mut obs_names, &cols, ctx)?;
    }
    if wrt.has_columns() {
        for en in entries.iter().chain(&rec_rows) {
            if tset.contains(&en.v) {
                check_closed_columns(en)?;
            }
        }
    }
    Ok(ExtendedBands {
        recurrences: act,
        rows: entries,
        rec_rows,
    })
}

/// The bands of one row variable `u` (a state, or a recurrence over its
/// own definition) with respect to every column in `cols` its inlined
/// right-hand side reads, in sorted column order.
fn row_bands(
    out: &mut Vec<JacEntry>,
    u: &str,
    rhs: &Expr,
    obs: &HashMap<String, Expr>,
    obs_names: &mut Option<HashSet<String>>,
    cols: &HashSet<String>,
    ctx: &Ctx,
) -> Result<(), BandError> {
    let shape_u = ctx.shapes.get(u).cloned().unwrap_or_default();
    let rhs = inline_observed_cached(rhs, obs, obs_names).map_err(|e| BandError(e.0))?;
    // Structural occurrence gate, driven from the equation: intersect the
    // RHS's free variables with the column set (a per-equation scan of
    // every target is quadratic on scalar-lowered documents). Sorting the
    // hits keeps the sorted-target entry order.
    let mut hits: Vec<String> = free_variables(&rhs)
        .into_iter()
        .filter(|v| cols.contains(v))
        .collect();
    hits.sort();
    for v in &hits {
        for b in merge_bands(bands(&rhs, v, ctx, &shape_u)?) {
            out.push(JacEntry {
                u: u.to_string(),
                v: v.clone(),
                band: normalize_band(&b),
            });
        }
    }
    Ok(())
}

/// A column of the parameters / forcing axes is a cell of a buffer, so it
/// must be placeable at build time: after inlining, its index expressions may
/// read only the entry's own row and contracted index names (literals, folded
/// metaparameters and `const` gathers carry no free name). A column reading
/// anything else (a parameter offset, a state) cannot be placed, so the entry
/// is refused rather than emitted.
/// The second column of a pair ([`crate::second_order`]) is held to the
/// same rule.
fn check_closed_columns(en: &JacEntry) -> Result<(), BandError> {
    check_closed(&en.band, &en.band.cidx, |c| {
        format!(
            "column index {} of `{}` in the equation for `{}`",
            expr_json(c),
            en.v,
            en.u
        )
    })
}

/// [`check_closed_columns`] over the columns `cols` of band `b`; `what`
/// names a column for the refusal.
pub(crate) fn check_closed(
    b: &Band,
    cols: &[Expr],
    what: impl Fn(&Expr) -> String,
) -> Result<(), BandError> {
    let mut bound: HashSet<&str> = b
        .ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(n) => Some(n.as_str()),
            RIdx::Fixed(_) => None,
        })
        .collect();
    bound.extend(b.contracted.iter().map(|(n, _, _)| n.as_str()));
    for c in cols {
        let mut fv: Vec<String> = free_variables(c).into_iter().collect();
        fv.sort();
        if let Some(n) = fv.iter().find(|n| !bound.contains(n.as_str())) {
            return Err(BandError(format!(
                "{} reads `{n}`, which is not closed at build time (a parameter or forcing \
                 column may read only literals, folded metaparameters, const gathers and \
                 its row or contracted index names)",
                what(c)
            )));
        }
    }
    Ok(())
}

pub(crate) fn expr_json(c: &Expr) -> String {
    serde_json::to_string(&crate::helpers::ser_expr(c)).expect("expression serializes")
}

/// The first-order passes of one emission ([`crate::jacobian_blocks`]), and
/// the structure class of each. A pass depends only on its axis and the
/// inlining seed a second axis adds ([`crate::second_order`]); that seed
/// changes nothing when the second axis is the first, so such a pair shares
/// the plain pass's key: a block list asking for `States` and `(States,
/// States)` differentiates the system once. (On the time axis the seed
/// changes no inlining, but it can make a recurrence active, spec §4.5.)
#[derive(Default)]
pub(crate) struct FirstOrder {
    ext: HashMap<(Wrt, Option<Wrt>), Rc<ExtendedBands>>,
    bands: HashMap<(Wrt, Option<Wrt>), Rc<Vec<JacEntry>>>,
    structure: HashMap<(Wrt, Option<Wrt>), &'static str>,
}

impl FirstOrder {
    fn key(a1: Wrt, a2: Option<Wrt>) -> (Wrt, Option<Wrt>) {
        (a1, if a2 == Some(a1) { None } else { a2 })
    }

    /// The extended `a1` pass seeded with `seed`, the targets of `a2`
    /// (`None` for a plain first-order block).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ext(
        &mut self,
        sv: &SysView,
        ctx: &Ctx,
        obs: &HashMap<String, Expr>,
        a1: Wrt,
        a2: Option<Wrt>,
        forcing: Option<&[String]>,
        seed: &[String],
    ) -> Result<Rc<ExtendedBands>, BandError> {
        let key = Self::key(a1, a2);
        if let Some(e) = self.ext.get(&key) {
            return Ok(e.clone());
        }
        let e = Rc::new(extended_pass(sv, ctx, obs, a1, forcing, seed)?);
        self.ext.insert(key, e.clone());
        Ok(e)
    }

    /// The materialized `a1` pass ([`Self::ext`] composed through its
    /// recurrences, spec §4.5).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bands(
        &mut self,
        sv: &SysView,
        ctx: &Ctx,
        obs: &HashMap<String, Expr>,
        a1: Wrt,
        a2: Option<Wrt>,
        forcing: Option<&[String]>,
        seed: &[String],
    ) -> Result<Rc<Vec<JacEntry>>, BandError> {
        let key = Self::key(a1, a2);
        if let Some(e) = self.bands.get(&key) {
            return Ok(e.clone());
        }
        let ext = self.ext(sv, ctx, obs, a1, a2, forcing, seed)?;
        let e = Rc::new(if ext.recurrences.is_empty() {
            ext.rows.clone()
        } else {
            materialize(sv, ctx, &ext)?
        });
        self.bands.insert(key, e.clone());
        Ok(e)
    }

    /// The structure class of the pass [`Self::bands`] returned for the
    /// same `(a1, a2)`.
    pub(crate) fn structure(
        &mut self,
        a1: Wrt,
        a2: Option<Wrt>,
        e1: &[JacEntry],
        ctx: &Ctx,
    ) -> &'static str {
        self.structure
            .entry(Self::key(a1, a2))
            .or_insert_with(|| crate::emit::detect_structure(e1, ctx))
    }
}

/// The refusal for a derivative LHS that is not `D(u)` or the lifted
/// `faq{D(index(u, i…))}` (shared with the evaluation document).
pub(crate) fn unsupported_derivative(eq: &earthsci_ast::Equation) -> BandError {
    BandError(format!(
        "unsupported derivative LHS {} (a state's derivative is `D(u)` \
                       or the lifted `faq{{D(index(u, i…))}}`)",
        serde_json::to_string(&crate::helpers::ser_expr(&eq.lhs)).expect("expression serializes")
    ))
}
