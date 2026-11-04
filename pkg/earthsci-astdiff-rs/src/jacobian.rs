//! Model-level driver (the reference's jacobian.jl).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use earthsci_ast::free_variables;
use earthsci_ast::{EsmFile, Expr, VariableType};

use crate::bands::{bands_in, merge_bands, normalize_band, placeholder, Band, BandError, Ctx};
use crate::clip::RIdx;
use crate::hashcons::Interner;
use crate::inline::inline_observed_cached;
use crate::recurrence::{
    active, lag, materialize, materialize_eq, recurrences, transfer_names, ExtendedBands,
};
use crate::system::{
    ctx_of, data_rules, inlinable_observed, is_derivative, lhs_state, observed_defs, sysview,
    view_states, SysView,
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
                v.shape.as_ref().is_some_and(|s| !s.is_empty()) && !data_rules(v).is_empty()
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
/// let sv = earthsci_astdiff::sysview_flat(&flat)?;
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

/// One block of the structure-only Jacobian (spec §2.1): a [`JacEntry`]
/// without its coefficient.
#[derive(Clone, Debug)]
pub struct StructureEntry {
    pub u: String,
    pub v: String,
    pub band: StructureBand,
}

/// A [`Band`] without its coefficient: the rows it covers (named by
/// `ridx`), the column-index expressions, the contracted column dimensions
/// and the `where` guard, exactly as in a full band.
#[derive(Clone, Debug)]
pub struct StructureBand {
    pub rows: Vec<(i64, i64)>,
    pub ridx: Vec<RIdx>,
    pub cidx: Vec<Expr>,
    pub contracted: Vec<(String, i64, i64)>,
    pub guard: Option<Expr>,
}

impl From<JacEntry> for StructureEntry {
    fn from(en: JacEntry) -> Self {
        let b = en.band;
        StructureEntry {
            u: en.u,
            v: en.v,
            band: StructureBand {
                rows: b.rows,
                ridx: b.ridx,
                cidx: b.cidx,
                contracted: b.contracted,
                guard: b.guard,
            },
        }
    }
}

/// The structure-only entries of `wrt` (spec §2.1), the reference's
/// `jacobian_bands(sv; wrt, forcing, coefficients = false)`. Regions are
/// clipped and sites found as in [`jacobian_bands_view`], but no derivative
/// is taken, no coefficient simplified, and bands merge on (`ridx`, `cidx`,
/// `contracted`, `guard`) alone. Their points include
/// [`jacobian_bands_view`]'s, and equal them unless a full band's
/// coefficient simplifies to zero. Through an active recurrence (§4.5) the
/// pass runs in full and the coefficients are dropped.
pub fn jacobian_structure_view(
    sv: &SysView,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Vec<StructureEntry>, BandError> {
    let ctx = ctx_of(sv)?;
    let obs = observed_defs(sv)?;
    let rows = bands_seeded_in(sv, &ctx, &obs, wrt, forcing, &[], false)?;
    Ok(rows.into_iter().map(StructureEntry::from).collect())
}

/// [`jacobian_structure_view`] of a model of `file` (`model_name` may be
/// `None` when the file has exactly one), its forcing set inferred.
pub fn jacobian_structure(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
) -> Result<Vec<StructureEntry>, BandError> {
    jacobian_structure_view(&sysview(file, model_name)?, wrt, None)
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
    bands_seeded_in(sv, ctx, obs, wrt, forcing, seed, true)
}

/// [`jacobian_bands_seeded`], or with `co = false` the structure-only
/// entries (spec §2.1), each with a placeholder coefficient. Through an
/// active recurrence the pass runs in full ([`extended_pass_in`]), so its
/// entries are then merged again on structure alone (§2.1 item 2: two
/// entries the full mode kept apart for their coefficients alone coalesce)
/// and pinned (item 3).
fn bands_seeded_in(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    seed: &[String],
    co: bool,
) -> Result<Vec<JacEntry>, BandError> {
    let ext = extended_pass_in(sv, ctx, obs, wrt, forcing, seed, co)?;
    if ext.recurrences.is_empty() {
        return Ok(ext.rows);
    }
    if co {
        return materialized(ctx, Cow::Owned(ext));
    }
    // A merge group is one column of one row EQUATION, as without a
    // recurrence: two equations of one state are never merged, even where
    // one's run of a column meets the next one's.
    let (rows, eqs) = materialize_eq(ctx, &ext)?;
    let mut out = Vec::with_capacity(rows.len());
    let mut it = rows.into_iter().zip(eqs).peekable();
    while let Some((first, q)) = it.next() {
        let (u, v) = (first.u, first.v);
        let mut group = vec![first.band];
        while let Some((en, _)) = it.next_if(|(en, q2)| *q2 == q && en.v == v) {
            group.push(en.band);
        }
        for b in group.iter_mut() {
            b.coef = placeholder();
        }
        for b in merge_bands(group) {
            out.push(JacEntry {
                u: u.clone(),
                v: v.clone(),
                band: normalize_band(&b),
            });
        }
    }
    Ok(out)
}

/// The band list of an extended pass (spec §4.5): its `rows` when no
/// recurrence is active, else [`materialize`]'s composition, interned like
/// the pass's own rows. Every caller that turns a pass into entries goes
/// through here, so none can skip the interning.
pub(crate) fn materialized(
    ctx: &Ctx,
    ext: Cow<'_, ExtendedBands>,
) -> Result<Vec<JacEntry>, BandError> {
    if ext.recurrences.is_empty() {
        return Ok(match ext {
            Cow::Owned(e) => e.rows,
            Cow::Borrowed(e) => e.rows.clone(),
        });
    }
    let mut rows = materialize(ctx, &ext)?;
    intern_entries(rows.iter_mut().map(|en| &mut en.band));
    Ok(rows)
}

/// One object per distinct subtree across every band's coefficient, guard
/// and column indices (`hashcons::Interner`): the entries share copies
/// instead of each holding its own.
pub(crate) fn intern_entries<'a>(bands: impl Iterator<Item = &'a mut Band>) {
    let mut t = Interner::new();
    for b in bands {
        t.intern_band(b);
    }
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
    extended_pass_in(sv, ctx, obs, wrt, forcing, seed, true)
}

/// [`extended_pass`], or with `co = false` its structure-only bands (spec
/// §2.1), each with a placeholder coefficient. An axis with an active
/// recurrence is the exception: its composition through the transfer
/// (§4.5) reads the coefficients, so that pass runs in full and only the
/// caller drops them (spec §2.1).
#[allow(clippy::too_many_arguments)]
pub(crate) fn extended_pass_in(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    seed: &[String],
    co: bool,
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
    let co = co || !act.is_empty();
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
    let mut row_eq = Vec::new();
    let mut obs_names = None;
    for (q, eq) in sv.equations.iter().enumerate() {
        let Some(u) = lhs_state(eq) else {
            if is_derivative(eq) {
                return Err(unsupported_derivative(eq));
            }
            continue;
        };
        row_bands(
            &mut entries,
            &u,
            &eq.rhs,
            obs,
            &mut obs_names,
            &cols,
            ctx,
            co,
        )?;
        row_eq.resize(entries.len(), q);
    }
    let mut rec_rows = Vec::new();
    for r in &act {
        let def = &src_obs[&r.name];
        row_bands(
            &mut rec_rows,
            &r.name,
            def,
            obs,
            &mut obs_names,
            &cols,
            ctx,
            co,
        )?;
    }
    for en in rec_rows.iter().filter(|en| en.u == en.v) {
        // every self band is a lag back along the axis, or refused
        if let Some(r) = act.iter().find(|r| r.name == en.u) {
            lag(r, &en.band)?;
        }
    }
    if wrt.has_columns() {
        for en in entries.iter().chain(&rec_rows) {
            if tset.contains(&en.v) {
                check_closed_columns(en)?;
            }
        }
    }
    let transfer = if act.is_empty() {
        HashMap::new()
    } else {
        transfer_names(sv, &recs)
    };
    // Every coefficient of the pass references shared subtrees (one interner
    // over the state and the recurrence rows): the products and the emitter
    // read these rows directly.
    intern_entries(
        entries
            .iter_mut()
            .chain(rec_rows.iter_mut())
            .map(|en| &mut en.band),
    );
    Ok(ExtendedBands {
        recurrences: act,
        rows: entries,
        rec_rows,
        row_eq,
        transfer,
    })
}

/// The bands of one row variable `u` (a state, or a recurrence over its
/// own definition) with respect to every column in `cols` its inlined
/// right-hand side reads, in sorted column order; structure-only (spec
/// §2.1) when `co` is false.
#[allow(clippy::too_many_arguments)]
fn row_bands(
    out: &mut Vec<JacEntry>,
    u: &str,
    rhs: &Expr,
    obs: &HashMap<String, Expr>,
    obs_names: &mut Option<HashSet<String>>,
    cols: &HashSet<String>,
    ctx: &Ctx,
    co: bool,
) -> Result<(), BandError> {
    let shape_u = ctx.shapes.get(u).cloned().unwrap_or_default();
    let rhs =
        inline_observed_cached(rhs, obs, obs_names, &ctx.shapes).map_err(|e| BandError(e.0))?;
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
        let mut bs = bands_in(&rhs, v, ctx, &shape_u, co)?;
        if !co {
            // Structure-only bands merge on structure alone (spec §2.1 item
            // 2): one-hot lifting has rewritten the placeholder per band
            // (`ifelse(_oh == κ + d, 1, 0)`), so it is reset before the
            // merge, as the reference drops it (`coef = nothing`).
            for b in bs.iter_mut() {
                b.coef = placeholder();
            }
        }
        for b in merge_bands(bs) {
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
/// States)` differentiates the system once. On the time axis the seed
/// changes no inlining, so every time pair shares the plain time pass too —
/// unless the view has a recurrence, which the seed can make active (spec
/// §4.5).
#[derive(Default)]
pub(crate) struct FirstOrder {
    ext: HashMap<(Wrt, Option<Wrt>), Rc<ExtendedBands>>,
    bands: HashMap<(Wrt, Option<Wrt>), Rc<Vec<JacEntry>>>,
    structure: HashMap<(Wrt, Option<Wrt>), &'static str>,
    has_recs: Option<bool>,
}

impl FirstOrder {
    fn key(
        &mut self,
        sv: &SysView,
        ctx: &Ctx,
        obs: &HashMap<String, Expr>,
        a1: Wrt,
        a2: Option<Wrt>,
    ) -> Result<(Wrt, Option<Wrt>), BandError> {
        if a1 == Wrt::Time && a2.is_some() && self.has_recs.is_none() {
            self.has_recs = Some(!recurrences(sv, obs, ctx)?.is_empty());
        }
        Ok(self.known_key(a1, a2))
    }

    /// [`Self::key`] once the view's recurrences are known (a time pair
    /// keeps its own key until then).
    fn known_key(&self, a1: Wrt, a2: Option<Wrt>) -> (Wrt, Option<Wrt>) {
        let shared = a2 == Some(a1) || (a1 == Wrt::Time && self.has_recs == Some(false));
        (a1, if shared { None } else { a2 })
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
        let key = self.key(sv, ctx, obs, a1, a2)?;
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
        let key = self.key(sv, ctx, obs, a1, a2)?;
        if let Some(e) = self.bands.get(&key) {
            return Ok(e.clone());
        }
        let ext = self.ext(sv, ctx, obs, a1, a2, forcing, seed)?;
        let e = Rc::new(materialized(ctx, Cow::Borrowed(&*ext))?);
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
        let key = self.known_key(a1, a2);
        self.structure
            .entry(key)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashcons::{NodeRef, StructIndex};

    fn sv_of(file: &EsmFile) -> SysView<'_> {
        sysview(file, None).unwrap()
    }

    fn load(name: &str) -> EsmFile {
        let p = format!(
            "{}/../../tests/valid/{name}.esm",
            env!("CARGO_MANIFEST_DIR")
        );
        earthsci_ast::load_path(&p).unwrap()
    }

    /// Whether the bands hold exactly one object per distinct subtree.
    fn one_per_subtree<'a>(bands: impl Iterator<Item = &'a Band>) -> bool {
        let mut seen = HashSet::new();
        let mut todo: Vec<Expr> = Vec::new();
        for b in bands {
            todo.push(b.coef.clone());
            todo.extend(b.guard.clone());
            todo.extend(b.cidx.iter().cloned());
        }
        let mut nodes = Vec::new();
        while let Some(e) = todo.pop() {
            if let Expr::Operator(a) = &e {
                if seen.insert(NodeRef(a.clone())) {
                    a.for_each_child(&mut |c: &Expr| todo.push(c.clone()));
                    nodes.push(e.clone());
                }
            }
        }
        let mut ix = StructIndex::new();
        let ids: HashSet<usize> = nodes.iter().map(|n| ix.sid(n)).collect();
        ids.len() == nodes.len()
    }

    #[test]
    fn every_path_interns_its_bands() {
        // the emitter's memoized pass and the products' extended pass
        let file = load("minmod_adv");
        let sv = sv_of(&file);
        let (ctx, obs) = (ctx_of(&sv).unwrap(), observed_defs(&sv).unwrap());
        let mut memo = FirstOrder::default();
        let e = memo
            .bands(&sv, &ctx, &obs, Wrt::States, None, None, &[])
            .unwrap();
        assert!(one_per_subtree(e.iter().map(|en| &en.band)));
        let ext = extended_pass(&sv, &ctx, &obs, Wrt::States, None, &[]).unwrap();
        assert!(one_per_subtree(ext.rows.iter().map(|en| &en.band)));
        // through an active recurrence: the extended rows and the
        // materialized composition
        let file = load("recurrence_sweep");
        let sv = sv_of(&file);
        let (ctx, obs) = (ctx_of(&sv).unwrap(), observed_defs(&sv).unwrap());
        let ext = extended_pass(&sv, &ctx, &obs, Wrt::States, None, &[]).unwrap();
        assert!(!ext.recurrences.is_empty());
        assert!(one_per_subtree(
            ext.rows.iter().chain(&ext.rec_rows).map(|en| &en.band)
        ));
        let m = materialized(&ctx, Cow::Borrowed(&ext)).unwrap();
        assert!(one_per_subtree(m.iter().map(|en| &en.band)));
        // second order
        let file = load("arrhenius_chem");
        let sv = sv_of(&file);
        let (ctx, obs) = (ctx_of(&sv).unwrap(), observed_defs(&sv).unwrap());
        for wrt in [(Wrt::States, Wrt::States), (Wrt::States, Wrt::Parameters)] {
            let (_, e2) = crate::second_order::bands2_view(&sv, &ctx, &obs, wrt, None).unwrap();
            assert!(one_per_subtree(e2.iter().map(|en| &en.band)));
        }
    }
}
