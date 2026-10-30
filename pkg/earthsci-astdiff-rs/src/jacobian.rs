//! Model-level driver (the reference's jacobian.jl).

use std::collections::{HashMap, HashSet};

use earthsci_ast::free_variables;
use earthsci_ast::{EsmFile, Expr, ParameterUpdate, VariableType};

use crate::bands::{bands, merge_bands, normalize_band, Band, BandError, Ctx};
use crate::clip::RIdx;
use crate::inline::inline_observed_cached;
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
fn targets(
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
            for cell in crate::numeric::product_first_fastest(&ranges) {
                out.push(crate::numeric::cellname(v, &cell));
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
/// entry's row and contracted index names), and a forcing request whose
/// right-hand sides read the forcing set but yield no band is an error
/// rather than an empty list (spec §4.3).
pub fn jacobian_bands_with_forcing(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Vec<JacEntry>, BandError> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    jacobian_bands_view(&sv, &ctx, &obs, wrt, forcing)
}

/// The body of [`jacobian_bands`], for callers (numeric assembly) that
/// already hold the view, its context and its observed definitions.
pub(crate) fn jacobian_bands_view(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Vec<JacEntry>, BandError> {
    let targets = targets(sv, wrt, forcing, obs)?;
    let target_list = targets.clone();
    let tset: HashSet<String> = targets.into_iter().collect();
    // Inline only the observed that can move with THIS `wrt`'s targets; the
    // rest have an identically-zero derivative and stay opaque reads. Time is
    // not pruned: it also enters implicitly through time-varying data.
    let pruned;
    let obs = if wrt == Wrt::Time {
        obs
    } else {
        pruned = inlinable_observed(obs, sv.equations, &tset);
        &pruned
    };
    let mut entries = Vec::new();
    let mut read_target = false;
    let mut obs_names = None;
    for eq in sv.equations {
        let Some(u) = lhs_state(eq) else {
            if is_derivative(eq) {
                return Err(unsupported_derivative(eq));
            }
            continue;
        };
        let shape_u = ctx.shapes.get(&u).cloned().unwrap_or_default();
        let rhs =
            inline_observed_cached(&eq.rhs, obs, &mut obs_names).map_err(|e| BandError(e.0))?;
        // Structural occurrence gate, driven from the equation: intersect the
        // RHS's free variables with the target set (a per-equation scan of
        // every target is quadratic on scalar-lowered documents). Sorting the
        // hits keeps the sorted-target entry order.
        let mut hits: Vec<String> = free_variables(&rhs)
            .into_iter()
            .filter(|v| tset.contains(v))
            .collect();
        hits.sort();
        read_target |= !hits.is_empty();
        for v in &hits {
            for b in merge_bands(bands(&rhs, v, ctx, &shape_u)?) {
                entries.push(JacEntry {
                    u: u.clone(),
                    v: v.clone(),
                    band: normalize_band(&b),
                });
            }
        }
    }
    if wrt.has_columns() {
        for en in &entries {
            check_closed_columns(en)?;
        }
        if wrt == Wrt::Forcing && read_target && entries.is_empty() {
            return Err(BandError(format!(
                "the right-hand sides read the forcing set {target_list:?} but no \
                 band was produced; an empty forcing block for a document that \
                 reads forcing is refused rather than emitted"
            )));
        }
    }
    Ok(entries)
}

/// A column of the parameters / forcing axes is a cell of a buffer, so it
/// must be placeable at build time: after inlining, its index expressions may
/// read only the entry's own row and contracted index names (literals, folded
/// metaparameters and `const` gathers carry no free name). A column reading
/// anything else (a parameter offset, a state) cannot be placed, so the entry
/// is refused rather than emitted.
fn check_closed_columns(en: &JacEntry) -> Result<(), BandError> {
    let b = &en.band;
    let mut bound: HashSet<&str> = b
        .ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(n) => Some(n.as_str()),
            RIdx::Fixed(_) => None,
        })
        .collect();
    bound.extend(b.contracted.iter().map(|(n, _, _)| n.as_str()));
    for c in &b.cidx {
        let mut fv: Vec<String> = free_variables(c).into_iter().collect();
        fv.sort();
        if let Some(n) = fv.iter().find(|n| !bound.contains(n.as_str())) {
            return Err(BandError(format!(
                "column index {} of `{}` in the equation for `{}` reads `{n}`, which \
                 is not closed at build time (a parameter or forcing column may read \
                 only literals, folded metaparameters, const gathers and its row or \
                 contracted index names)",
                serde_json::to_string(&crate::helpers::ser_expr(c)).expect("expression serializes"),
                en.v,
                en.u
            )));
        }
    }
    Ok(())
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
