//! Model-level driver (the reference's jacobian.jl).

use std::collections::HashSet;

use earthsci_ast::free_variables;
use earthsci_ast::{EsmFile, VariableType};

use crate::bands::{bands, merge_bands, normalize_band, Band, BandError};
use crate::inline::inline_observed;
use crate::system::{ctx_of, inlinable_observed, lhs_state, observed_defs, sysview,
                    view_states};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrt {
    States,
    Parameters,
    Time,
}

impl Wrt {
    pub fn as_str(&self) -> &'static str {
        match self {
            Wrt::States => "states",
            Wrt::Parameters => "parameters",
            Wrt::Time => "time",
        }
    }
}

#[derive(Clone, Debug)]
pub struct JacEntry {
    pub u: String,
    pub v: String,
    pub band: Band,
}

pub fn jacobian_bands(file: &EsmFile, model_name: Option<&str>, wrt: Wrt)
                      -> Result<Vec<JacEntry>, BandError> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let targets: Vec<String> = match wrt {
        Wrt::Time => vec!["t".to_string()],
        Wrt::States => view_states(&sv),
        Wrt::Parameters => {
            let mut t: Vec<String> = sv.variables.iter()
                .filter(|(_, v)| v.var_type == VariableType::Parameter)
                .map(|(n, _)| n.clone()).collect();
            t.sort();
            t
        }
    };
    let tset: HashSet<String> = targets.into_iter().collect();
    // Inline only the observed that can move with THIS `wrt`'s targets; the
    // rest have an identically-zero derivative and stay opaque reads. Time is
    // not pruned: it also enters implicitly through time-varying data.
    let mut obs = observed_defs(&sv);
    if wrt != Wrt::Time {
        obs = inlinable_observed(&obs, sv.equations, &tset);
    }
    let mut entries = Vec::new();
    for eq in sv.equations {
        let Some(u) = lhs_state(eq) else { continue };
        let shape_u = ctx.shapes.get(&u).cloned().unwrap_or_default();
        let rhs = inline_observed(&eq.rhs, &obs)
            .map_err(|e| BandError(e.0))?;
        // Structural occurrence gate, driven from the equation: intersect the
        // RHS's free variables with the target set (a per-equation scan of
        // every target is quadratic on scalar-lowered documents). Sorting the
        // hits keeps the sorted-target entry order.
        let mut hits: Vec<String> = free_variables(&rhs).into_iter()
            .filter(|v| tset.contains(v)).collect();
        hits.sort();
        for v in &hits {
            for b in merge_bands(bands(&rhs, v, &ctx, &shape_u)?) {
                entries.push(JacEntry {
                    u: u.clone(),
                    v: v.clone(),
                    band: normalize_band(&b),
                });
            }
        }
    }
    Ok(entries)
}
