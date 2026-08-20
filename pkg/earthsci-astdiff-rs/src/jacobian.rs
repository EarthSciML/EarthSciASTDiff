//! Model-level driver (the reference's jacobian.jl).

use earthsci_ast::expression::free_variables;
use earthsci_ast::types::{EsmFile, VariableType};

use crate::bands::{bands, merge_bands, normalize_band, Band, BandError};
use crate::inline::inline_observed;
use crate::system::{ctx_of, lhs_state, observed_defs, sysview};

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
        _ => {
            let want = if wrt == Wrt::States {
                VariableType::State
            } else {
                VariableType::Parameter
            };
            let mut t: Vec<String> = sv.variables.iter()
                .filter(|(_, v)| v.var_type == want)
                .map(|(n, _)| n.clone()).collect();
            t.sort();
            t
        }
    };
    let obs = observed_defs(&sv);
    let mut entries = Vec::new();
    for eq in sv.equations {
        let Some(u) = lhs_state(eq) else { continue };
        let shape_u = ctx.shapes.get(&u).cloned().unwrap_or_default();
        let rhs = inline_observed(&eq.rhs, &obs)
            .map_err(|e| BandError(e.0))?;
        let fv = free_variables(&rhs);
        for v in &targets {
            if !fv.contains(v) {
                continue;                        // structural occurrence gate
            }
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
