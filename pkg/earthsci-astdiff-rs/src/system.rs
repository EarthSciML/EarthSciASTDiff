//! Uniform system view over one model of a loaded `EsmFile` (the reference's
//! system.jl; template expansion already happened inside `load`).

use std::collections::HashMap;

use earthsci_ast::types::{EsmFile, Equation, Expr, IndexSet, Model,
                          ModelVariable, VariableType};

use crate::bands::{BandError, Ctx};

pub struct SysView<'a> {
    pub variables: &'a HashMap<String, ModelVariable>,
    pub equations: &'a Vec<Equation>,
    pub index_sets: HashMap<String, IndexSet>,
}

fn models(file: &EsmFile) -> Result<&std::collections::HashMap<String, Model>,
                                     BandError> {
    file.models.as_ref()
        .ok_or_else(|| BandError("document has no models".into()))
}

pub fn only_model(file: &EsmFile) -> Result<&String, BandError> {
    let m = models(file)?;
    if m.len() != 1 {
        return Err(BandError(format!(
            "document has {} models; pass model_name", m.len())));
    }
    Ok(m.keys().next().unwrap())
}

pub fn sysview<'a>(file: &'a EsmFile, model_name: Option<&str>)
                   -> Result<SysView<'a>, BandError> {
    let name = match model_name {
        Some(n) => n.to_string(),
        None => only_model(file)?.clone(),
    };
    let model: &Model = models(file)?.get(&name)
        .ok_or_else(|| BandError(format!("no model `{name}`")))?;
    Ok(SysView {
        variables: &model.variables,
        equations: &model.equations,
        index_sets: file.index_sets.clone().unwrap_or_default(),
    })
}

pub fn ctx_of(sv: &SysView) -> Result<Ctx, BandError> {
    let mut ctx = Ctx::default();
    for (n, s) in &sv.index_sets {
        if let Some(sz) = s.size {
            ctx.index_sets.insert(n.clone(), sz);
        }
    }
    for (n, var) in sv.variables {
        if let Some(shape) = &var.shape {
            if !shape.is_empty() {
                let dims: Result<Vec<i64>, BandError> = shape.iter().map(|x| {
                    if let Ok(i) = x.parse::<i64>() {
                        Ok(i)
                    } else {
                        ctx.index_sets.get(x).copied().ok_or_else(|| {
                            BandError(format!("unknown index set `{x}` in shape"))
                        })
                    }
                }).collect();
                ctx.shapes.insert(n.clone(), dims?);
            }
        }
        if var.var_type == VariableType::Parameter {
            ctx.params.insert(n.clone());
        }
    }
    Ok(ctx)
}

/// The state an equation integrates: plain `D(u)` or the pointwise-lifted
/// `aggregate{ expr: D(index(u, i…)) }` form.
pub fn lhs_state(eq: &Equation) -> Option<String> {
    let Expr::Operator(l) = &eq.lhs else { return None };
    if l.op == "aggregate" {
        if let Some(body) = l.expr.as_deref() {
            if let Expr::Operator(d) = body {
                if d.op == "D" {
                    if let Some(Expr::Operator(ix)) = d.args.first() {
                        if ix.op == "index" {
                            if let Some(Expr::Variable(u)) = ix.args.first() {
                                return Some(u.clone());
                            }
                        }
                    }
                }
            }
        }
        return None;
    }
    if l.op == "D" {
        if let Some(Expr::Variable(u)) = l.args.first() {
            return Some(u.clone());
        }
    }
    None
}

pub fn observed_defs(sv: &SysView) -> HashMap<String, Expr> {
    sv.variables.iter()
        .filter(|(_, v)| v.var_type == VariableType::Observed
                && v.expression.is_some())
        .map(|(n, v)| (n.clone(), v.expression.clone().unwrap()))
        .collect()
}
