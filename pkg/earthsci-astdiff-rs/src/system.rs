//! Uniform system view over one model of a loaded `EsmFile` (the reference's
//! system.jl; template expansion already happened inside `load`).

use std::collections::{HashMap, HashSet};

use earthsci_ast::extension::types::IndexSet;
use earthsci_ast::{free_variables, Equation, EsmFile, Expr, Model, ModelVariable,
                   VariableType};
use indexmap::IndexMap;

use crate::bands::{BandError, Ctx};

pub struct SysView<'a> {
    pub variables: &'a IndexMap<String, ModelVariable>,
    pub equations: &'a Vec<Equation>,
    pub index_sets: IndexMap<String, IndexSet>,
}

fn models(file: &EsmFile) -> Result<&IndexMap<String, Model>, BandError> {
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
/// `faq{ expr: D(index(u, i…)) }` form.
///
/// Only the esm 1.1.0 `faq` tag is matched: EarthSciAST's loader rewrites the
/// deprecated `aggregate` alias to `faq` at the wire boundary. Matching the old
/// tag instead silently classifies every array state equation as
/// non-differential (zero bands, no error) — the reference's faq_tag_test.jl.
pub fn lhs_state(eq: &Equation) -> Option<String> {
    let Expr::Operator(l) = &eq.lhs else { return None };
    if l.op == "faq" {
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

/// The variable an equation DEFINES by a bare-variable LHS (`y ~ f(…)`).
pub fn lhs_definition(eq: &Equation) -> Option<&String> {
    match &eq.lhs {
        Expr::Variable(n) => Some(n),
        _ => None,
    }
}

/// Observed-variable definitions, for inlining. esm 1.0.0 removed the
/// `observed` declared type and `variable.expression`: an observed is an
/// unknown some equation names bare on the left, and that no `D(·)` equation
/// also targets (mirrors `EarthSciAST::observed_definitions`). First
/// definition wins.
pub fn observed_defs(sv: &SysView) -> HashMap<String, Expr> {
    let states: HashSet<String> = sv.equations.iter()
        .filter_map(lhs_state).collect();
    let mut defs = HashMap::new();
    for eq in sv.equations {
        let Some(n) = lhs_definition(eq) else { continue };
        if states.contains(n) {
            continue;
        }
        let unknown = sv.variables.get(n)
            .is_some_and(|v| v.var_type == VariableType::Unknown);
        if unknown && !defs.contains_key(n) {
            defs.insert(n.clone(), eq.rhs.clone());
        }
    }
    defs
}

/// The unknowns occupying a solver slot: every unknown MINUS the observed
/// ones (the 1.0.0 replacement for the 0.x `state` type; an algebraic unknown
/// still counts as a slot). Sorted.
pub fn view_states(sv: &SysView) -> Vec<String> {
    let obs = observed_defs(sv);
    let mut out: Vec<String> = sv.variables.iter()
        .filter(|(n, v)| v.var_type == VariableType::Unknown
                && !obs.contains_key(*n))
        .map(|(n, _)| n.clone()).collect();
    out.sort();
    out
}

// An expression whose true dependencies `free_variables` cannot see: a
// surviving template reference exposes only its bindings. Treated as
// target-DEPENDENT so it is kept (and loudly rejected by the inliner) rather
// than silently pruned to zero.
fn opaque_deps(e: &Expr) -> bool {
    let Expr::Operator(n) = e else { return false };
    if n.op == "apply_expression_template" {
        return true;
    }
    let mut found = false;
    n.for_each_child(&mut |c: &Expr| found = found || opaque_deps(c));
    found
}

// Names occurring under an index SUBSCRIPT — `args[1..]` of an `index` node.
fn index_arg_names(acc: &mut HashSet<String>, e: &Expr) {
    let Expr::Operator(n) = e else { return };
    if n.op == "index" {
        if let Some(a0) = n.args.first() {
            index_arg_names(acc, a0);
        }
        for x in n.args.iter().skip(1) {
            acc.extend(free_variables(x));
        }
        return;
    }
    n.for_each_child(&mut |c: &Expr| index_arg_names(acc, c));
}

/// The subset of the observed definitions the band calculus has to inline
/// (the reference's `inlinable_observed`, system.jl). Kept: (1) the
/// reflexive-transitive READER closure of the observed that name a `targets`
/// member (everything else is constant w.r.t. every target — derivative
/// identically zero); and (2) any observed reached through an index
/// subscript, whose value a column index needs at build time. Joint fixpoint.
/// The seed MUST be this `wrt`'s targets: an emissions field is
/// state-independent but not parameter-independent.
pub fn inlinable_observed(obs: &HashMap<String, Expr>, equations: &[Equation],
                          targets: &HashSet<String>) -> HashMap<String, Expr> {
    let mut readers: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut work: Vec<&str> = Vec::new();
    for (n, e) in obs {
        let fv = free_variables(e);
        if !fv.is_disjoint(targets) || opaque_deps(e) {
            work.push(n);
        }
        for m in &fv {
            if let Some((k, _)) = obs.get_key_value(m) {
                readers.entry(k.as_str()).or_default().push(n);
            }
        }
    }
    let mut keep: HashSet<&str> = work.iter().copied().collect();
    while let Some(m) = work.pop() {                  // (1) reader closure
        for &n in readers.get(m).map(|v| v.as_slice()).unwrap_or(&[]) {
            if keep.insert(n) {
                work.push(n);
            }
        }
    }
    loop {                                            // (2) subscript fixpoint
        let mut acc = HashSet::new();
        for eq in equations {
            index_arg_names(&mut acc, &eq.rhs);
        }
        for n in &keep {
            index_arg_names(&mut acc, &obs[*n]);
        }
        let fresh: Vec<&str> = acc.iter()
            .filter_map(|n| obs.get_key_value(n).map(|(k, _)| k.as_str()))
            .filter(|n| !keep.contains(n)).collect();
        if fresh.is_empty() {
            break;
        }
        keep.extend(fresh);
    }
    keep.into_iter().map(|n| (n.to_string(), obs[n].clone())).collect()
}
