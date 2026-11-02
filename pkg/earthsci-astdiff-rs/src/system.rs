//! Uniform system view over the two differentiable inputs, one model of a
//! loaded `EsmFile` ([`sysview`]) and a coupled document's
//! `FlattenedSystem` ([`sysview_flat`]) (the reference's system.jl; template
//! expansion already happened inside `load`).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use earthsci_ast::extension::types::IndexSet;
use earthsci_ast::{
    free_variables, Domain, Equation, EsmFile, Expr, ExpressionNode, FlattenedSystem, Model,
    ModelVariable, VariableType,
};
use indexmap::IndexMap;

use crate::bands::{BandError, Ctx};

/// A model as the band calculus sees it: its variables, equations and index
/// sets. Build one with [`sysview`] (one model of a document) or
/// [`sysview_flat`] (a flattened system).
///
/// Each variable is borrowed from its model or flattened system, except a
/// pointwise-lifted state of a flattened system, which the view owns with
/// its grid shape (see [`sysview_flat`]).
pub struct SysView<'a> {
    pub variables: IndexMap<String, Cow<'a, ModelVariable>>,
    pub equations: &'a Vec<Equation>,
    pub index_sets: IndexMap<String, IndexSet>,
    /// The document's (or flattened system's) domain, `None` when it has
    /// none: the evaluation document carries its time origin (spec §8.1).
    pub domain: Option<&'a Domain>,
}

fn models(file: &EsmFile) -> Result<&IndexMap<String, Model>, BandError> {
    file.models
        .as_ref()
        .ok_or_else(|| BandError("document has no models".into()))
}

pub fn only_model(file: &EsmFile) -> Result<&String, BandError> {
    let m = models(file)?;
    if m.len() != 1 {
        return Err(BandError(format!(
            "document has {} models; pass model_name",
            m.len()
        )));
    }
    Ok(m.keys().next().unwrap())
}

/// View a model of `file` for differentiation. `model_name` may be `None`
/// when the file has exactly one model.
pub fn sysview<'a>(file: &'a EsmFile, model_name: Option<&str>) -> Result<SysView<'a>, BandError> {
    let name = match model_name {
        Some(n) => n.to_string(),
        None => only_model(file)?.clone(),
    };
    let model: &Model = models(file)?
        .get(&name)
        .ok_or_else(|| BandError(format!("no model `{name}`")))?;
    Ok(SysView {
        variables: model
            .variables
            .iter()
            .map(|(n, v)| (n.clone(), Cow::Borrowed(v)))
            .collect(),
        equations: &model.equations,
        index_sets: file.index_sets.clone().unwrap_or_default(),
        domain: file.domain.as_ref(),
    })
}

/// View a flattened system (EarthSciAST's `flatten` of a coupled document)
/// for differentiation: the reference's `sysview(flat::FlattenedSystem)`.
///
/// The variables are the union of the system's state variables, parameters
/// and observed variables, merged in that order: a name in more than one map
/// (an arrayed observed is both a state variable and an observed variable)
/// keeps its first position and takes the last map's declaration, as the
/// reference and the Python binding merge them. The equations and index sets
/// are the system's own, in its order, so every band function gives the same
/// entries, in the same order, as it gives for a one-model document holding
/// these variables and equations.
///
/// A state the pointwise spatial lift (esm-spec §10.5) put on a grid takes
/// the shape `flatten` records on it: from EarthSciAST 0.8, every binding's
/// `flatten` names the lift's index sets there (esm-libraries-spec §4.7.5,
/// "Pointwise lift: index sets": an operand's declared shape at the lift's
/// loops, else the one index set of that size, else a synthetic
/// `_liftdim<d>_<extent>` it adds to `index_sets`), so the view uses the
/// lifted state's declaration as it stands. Rust's `flatten` before 0.8
/// keeps the declared scalar shape and records only the extents, in
/// `lifted_shapes`; for such a state alone the view falls back to rules 2
/// and 3 of that section (it cannot apply rule 1), which agree with 0.8
/// whenever each extent names one index set or none.
///
/// The reference expands surviving `apply_expression_template` references
/// first (`expand_flattened_refs`); there is nothing to expand here, since
/// EarthSciAST's Rust `load` expands every reference before `flatten` runs.
pub fn sysview_flat(flat: &FlattenedSystem) -> SysView<'_> {
    let mut variables: IndexMap<String, Cow<'_, ModelVariable>> = IndexMap::new();
    for (n, v) in flat
        .state_variables
        .iter()
        .chain(&flat.parameters)
        .chain(&flat.observed_variables)
    {
        variables.insert(n.clone(), Cow::Borrowed(v));
    }
    let mut index_sets = flat.index_sets.clone();
    for (name, extents) in &flat.lifted_shapes {
        let Some(var) = variables.get_mut(name) else {
            continue;
        };
        if var.shape.as_ref().is_some_and(|s| !s.is_empty()) {
            continue; // the shape `flatten` recorded (EarthSciAST >= 0.8)
        }
        let axes = extents
            .iter()
            .enumerate()
            .map(|(d, &n)| lift_axis(&mut index_sets, d + 1, n))
            .collect();
        var.to_mut().shape = Some(axes);
    }
    SysView {
        variables,
        equations: &flat.equations,
        index_sets,
        domain: flat.domain.as_ref(),
    }
}

/// The fallback for a `flatten` that records no lifted shape (EarthSciAST
/// before 0.8): the index set naming grid dimension `d` (1-based) of extent
/// `n` of a lifted state is the one declared set of that size, else the
/// synthetic dense `_liftdim<d>_<n>` (added to `index_sets` once).
fn lift_axis(index_sets: &mut IndexMap<String, IndexSet>, d: usize, n: i64) -> String {
    let mut sized = index_sets
        .iter()
        .filter(|(k, s)| s.size == Some(n) && !k.starts_with("_liftdim"))
        .map(|(k, _)| k);
    if let (Some(k), None) = (sized.next(), sized.next()) {
        return k.clone();
    }
    let name = format!("_liftdim{d}_{n}");
    if !index_sets.contains_key(&name) {
        let set = serde_json::from_value(serde_json::json!({"kind": "interval", "size": n}))
            .expect("an interval index set deserializes");
        index_sets.insert(name.clone(), set);
    }
    name
}

pub fn ctx_of(sv: &SysView) -> Result<Ctx, BandError> {
    let mut ctx = Ctx::default();
    for (n, s) in &sv.index_sets {
        if let Some(sz) = s.size {
            ctx.index_sets.insert(n.clone(), sz);
        }
    }
    for (n, var) in sv.variables.iter() {
        if let Some(shape) = &var.shape {
            if !shape.is_empty() {
                let dims: Result<Vec<i64>, BandError> = shape
                    .iter()
                    .map(|x| {
                        if let Ok(i) = x.parse::<i64>() {
                            Ok(i)
                        } else {
                            ctx.index_sets.get(x).copied().ok_or_else(|| {
                                BandError(format!("unknown index set `{x}` in shape"))
                            })
                        }
                    })
                    .collect();
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
    let Expr::Operator(l) = &eq.lhs else {
        return None;
    };
    if l.op == "faq" {
        if let Some(Expr::Operator(d)) = l.expr.as_deref() {
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
        return None;
    }
    if l.op == "D" {
        if let Some(Expr::Variable(u)) = l.args.first() {
            return Some(u.clone());
        }
    }
    None
}

/// Whether an equation's LHS is a time derivative of any form — including
/// ones [`lhs_state`] does not recognize (`D(index(u, 1))`), which no pass
/// supports and none may silently skip.
pub fn is_derivative(eq: &Equation) -> bool {
    let Expr::Operator(l) = &eq.lhs else {
        return false;
    };
    l.op == "D"
        || (l.op == "faq" && matches!(l.expr.as_deref(), Some(Expr::Operator(d)) if d.op == "D"))
}

/// The variable a non-derivative equation defines: a bare LHS (`y ~ …`) or
/// an indexed one (`y[i…] ~ …`, lifted or pinned).
pub(crate) fn defined_name(eq: &Equation) -> Option<String> {
    lhs_definition(eq)
        .cloned()
        .or_else(|| indexed_definition(eq).map(|(n, _)| n.clone()))
}

/// The variable an equation DEFINES by a bare-variable LHS (`y ~ f(…)`).
pub fn lhs_definition(eq: &Equation) -> Option<&String> {
    match &eq.lhs {
        Expr::Variable(n) => Some(n),
        _ => None,
    }
}

/// An arrayed-observed definition `faq{expr: index(y, i…)} ~ f(i…)` (`y[i…]
/// ~ f(i…)` as documents spell it) rewritten to the whole-array form `y ~
/// faq{output_idx: [i…], expr: f(i…)}` the inliner β-reduces. Mirrors the
/// indexed half of `EarthSciAST::observed_definitions`. `None` when the LHS
/// is not an indexed read of a bare name; `Some((y, None))` when it is one
/// whose subscripts are not exactly the output indices (a partial or pinned
/// definition such as `y[1] ~ …`, not rewritable).
fn indexed_definition(eq: &Equation) -> Option<(&String, Option<Expr>)> {
    let Expr::Operator(l) = &eq.lhs else {
        return None;
    };
    let (faq, ix) = if l.op == "faq" {
        match l.expr.as_deref() {
            Some(Expr::Operator(ix)) if ix.op == "index" => (Some(l), ix),
            _ => return None,
        }
    } else {
        (None, l)
    };
    if ix.op != "index" {
        return None;
    }
    let Some(Expr::Variable(name)) = ix.args.first() else {
        return None;
    };
    let sub: Option<Vec<String>> = ix.args[1..]
        .iter()
        .map(|x| match x {
            Expr::Variable(n) => Some(n.clone()),
            _ => None,
        })
        .collect();
    let (Some(faq), Some(sub)) = (faq, sub) else {
        return Some((name, None));
    };
    let distinct = sub.iter().collect::<HashSet<_>>().len() == sub.len();
    if faq.output_idx.as_ref() != Some(&sub) || !distinct {
        return Some((name, None));
    }
    Some((
        name,
        Some(Expr::operator(ExpressionNode {
            op: "faq".into(),
            output_idx: Some(sub),
            expr: Some(Box::new(eq.rhs.clone())),
            ranges: faq.ranges.clone(),
            ..Default::default()
        })),
    ))
}

/// Observed-variable definitions, for inlining. esm 1.0.0 removed the
/// `observed` declared type and `variable.expression`: an observed is an
/// unknown some equation names on the left — bare, or indexed (an arrayed
/// observed, rewritten by [`indexed_definition`]) — and that no `D(·)`
/// equation also targets (mirrors `EarthSciAST::observed_definitions`).
/// First definition wins; an arrayed observed that is defined piecewise or
/// partially is rejected rather than left to become a solver slot.
pub fn observed_defs(sv: &SysView) -> Result<HashMap<String, Expr>, BandError> {
    let states: HashSet<String> = sv.equations.iter().filter_map(lhs_state).collect();
    let mut defs = HashMap::new();
    let mut indexed: HashSet<String> = HashSet::new();
    for eq in sv.equations {
        let bare = lhs_definition(eq);
        let (n, d) = match bare {
            Some(n) => (n, Some(eq.rhs.clone())),
            None => match indexed_definition(eq) {
                Some(nd) => nd,
                None => continue,
            },
        };
        let unknown = sv
            .variables
            .get(n)
            .is_some_and(|v| v.var_type == VariableType::Unknown);
        if states.contains(n) || !unknown {
            continue;
        }
        let Some(d) = d else {
            return Err(BandError(format!(
                "arrayed observed `{n}`: defining LHS subscripts must be the \
                 distinct output indices (partial definitions are not \
                 supported)"
            )));
        };
        if defs.contains_key(n) {
            if indexed.contains(n) || bare.is_none() {
                return Err(BandError(format!(
                    "arrayed observed `{n}` is defined by more than one \
                     equation (piecewise definitions are not supported)"
                )));
            }
            continue;
        }
        defs.insert(n.clone(), d);
        if bare.is_none() {
            indexed.insert(n.clone());
        }
    }
    Ok(defs)
}

/// The unknowns occupying a solver slot: every unknown MINUS the observed
/// ones `obs` ([`observed_defs`]; the 1.0.0 replacement for the 0.x `state`
/// type; an algebraic unknown still counts as a slot). Sorted.
pub fn view_states(sv: &SysView, obs: &HashMap<String, Expr>) -> Vec<String> {
    let mut out: Vec<String> = sv
        .variables
        .iter()
        .filter(|(n, v)| v.var_type == VariableType::Unknown && !obs.contains_key(*n))
        .map(|(n, _)| n.clone())
        .collect();
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
pub fn inlinable_observed(
    obs: &HashMap<String, Expr>,
    equations: &[Equation],
    targets: &HashSet<String>,
) -> HashMap<String, Expr> {
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
    while let Some(m) = work.pop() {
        // (1) reader closure
        for &n in readers.get(m).map(|v| v.as_slice()).unwrap_or(&[]) {
            if keep.insert(n) {
                work.push(n);
            }
        }
    }
    // (2) subscript fixpoint. The equations are scanned once; each pass then
    // scans only the bodies it just added, since nothing else can change.
    let mut acc = HashSet::new();
    for eq in equations {
        index_arg_names(&mut acc, &eq.rhs);
    }
    let mut added: Vec<&str> = keep.iter().copied().collect();
    loop {
        for n in &added {
            index_arg_names(&mut acc, &obs[*n]);
        }
        added = acc
            .iter()
            .filter_map(|n| obs.get_key_value(n).map(|(k, _)| k.as_str()))
            .filter(|n| !keep.contains(n))
            .collect();
        if added.is_empty() {
            break;
        }
        keep.extend(added.iter().copied());
    }
    keep.into_iter()
        .map(|n| (n.to_string(), obs[n].clone()))
        .collect()
}
