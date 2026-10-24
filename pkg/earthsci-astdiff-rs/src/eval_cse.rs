//! Evaluation-model CSE (the reference's eval_cse.jl): repeated coefficient
//! subtrees become observed-array variables of the derived evaluation
//! document, each defined by a `makearray` over the union of its use regions,
//! and every occurrence becomes an `index` read of it. Byte-gated through the
//! `*.evaldoc.json` goldens, so the naming, axis order and region order follow
//! the reference exactly.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::free_variables;
use earthsci_ast::Expr;
use serde_json::{json, Map, Value};

use crate::clip::RIdx;
use crate::emit::cse_templates;
use crate::helpers::{map_expr_children, op, ser_expr};
use crate::jacobian::JacEntry;
use crate::system::SysView;

/// The hoisted observed of the derived document.
pub struct Hoisted {
    /// Variable name → serialized variable declaration, in hoisting order.
    pub obs: Vec<(String, Value)>,
    /// Variable name → serialized defining RHS (a bare-LHS equation), same order.
    pub obs_eqs: Vec<(String, Value)>,
    /// Index-set additions (axis name → size).
    pub axes: Vec<(String, i64)>,
    /// The entries' coefficients, rewritten to read the hoisted observed.
    pub coefs: Vec<Expr>,
}

fn template_refs(e: &Expr, acc: &mut HashSet<String>) {
    let Expr::Operator(n) = e else { return };
    if n.op == "apply_expression_template" {
        if let Some(nm) = &n.name {
            acc.insert(nm.clone());
        }
    }
    n.for_each_child(&mut |c: &Expr| template_refs(c, acc));
}

fn refs_of(e: &Expr) -> HashSet<String> {
    let mut acc = HashSet::new();
    template_refs(e, &mut acc);
    acc
}

fn deref(e: &Expr, readmap: &HashMap<String, Expr>) -> Expr {
    let Expr::Operator(n) = e else { return e.clone() };
    if n.op == "apply_expression_template" {
        if let Some(r) = n.name.as_ref().and_then(|nm| readmap.get(nm)) {
            return r.clone();
        }
    }
    map_expr_children(e, &mut |x| deref(x, readmap))
}

/// Extract repeated coefficient subtrees (≥ `min_nodes` nodes, ≥ 2
/// occurrences) as observed of the derived document. `None` when there is
/// nothing to hoist, or when a template's free index names are not bound by
/// every band that uses it (the caller then emits the plain document).
pub fn hoist_observed(entries: &[JacEntry], sv: &SysView, min_nodes: usize)
                      -> Option<Hoisted> {
    let coefs: Vec<Expr> = entries.iter().map(|en| en.band.coef.clone()).collect();
    // Hoisted names must be fresh in the derived model's variable namespace.
    let mut pfx = "Jcse".to_string();
    while sv.variables.keys().any(|k| k.starts_with(&pfx)) {
        pfx.push('x');
    }
    let (templates, rewritten) = cse_templates(&coefs, min_nodes, &pfx);
    if templates.is_empty() {
        return None;
    }
    let body: HashMap<&str, &Expr> =
        templates.iter().map(|(n, b)| (n.as_str(), b)).collect();
    let tnames: Vec<&str> = templates.iter().map(|(n, _)| n.as_str()).collect();

    // Per entry: index name → (lo, hi) — row dims plus free contracted dims.
    let boxof: Vec<HashMap<String, (i64, i64)>> = entries.iter().map(|en| {
        let mut d: HashMap<String, (i64, i64)> = en.band.ridx.iter().enumerate()
            .filter_map(|(k, r)| match r {
                RIdx::Name(nm) => Some((nm.clone(), en.band.rows[k])),
                RIdx::Fixed(_) => None,
            }).collect();
        for (n, lo, hi) in &en.band.contracted {
            d.insert(n.clone(), (*lo, *hi));
        }
        d
    }).collect();
    let allnames: HashSet<String> =
        boxof.iter().flat_map(|b| b.keys().cloned()).collect();

    // Which entries evaluate each template, transitively through the
    // reference DAG.
    let refs_body: HashMap<&str, HashSet<String>> =
        tnames.iter().map(|n| (*n, refs_of(body[n]))).collect();
    let mut uses: HashMap<&str, HashSet<usize>> =
        tnames.iter().map(|n| (*n, HashSet::new())).collect();
    for (k, e) in rewritten.iter().enumerate() {
        for n in refs_of(e) {
            uses.get_mut(n.as_str()).unwrap().insert(k);
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for n in &tnames {
            for n2 in &refs_body[n] {
                let add: Vec<usize> = uses[n].iter().copied().collect();
                let tgt = uses.get_mut(n2.as_str()).unwrap();
                let before = tgt.len();
                tgt.extend(add);
                changed |= tgt.len() != before;
            }
        }
    }
    let live: Vec<&str> = tnames.iter().copied().filter(|n| !uses[n].is_empty()).collect();

    // Free row-index names per template (transitive; sorted = the axis order).
    fn s_of<'a>(n: &'a str, body: &HashMap<&str, &Expr>,
                refs_body: &HashMap<&str, HashSet<String>>,
                allnames: &HashSet<String>, memo: &mut HashMap<String, Vec<String>>)
                -> Vec<String> {
        if let Some(s) = memo.get(n) {
            return s.clone();
        }
        let mut s: HashSet<String> = free_variables(body[n]).into_iter()
            .filter(|x| allnames.contains(x)).collect();
        for n2 in &refs_body[n] {
            s.extend(s_of(n2, body, refs_body, allnames, memo));
        }
        let mut v: Vec<String> = s.into_iter().collect();
        v.sort();
        memo.insert(n.to_string(), v.clone());
        v
    }
    let mut smemo: HashMap<String, Vec<String>> = HashMap::new();
    let s: HashMap<&str, Vec<String>> = live.iter()
        .map(|n| (*n, s_of(n, &body, &refs_body, &allnames, &mut smemo))).collect();
    // Consistency: every user band must bind every axis name.
    for n in &live {
        for &k in &uses[n] {
            if !s[n].iter().all(|x| boxof[k].contains_key(x)) {
                return None;
            }
        }
    }

    // Use boxes (deduplicated, ascending entry order).
    let mut boxes: HashMap<&str, Vec<Vec<(i64, i64)>>> = HashMap::new();
    for n in &live {
        let mut ks: Vec<usize> = uses[n].iter().copied().collect();
        ks.sort();
        let mut bs: Vec<Vec<(i64, i64)>> = Vec::new();
        for k in ks {
            let b: Vec<(i64, i64)> = s[n].iter().map(|x| boxof[k][x]).collect();
            if !bs.contains(&b) {
                bs.push(b);
            }
        }
        boxes.insert(n, bs);
    }

    // Axis index sets: one per distinct extent, reusing an existing set of
    // exactly that size when the name is free.
    let mut axes: Vec<(String, i64)> = Vec::new();
    let mut existing: BTreeMap<String, i64> = sv.index_sets.iter()
        .filter_map(|(n, s)| s.size.map(|sz| (n.clone(), sz))).collect();
    let mut axis_for = |sz: i64| -> String {
        let nm = crate::emit::jcax_name(|n| existing.get(n).is_some_and(|e| *e != sz), sz);
        if !existing.contains_key(&nm) {
            axes.push((nm.clone(), sz));
            existing.insert(nm.clone(), sz);
        }
        nm
    };

    let readmap: HashMap<String, Expr> = live.iter().map(|n| {
        let r = if s[n].is_empty() {
            Expr::Variable(n.to_string())
        } else {
            let mut args = vec![Expr::Variable(n.to_string())];
            args.extend(s[n].iter().map(|x| Expr::Variable(x.clone())));
            op("index", args)
        };
        (n.to_string(), r)
    }).collect();

    let mut obs = Vec::new();
    let mut obs_eqs = Vec::new();
    for n in &live {
        let b = deref(body[n], &readmap);
        if s[n].is_empty() {
            obs.push((n.to_string(), json!({"type": "unknown"})));
            obs_eqs.push((n.to_string(), ser_expr(&b)));
            continue;
        }
        let exts: Vec<i64> = (0..s[n].len())
            .map(|j| boxes[n].iter().map(|bx| bx[j].1).max().unwrap()).collect();
        let mut regions = vec![json!(exts.iter().map(|e| json!([1, e])).collect::<Vec<_>>())];
        let mut values = vec![json!(0.0)];
        for bx in &boxes[n] {
            regions.push(json!(bx.iter().map(|(lo, hi)| json!([lo, hi])).collect::<Vec<_>>()));
            let ranges: Map<String, Value> = s[n].iter().enumerate()
                .map(|(j, x)| (x.clone(), json!([bx[j].0, bx[j].1]))).collect();
            values.push(json!({"op": "faq", "args": [], "output_idx": s[n],
                               "ranges": ranges, "expr": ser_expr(&b)}));
        }
        let shape: Vec<String> = exts.iter().map(|e| axis_for(*e)).collect();
        obs.push((n.to_string(), json!({"type": "unknown", "shape": shape})));
        obs_eqs.push((n.to_string(), json!({"op": "makearray", "args": [],
                                            "regions": regions, "values": values})));
    }
    let coefs = rewritten.iter().map(|e| deref(e, &readmap)).collect();
    Some(Hoisted { obs, obs_eqs, axes, coefs })
}
