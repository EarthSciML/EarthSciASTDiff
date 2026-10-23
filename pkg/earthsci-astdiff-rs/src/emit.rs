//! §4 band-entry serialization, template CSE, structure classification,
//! factorization plans, and the block emission — golden-gated against the
//! reference byte-for-byte.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::{Expr, ExpressionNode, VariableType};
use serde_json::{json, Map, Value};

use crate::bands::{Band, Ctx};
use crate::clip::RIdx;
use crate::eval_cse::hoist_observed;
use crate::helpers::{map_expr_children, ser_expr, stable_json};
use crate::jacobian::{JacEntry, Wrt};
use crate::system::{lhs_state, SysView};

pub fn serialize_band(en: &JacEntry) -> Value {
    let b = &en.band;
    let mut d = serde_json::Map::new();
    d.insert("row".into(), json!(en.u));
    d.insert("col".into(), json!(en.v));
    d.insert("rows".into(),
             Value::Array(b.rows.iter()
                          .map(|(lo, hi)| json!([lo, hi])).collect()));
    d.insert("row_idx".into(),
             Value::Array(b.ridx.iter().map(|r| match r {
                 RIdx::Name(n) => json!(n),
                 RIdx::Fixed(i) => json!(i),
             }).collect()));
    d.insert("col_idx".into(),
             Value::Array(b.cidx.iter().map(ser_expr).collect()));
    d.insert("coef".into(), ser_expr(&b.coef));
    if !b.contracted.is_empty() {
        d.insert("contracted".into(),
                 Value::Array(b.contracted.iter()
                              .map(|(n, lo, hi)| json!([n, lo, hi]))
                              .collect()));
    }
    Value::Object(d)
}

// ── template CSE (cse.jl) ────────────────────────────────────────────────────

struct Intern {
    ids: HashMap<String, usize>,
    reps: Vec<Expr>,
    nnodes: Vec<usize>,
    counts: Vec<usize>,
}

impl Intern {
    fn new() -> Self {
        Intern { ids: HashMap::new(), reps: vec![], nnodes: vec![],
                 counts: vec![] }
    }

    fn intern(&mut self, e: &Expr) -> usize {
        let mut childids = Vec::new();
        if let Expr::Operator(n) = e {
            n.map_children(&mut |c: &Expr| {
                childids.push(self.intern(c));
                c.clone()
            });
        }
        let (key, nn) = self.shell_key(e, &childids);
        if let Some(&id) = self.ids.get(&key) {
            self.counts[id - 1] += 1;
            return id;
        }
        self.reps.push(e.clone());
        self.nnodes.push(nn);
        self.counts.push(1);
        let id = self.reps.len();
        self.ids.insert(key, id);
        id
    }

    fn shell_key(&self, e: &Expr, childids: &[usize]) -> (String, usize) {
        if let Expr::Operator(n) = e {
            let mut i = 0usize;
            let shell = n.map_children(&mut |_c: &Expr| {
                i += 1;
                Expr::Variable(format!("__cse⟦{}⟧", childids[i - 1]))
            });
            let nn = 1 + childids.iter().map(|&c| self.nnodes[c - 1])
                .sum::<usize>();
            (stable_json(&ser_expr(&Expr::operator(shell))), nn)
        } else {
            (stable_json(&ser_expr(e)), 1)
        }
    }
}

fn replace_id(e: &Expr, it: &Intern, target: usize, repl: &Expr)
              -> (Expr, usize) {
    let mut childids = Vec::new();
    let mut newchildren = Vec::new();
    if let Expr::Operator(n) = e {
        n.map_children(&mut |c: &Expr| {
            let (nc, cid) = replace_id(c, it, target, repl);
            childids.push(cid);
            newchildren.push(nc);
            c.clone()
        });
    }
    let (key, _) = it.shell_key(e, &childids);
    let eid = it.ids[&key];
    if eid == target {
        return (repl.clone(), eid);
    }
    let out = if !newchildren.is_empty() {
        let mut i = 0usize;
        map_expr_children(e, &mut |_c| {
            i += 1;
            newchildren[i - 1].clone()
        })
    } else {
        e.clone()
    };
    (out, eid)
}

/// Greedy largest-first extraction; deterministic (first-encounter order,
/// first-max tie-break) to match the reference byte-for-byte.
pub fn cse_templates(exprs: &[Expr], min_nodes: usize, prefix: &str)
                     -> (Vec<(String, Expr)>, Vec<Expr>) {
    let mut work: Vec<Expr> = exprs.to_vec();
    let nin = work.len();
    let mut names: Vec<String> = Vec::new();
    loop {
        let mut it = Intern::new();
        for e in &work {
            it.intern(e);
        }
        let mut best = 0usize;
        for id in 1..=it.reps.len() {
            if it.counts[id - 1] < 2 || it.nnodes[id - 1] < min_nodes {
                continue;
            }
            let Expr::Operator(n) = &it.reps[id - 1] else { continue };
            if n.op == "apply_expression_template" {
                continue;
            }
            if best == 0 || it.nnodes[id - 1] > it.nnodes[best - 1] {
                best = id;
            }
        }
        if best == 0 {
            break;
        }
        let name = format!("{prefix}{}", names.len() + 1);
        names.push(name.clone());
        let repl = Expr::operator(ExpressionNode {
            op: "apply_expression_template".into(),
            name: Some(name),
            ..Default::default()
        });
        let body = it.reps[best - 1].clone();
        work = work.iter().map(|e| replace_id(e, &it, best, &repl).0).collect();
        work.push(body);
    }
    let templates = names.iter().enumerate()
        .map(|(k, n)| (n.clone(), work[nin + k].clone())).collect();
    (templates, work[..nin].to_vec())
}

// ── structure classification (structure.jl) ─────────────────────────────────

fn is_lit(e: &Expr) -> bool {
    matches!(e, Expr::Integer(_) | Expr::Number(_))
}

fn offset_class(c: &Expr, r: &RIdx) -> &'static str {
    match r {
        RIdx::Name(rn) => {
            if matches!(c, Expr::Variable(v) if v == rn) {
                return "diag";
            }
            if let Expr::Operator(n) = c {
                if (n.op == "+" || n.op == "-") && n.args.len() == 2 {
                    let (a, b) = (&n.args[0], &n.args[1]);
                    if matches!(a, Expr::Variable(v) if v == rn) && is_lit(b) {
                        return "affine";
                    }
                    if n.op == "+" && matches!(b, Expr::Variable(v) if v == rn)
                        && is_lit(a)
                    {
                        return "affine";
                    }
                }
            }
            if is_lit(c) {
                return "fixed";
            }
            "other"
        }
        RIdx::Fixed(ri) => match c {
            Expr::Integer(i) => if i == ri { "diag" } else { "fixed" },
            Expr::Number(x) => {
                if (x.round() as i64) == *ri { "diag" } else { "fixed" }
            }
            _ => "other",
        },
    }
}

pub fn detect_structure(entries: &[JacEntry], ctx: &crate::bands::Ctx)
                        -> &'static str {
    if entries.is_empty() {
        return "empty";
    }
    let mut seen_affine = false;
    for en in entries {
        let rowarr = ctx.shapes.contains_key(&en.u);
        let colarr = ctx.shapes.contains_key(&en.v);
        if rowarr != colarr {
            return "general";
        }
        let b = &en.band;
        if b.cidx.is_empty() {
            continue;
        }
        for (c, r) in b.cidx.iter().zip(b.ridx.iter()) {
            match offset_class(c, r) {
                "other" | "fixed" => return "general",
                "affine" => seen_affine = true,
                _ => {}
            }
        }
    }
    if seen_affine { "banded" } else { "block_diagonal" }
}

// ── factorization plan (factorization.jl) ───────────────────────────────────

pub fn lu_fill(pattern: &[Vec<bool>]) -> Vec<Vec<bool>> {
    let mut p: Vec<Vec<bool>> = pattern.to_vec();
    let n = p.len();
    for k in 0..n {
        for i in (k + 1)..n {
            if p[i][k] {
                for j in (k + 1)..n {
                    p[i][j] = p[i][j] || p[k][j];
                }
            }
        }
    }
    p
}

pub fn markowitz_ordering(pattern: &[Vec<bool>]) -> Vec<usize> {
    let mut p: Vec<Vec<bool>> = pattern.to_vec();
    let n = p.len();
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut order = Vec::new();
    while !remaining.is_empty() {
        let mut best = remaining[0];
        let mut bestcost = i64::MAX;
        for &i in &remaining {
            let r = remaining.iter().filter(|&&j| p[i][j]).count() as i64 - 1;
            let c = remaining.iter().filter(|&&j| p[j][i]).count() as i64 - 1;
            let cost = r * c;
            if cost < bestcost {
                best = i;
                bestcost = cost;
            }
        }
        order.push(best);
        let rest: Vec<usize> =
            remaining.into_iter().filter(|&i| i != best).collect();
        for &i in &rest {
            if p[i][best] {
                for &j in &rest {
                    p[i][j] = p[i][j] || p[best][j];
                }
            }
        }
        remaining = rest;
    }
    order.iter().map(|&i| i + 1).collect()   // 1-based, like the reference
}

fn nzpairs(m: &[Vec<bool>]) -> Value {
    let mut pairs = Vec::new();
    for i in 0..m.len() {
        for j in 0..m[i].len() {
            if m[i][j] {
                pairs.push((i + 1, j + 1));
            }
        }
    }
    pairs.sort();
    Value::Array(pairs.into_iter().map(|(i, j)| json!([i, j])).collect())
}

pub fn block_diagonal_plan(entries: &[JacEntry]) -> Value {
    let mut vars: Vec<String> = entries.iter()
        .flat_map(|en| [en.u.clone(), en.v.clone()]).collect();
    vars.sort();
    vars.dedup();
    let slot: BTreeMap<&String, usize> =
        vars.iter().enumerate().map(|(i, v)| (v, i)).collect();
    let ns = vars.len();
    let mut p = vec![vec![false; ns]; ns];
    for i in 0..ns {
        p[i][i] = true;                       // W = I − γJ has the diagonal
    }
    for en in entries {
        p[slot[&en.u]][slot[&en.v]] = true;
    }
    let ordering = markowitz_ordering(&p);
    let idx: Vec<usize> = ordering.iter().map(|o| o - 1).collect();
    let pp: Vec<Vec<bool>> = idx.iter()
        .map(|&i| idx.iter().map(|&j| p[i][j]).collect()).collect();
    json!({
        "type": "block_diagonal_lu",
        "block_vars": idx.iter().map(|&i| vars[i].clone()).collect::<Vec<_>>(),
        "ordering": ordering,
        "pattern": nzpairs(&pp),
        "lu_pattern": nzpairs(&lu_fill(&pp)),
    })
}

// ── the block ────────────────────────────────────────────────────────────────

pub fn jacobian_block(file: &earthsci_ast::EsmFile,
                      model_name: Option<&str>, wrt: Wrt,
                      cse: bool, cse_min_nodes: usize)
                      -> Result<Value, crate::bands::BandError> {
    let entries = crate::jacobian::jacobian_bands(file, model_name, wrt)?;
    let sv: SysView = crate::system::sysview(file, model_name)?;
    let ctx = crate::system::ctx_of(&sv)?;
    let structure = detect_structure(&entries, &ctx);
    let mut ser_entries: Vec<Value> =
        entries.iter().map(serialize_band).collect();
    let mut templates: Vec<(String, Expr)> = Vec::new();
    if cse {
        let coefs: Vec<Expr> =
            entries.iter().map(|en| en.band.coef.clone()).collect();
        let (tpl, rewritten) = cse_templates(&coefs, cse_min_nodes, "jt");
        if !tpl.is_empty() {
            ser_entries = entries.iter().zip(rewritten)
                .map(|(en, coef)| {
                    let mut en2 = en.clone();
                    en2.band = Band { coef, ..en.band.clone() };
                    serialize_band(&en2)
                })
                .collect();
            templates = tpl;
        }
    }
    let mut block = serde_json::Map::new();
    block.insert("wrt".into(), json!(wrt.as_str()));
    block.insert("entries".into(), Value::Array(ser_entries));
    block.insert("structure".into(), json!(structure));
    if !templates.is_empty() {
        let mut t = serde_json::Map::new();
        for (n, body) in &templates {
            t.insert(n.clone(),
                     json!({"params": [], "body": ser_expr(body)}));
        }
        block.insert("expression_templates".into(), Value::Object(t));
    }
    if structure == "block_diagonal" {
        block.insert("factorization".into(), block_diagonal_plan(&entries));
    }
    Ok(Value::Object(block))
}

// ── the derived evaluation document (emit.jl `_evaluation_document`) ────────
//
// The source's observed definitions with its state derivatives ZEROED, plus
// one band-field state `J_k` per entry whose RHS is the band's coefficient
// field: `D(J_k) = makearray(0, band region ← faq(coef))`. Byte-gated by the
// `*.evaldoc.json` goldens.

/// A shape entry as the reference writes it: an integer extent stays an
/// integer, an index-set name a string.
fn shape_json(shape: &[String]) -> Value {
    Value::Array(shape.iter().map(|x| match x.parse::<i64>() {
        Ok(i) => json!(i),
        Err(_) => json!(x),
    }).collect())
}

fn base_document(sv: &SysView, name: &str) -> (Map<String, Value>, Map<String, Value>) {
    let mut vars = Map::new();
    let mut names: Vec<&String> = sv.variables.keys().collect();
    names.sort();
    for n in names {
        let var = &sv.variables[n];
        let mut d = Map::new();
        d.insert("type".into(), json!(if var.var_type == VariableType::Parameter {
            "parameter"
        } else {
            "unknown"
        }));
        if let Some(sh) = var.shape.as_ref().filter(|s| !s.is_empty()) {
            d.insert("shape".into(), shape_json(sh));
        }
        if let Some(def) = &var.default {
            d.insert("default".into(), serde_json::to_value(def).expect("default serializes"));
        }
        vars.insert(n.clone(), Value::Object(d));
    }
    let isets: Map<String, Value> = sv.index_sets.iter()
        .filter_map(|(n, s)| s.size.map(|sz| (n.clone(), json!({"kind": "interval", "size": sz}))))
        .collect();
    let mut doc = Map::new();
    // 1.1.0: the band coefficients may carry `faq` nodes.
    doc.insert("esm".into(), json!("1.1.0"));
    doc.insert("metadata".into(), json!({"name": name}));
    doc.insert("index_sets".into(), Value::Object(isets));
    let mut model = Map::new();
    model.insert("variables".into(), Value::Object(vars));
    (doc, model)
}

/// Every source `D(u)` equation becomes `D(u) = 0` (a zero `makearray` over
/// an array state's shape), in place of the first and with any further ones
/// dropped. The states keep their slots, so the observed and band fields read
/// them unchanged, but evaluating the document no longer recomputes `f`.
/// (Carrying them as parameters instead is unsafe: EarthSciAST caches
/// observed that read only parameters, so an in-place `u` update goes stale.)
fn zeroed_equations(sv: &SysView, ctx: &Ctx) -> Vec<Value> {
    let mut done: HashSet<String> = HashSet::new();
    let mut eqs = Vec::new();
    for eq in sv.equations {
        match lhs_state(eq) {
            None => eqs.push(json!({"lhs": ser_expr(&eq.lhs), "rhs": ser_expr(&eq.rhs)})),
            Some(u) if done.insert(u.clone()) => {
                let shape = ctx.shapes.get(&u).cloned().unwrap_or_default();
                let rhs = if shape.is_empty() {
                    json!(0.0)
                } else {
                    json!({"op": "makearray", "args": [],
                           "regions": [shape.iter().map(|n| json!([1, n])).collect::<Vec<_>>()],
                           "values": [0.0]})
                };
                eqs.push(json!({"lhs": {"op": "D", "args": [u], "wrt": "t"}, "rhs": rhs}));
            }
            Some(_) => {}
        }
    }
    eqs
}

/// A literal-size axis index set of the derived document (`jcax<size>`,
/// reused when present with that size).
fn axis_name(isets: &mut Map<String, Value>, sz: i64) -> String {
    let mut nm = format!("jcax{sz}");
    while isets.get(&nm).is_some_and(|s| s.get("size") != Some(&json!(sz))) {
        nm.push('x');
    }
    if !isets.contains_key(&nm) {
        isets.insert(nm.clone(), json!({"kind": "interval", "size": sz}));
    }
    nm
}

/// The derived evaluation document and its band-state names (`J_1`, …).
/// With `cse`, repeated coefficient subtrees are first hoisted into observed
/// ([`crate::eval_cse::hoist_observed`]).
pub fn evaluation_document(sv: &SysView, entries: &[JacEntry], cse: bool,
                           cse_min_nodes: usize)
                           -> Result<(Value, Vec<String>), crate::bands::BandError> {
    let ctx = crate::system::ctx_of(sv)?;
    let (mut doc, mut model) = base_document(sv, "JacobianEval");
    let mut eqs = zeroed_equations(sv, &ctx);
    let mut isets = match doc.remove("index_sets") {
        Some(Value::Object(m)) => m,
        _ => Map::new(),
    };
    let Some(Value::Object(vars)) = model.get_mut("variables") else { unreachable!() };
    let h = if cse { hoist_observed(entries, sv, cse_min_nodes) } else { None };
    let coefs: Vec<Expr> = match &h {
        Some(h) => h.coefs.clone(),
        None => entries.iter().map(|en| en.band.coef.clone()).collect(),
    };
    if let Some(h) = &h {
        for (nm, sz) in &h.axes {
            isets.insert(nm.clone(), json!({"kind": "interval", "size": sz}));
        }
        for (nm, vd) in &h.obs {
            vars.insert(nm.clone(), vd.clone());
        }
        for (nm, rhs) in &h.obs_eqs {
            eqs.push(json!({"lhs": nm, "rhs": rhs}));
        }
    }
    let mut names = Vec::new();
    for (k, en) in entries.iter().enumerate() {
        let b = &en.band;
        let nm = format!("J_{}", k + 1);
        names.push(nm.clone());
        let shape_u = ctx.shapes.get(&en.u).cloned().unwrap_or_default();
        let cr = &b.contracted;
        let rhs = if shape_u.is_empty() && cr.is_empty() {
            vars.insert(nm.clone(), json!({"type": "unknown", "default": 0.0}));
            ser_expr(&coefs[k])
        } else {
            // The band field spans the row dims PLUS any free contracted dims.
            let mut shape: Vec<Value> = if shape_u.is_empty() {
                Vec::new()
            } else {
                match shape_json(sv.variables[&en.u].shape.as_deref().unwrap_or(&[])) {
                    Value::Array(a) => a,
                    _ => unreachable!(),
                }
            };
            shape.extend(cr.iter().map(|(_, _, hi)| json!(axis_name(&mut isets, *hi))));
            vars.insert(nm.clone(), json!({"type": "unknown", "default": 0.0, "shape": shape}));
            let free: Vec<(usize, &String)> = b.ridx.iter().enumerate()
                .filter_map(|(k2, r)| match r {
                    RIdx::Name(s) => Some((k2, s)),
                    RIdx::Fixed(_) => None,
                }).collect();
            let mut aggnames: Vec<Value> = free.iter().map(|(_, r)| json!(r)).collect();
            aggnames.extend(cr.iter().map(|(n, _, _)| json!(n)));
            let val = if aggnames.is_empty() {
                ser_expr(&coefs[k])
            } else {
                let mut rngs = Map::new();
                for (k2, r) in &free {
                    rngs.insert((*r).clone(), json!([b.rows[*k2].0, b.rows[*k2].1]));
                }
                for (n, lo, hi) in cr {
                    rngs.insert(n.clone(), json!([lo, hi]));
                }
                json!({"op": "faq", "args": [], "output_idx": aggnames,
                       "ranges": rngs, "expr": ser_expr(&coefs[k])})
            };
            let mut full: Vec<Value> = shape_u.iter().map(|n| json!([1, n])).collect();
            full.extend(cr.iter().map(|(_, _, hi)| json!([1, hi])));
            let mut region: Vec<Value> = b.rows.iter().map(|(lo, hi)| json!([lo, hi])).collect();
            region.extend(cr.iter().map(|(_, lo, hi)| json!([lo, hi])));
            json!({"op": "makearray", "args": [], "regions": [full, region],
                   "values": [0.0, val]})
        };
        eqs.push(json!({"lhs": {"op": "D", "args": [nm], "wrt": "t"}, "rhs": rhs}));
    }
    model.insert("equations".into(), Value::Array(eqs));
    doc.insert("index_sets".into(), Value::Object(isets));
    doc.insert("models".into(), json!({"JacobianEval": Value::Object(model)}));
    Ok((Value::Object(doc), names))
}
