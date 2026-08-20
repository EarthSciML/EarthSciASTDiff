//! §4 band-entry serialization, template CSE, structure classification,
//! factorization plans, and the block emission — golden-gated against the
//! reference byte-for-byte.

use std::collections::{BTreeMap, HashMap};

use earthsci_ast::types::{Expr, ExpressionNode};
use serde_json::{json, Value};

use crate::bands::Band;
use crate::clip::RIdx;
use crate::helpers::{map_expr_children, ser_expr, stable_json};
use crate::jacobian::{JacEntry, Wrt};
use crate::system::SysView;

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

pub fn jacobian_block(file: &earthsci_ast::types::EsmFile,
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
