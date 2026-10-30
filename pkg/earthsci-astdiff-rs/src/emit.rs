//! §4 band-entry serialization, template CSE, structure classification,
//! factorization plans, and the block emission — golden-gated against the
//! reference byte-for-byte.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::{free_variables, Expr, ExpressionNode, VariableType};
use serde_json::{json, Map, Value};

use crate::bands::{Band, Ctx};
use crate::clip::RIdx;
use crate::eval_cse::hoist_observed;
use crate::helpers::{map_expr_children, ser_expr, stable_json};
use crate::jacobian::unsupported_derivative;
use crate::jacobian::{JacEntry, Wrt};
use crate::second_order::JacEntry2;
use crate::system::{defined_name, is_derivative, lhs_state, SysView};

/// One entry of the `jacobians` block, as JSON (spec §4).
pub fn serialize_band(en: &JacEntry) -> Value {
    let b = &en.band;
    let mut d = serde_json::Map::new();
    d.insert("row".into(), json!(en.u));
    d.insert("col".into(), json!(en.v));
    d.insert(
        "rows".into(),
        Value::Array(b.rows.iter().map(|(lo, hi)| json!([lo, hi])).collect()),
    );
    d.insert(
        "row_idx".into(),
        Value::Array(
            b.ridx
                .iter()
                .map(|r| match r {
                    RIdx::Name(n) => json!(n),
                    RIdx::Fixed(i) => json!(i),
                })
                .collect(),
        ),
    );
    d.insert(
        "col_idx".into(),
        Value::Array(b.cidx.iter().map(ser_expr).collect()),
    );
    d.insert("coef".into(), ser_expr(&b.coef));
    if !b.contracted.is_empty() {
        d.insert(
            "contracted".into(),
            Value::Array(
                b.contracted
                    .iter()
                    .map(|(n, lo, hi)| json!([n, lo, hi]))
                    .collect(),
            ),
        );
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
        Intern {
            ids: HashMap::new(),
            reps: vec![],
            nnodes: vec![],
            counts: vec![],
        }
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
            let nn = 1 + childids.iter().map(|&c| self.nnodes[c - 1]).sum::<usize>();
            (stable_json(&ser_expr(&Expr::operator(shell))), nn)
        } else {
            (stable_json(&ser_expr(e)), 1)
        }
    }
}

fn replace_id(e: &Expr, it: &Intern, target: usize, repl: &Expr) -> (Expr, usize) {
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
pub fn cse_templates(
    exprs: &[Expr],
    min_nodes: usize,
    prefix: &str,
) -> (Vec<(String, Expr)>, Vec<Expr>) {
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
            let Expr::Operator(n) = &it.reps[id - 1] else {
                continue;
            };
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
        work = work
            .iter()
            .map(|e| replace_id(e, &it, best, &repl).0)
            .collect();
        work.push(body);
    }
    let templates = names
        .iter()
        .enumerate()
        .map(|(k, n)| (n.clone(), work[nin + k].clone()))
        .collect();
    (templates, work[..nin].to_vec())
}

// ── structure classification (structure.jl) ─────────────────────────────────

fn is_lit(e: &Expr) -> bool {
    matches!(e, Expr::Integer(_) | Expr::Number(_))
}

pub(crate) fn offset_class(c: &Expr, r: &RIdx) -> &'static str {
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
                    if n.op == "+" && matches!(b, Expr::Variable(v) if v == rn) && is_lit(a) {
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
            Expr::Integer(i) => {
                if i == ri {
                    "diag"
                } else {
                    "fixed"
                }
            }
            Expr::Number(x) => {
                if (x.round() as i64) == *ri {
                    "diag"
                } else {
                    "fixed"
                }
            }
            _ => "other",
        },
    }
}

/// Classify the band list as `"block_diagonal"`, `"banded"` or `"general"`
/// (or `"empty"` / `"other"`), read off the column-index expressions without
/// evaluating anything. Use it to choose a matrix type.
pub fn detect_structure(entries: &[JacEntry], ctx: &crate::bands::Ctx) -> &'static str {
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
    if seen_affine {
        "banded"
    } else {
        "block_diagonal"
    }
}

// ── factorization plan (factorization.jl) ───────────────────────────────────

/// Symbolic (no-pivoting) LU: the pattern plus every fill-in entry.
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

/// A Markowitz pivot ordering of a block pattern (1-based), minimizing fill-in
/// greedily.
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
        let rest: Vec<usize> = remaining.into_iter().filter(|&i| i != best).collect();
        for &i in &rest {
            if p[i][best] {
                for &j in &rest {
                    p[i][j] = p[i][j] || p[best][j];
                }
            }
        }
        remaining = rest;
    }
    order.iter().map(|&i| i + 1).collect() // 1-based, like the reference
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

/// The per-cell block LU of a block-diagonal band list (factorization.jl's
/// `BlockDiagonalPlan`): block variables in Markowitz order, the 1-based
/// ordering, and the permuted pattern and its fill-in LU pattern.
pub struct BlockLu {
    pub block_vars: Vec<String>,
    pub ordering: Vec<usize>,
    pub pattern: Vec<Vec<bool>>,
    pub lu: Vec<Vec<bool>>,
}

/// Symbolic factorization info for a block-diagonal band list: the per-cell
/// block LU with Markowitz ordering and fill-in ([`BlockLu`]).
pub fn block_diagonal_lu(entries: &[JacEntry]) -> BlockLu {
    let mut vars: Vec<String> = entries
        .iter()
        .flat_map(|en| [en.u.clone(), en.v.clone()])
        .collect();
    vars.sort();
    vars.dedup();
    let slot: BTreeMap<&String, usize> = vars.iter().enumerate().map(|(i, v)| (v, i)).collect();
    let ns = vars.len();
    let mut p = vec![vec![false; ns]; ns];
    for i in 0..ns {
        p[i][i] = true; // W = I − γJ has the diagonal
    }
    for en in entries {
        p[slot[&en.u]][slot[&en.v]] = true;
    }
    let ordering = markowitz_ordering(&p);
    let idx: Vec<usize> = ordering.iter().map(|o| o - 1).collect();
    let pp: Vec<Vec<bool>> = idx
        .iter()
        .map(|&i| idx.iter().map(|&j| p[i][j]).collect())
        .collect();
    BlockLu {
        block_vars: idx.iter().map(|&i| vars[i].clone()).collect(),
        ordering,
        lu: lu_fill(&pp),
        pattern: pp,
    }
}

pub fn block_diagonal_plan(entries: &[JacEntry]) -> Value {
    let b = block_diagonal_lu(entries);
    json!({
        "type": "block_diagonal_lu",
        "block_vars": b.block_vars,
        "ordering": b.ordering,
        "pattern": nzpairs(&b.pattern),
        "lu_pattern": nzpairs(&b.lu),
    })
}

// ── the block ────────────────────────────────────────────────────────────────

/// The `jacobians` block for one model (spec §4): entries (template-CSE'd when
/// `cse` is set, hoisting subtrees of at least `cse_min_nodes` nodes) and —
/// on the state and time axes — the structure classification and the
/// factorization plan where one exists. A parameters or forcing block carries
/// its column space instead (`columns`, [`crate::jacobian_columns`]). The
/// forcing set is inferred; see [`jacobian_block_with_forcing`].
pub fn jacobian_block(
    file: &earthsci_ast::EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    cse: bool,
    cse_min_nodes: usize,
) -> Result<Value, crate::bands::BandError> {
    jacobian_block_with_forcing(file, model_name, wrt, None, cse, cse_min_nodes)
}

/// One block's `wrt` in a model's `jacobians` entry: a single axis
/// (first-order entries) or an axis PAIR (second-order entries, spec §4.4).
/// A [`Wrt`] converts into the single-axis form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockWrt {
    Axis(Wrt),
    Pair(Wrt, Wrt),
}

impl From<Wrt> for BlockWrt {
    fn from(w: Wrt) -> Self {
        BlockWrt::Axis(w)
    }
}

impl From<(Wrt, Wrt)> for BlockWrt {
    fn from((a1, a2): (Wrt, Wrt)) -> Self {
        BlockWrt::Pair(a1, a2)
    }
}

impl BlockWrt {
    fn label(&self) -> String {
        match self {
            BlockWrt::Axis(w) => w.as_str().to_string(),
            BlockWrt::Pair(a1, a2) => format!("[{}, {}]", a1.as_str(), a2.as_str()),
        }
    }
}

/// The blocks for several distinct axes (or axis pairs, spec §4.4) of one
/// model (`[States, Forcing]`, or [`BlockWrt`]s mixing both forms), in the
/// given order: a model's `jacobians` entry in its ARRAY form (spec §4).
/// Repeated or no axes are an error.
pub fn jacobian_blocks<W: Into<BlockWrt> + Copy>(
    file: &earthsci_ast::EsmFile,
    model_name: Option<&str>,
    wrt: &[W],
    forcing: Option<&[String]>,
    cse: bool,
    cse_min_nodes: usize,
) -> Result<Value, crate::bands::BandError> {
    let wrt: Vec<BlockWrt> = wrt.iter().map(|w| (*w).into()).collect();
    let distinct: HashSet<BlockWrt> = wrt.iter().copied().collect();
    if wrt.is_empty() || distinct.len() != wrt.len() {
        return Err(crate::bands::BandError(format!(
            "wrt must name one or more distinct axes, got {:?}",
            wrt.iter().map(BlockWrt::label).collect::<Vec<_>>()
        )));
    }
    Ok(Value::Array(
        wrt.iter()
            .map(|w| match *w {
                BlockWrt::Axis(w) => {
                    jacobian_block_with_forcing(file, model_name, w, forcing, cse, cse_min_nodes)
                }
                BlockWrt::Pair(a1, a2) => crate::second_order::jacobian_block2(
                    file,
                    model_name,
                    (a1, a2),
                    forcing,
                    cse,
                    cse_min_nodes,
                ),
            })
            .collect::<Result<_, _>>()?,
    ))
}

/// [`jacobian_block`] with an explicit forcing set (`None` infers it).
pub fn jacobian_block_with_forcing(
    file: &earthsci_ast::EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    forcing: Option<&[String]>,
    cse: bool,
    cse_min_nodes: usize,
) -> Result<Value, crate::bands::BandError> {
    let entries = crate::jacobian::jacobian_bands_with_forcing(file, model_name, wrt, forcing)?;
    let sv: SysView = crate::system::sysview(file, model_name)?;
    let ctx = crate::system::ctx_of(&sv)?;
    let mut ser_entries: Vec<Value> = entries.iter().map(serialize_band).collect();
    let mut templates: Vec<(String, Expr)> = Vec::new();
    if cse {
        let coefs: Vec<Expr> = entries.iter().map(|en| en.band.coef.clone()).collect();
        let (tpl, rewritten) = cse_templates(&coefs, cse_min_nodes, "jt");
        if !tpl.is_empty() {
            ser_entries = entries
                .iter()
                .zip(rewritten)
                .map(|(en, coef)| {
                    let mut en2 = en.clone();
                    en2.band = Band {
                        coef,
                        ..en.band.clone()
                    };
                    serialize_band(&en2)
                })
                .collect();
            templates = tpl;
        }
    }
    let mut block = serde_json::Map::new();
    block.insert("wrt".into(), json!(wrt.as_str()));
    block.insert("entries".into(), Value::Array(ser_entries));
    // A parameter or forcing block describes a rectangular matrix, which the
    // structure classes and the factorization plan do not: it carries its
    // column space instead, so a consumer can size and order the columns
    // without the source document.
    let structure = if wrt.has_columns() {
        let cols = crate::jacobian::jacobian_columns(&sv, wrt, forcing)?;
        block.insert(
            "columns".into(),
            Value::Array(
                cols.iter()
                    .map(|(n, shape)| json!({"name": n, "shape": shape}))
                    .collect(),
            ),
        );
        None
    } else {
        let structure = detect_structure(&entries, &ctx);
        block.insert("structure".into(), json!(structure));
        Some(structure)
    };
    if !templates.is_empty() {
        let mut t = serde_json::Map::new();
        for (n, body) in &templates {
            t.insert(n.clone(), json!({"params": [], "body": ser_expr(body)}));
        }
        block.insert("expression_templates".into(), Value::Object(t));
    }
    if structure == Some("block_diagonal") {
        block.insert("factorization".into(), block_diagonal_plan(&entries));
    }
    Ok(Value::Object(block))
}

// ── parsing a block back (emit.jl `parse_jacobian_block(s)`) ────────────────

fn perr(msg: impl Into<String>) -> crate::bands::BandError {
    crate::bands::BandError(msg.into())
}

/// Expand `apply_expression_template` references against the block's
/// zero-parameter `expression_templates` (bodies may reference earlier ones).
fn expand_templates(
    e: &Expr,
    tpl: &HashMap<String, Expr>,
) -> Result<Expr, crate::bands::BandError> {
    if let Expr::Operator(n) = e {
        if n.op == "apply_expression_template" {
            let name = n.name.as_deref().unwrap_or_default();
            let body = tpl
                .get(name)
                .ok_or_else(|| perr(format!("unknown expression template `{name}`")))?;
            return expand_templates(body, tpl);
        }
    }
    let mut err = None;
    let out = map_expr_children(e, &mut |c| match expand_templates(c, tpl) {
        Ok(x) => x,
        Err(er) => {
            err.get_or_insert(er);
            c.clone()
        }
    });
    err.map_or(Ok(out), Err)
}

fn parse_expr(v: &Value) -> Result<Expr, crate::bands::BandError> {
    serde_json::from_value(v.clone()).map_err(|e| perr(format!("bad expression: {e}")))
}

fn parse_pairs(v: Option<&Value>, what: &str) -> Result<Vec<(i64, i64)>, crate::bands::BandError> {
    v.and_then(Value::as_array)
        .ok_or_else(|| perr(format!("band without `{what}`")))?
        .iter()
        .map(|p| match p.as_array().map(|a| a.as_slice()) {
            Some([lo, hi]) => Ok((
                lo.as_i64().ok_or_else(|| perr("non-integer bound"))?,
                hi.as_i64().ok_or_else(|| perr("non-integer bound"))?,
            )),
            _ => Err(perr(format!("malformed `{what}` pair"))),
        })
        .collect()
}

/// One §4 band entry back from JSON (the inverse of [`serialize_band`]).
pub fn parse_band(d: &Value) -> Result<JacEntry, crate::bands::BandError> {
    let s = |k: &str| {
        d.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| perr(format!("band without `{k}`")))
    };
    let ridx = d
        .get("row_idx")
        .and_then(Value::as_array)
        .ok_or_else(|| perr("band without `row_idx`"))?
        .iter()
        .map(|r| match r {
            Value::String(n) => Ok(RIdx::Name(n.clone())),
            _ => r
                .as_i64()
                .map(RIdx::Fixed)
                .ok_or_else(|| perr("malformed `row_idx`")),
        })
        .collect::<Result<_, _>>()?;
    let cidx = d
        .get("col_idx")
        .and_then(Value::as_array)
        .ok_or_else(|| perr("band without `col_idx`"))?
        .iter()
        .map(parse_expr)
        .collect::<Result<_, _>>()?;
    let contracted = match d.get("contracted").and_then(Value::as_array) {
        None => vec![],
        Some(cs) => cs
            .iter()
            .map(|c| match c.as_array().map(|a| a.as_slice()) {
                Some([n, lo, hi]) => Ok((
                    n.as_str()
                        .ok_or_else(|| perr("malformed `contracted`"))?
                        .to_string(),
                    lo.as_i64().ok_or_else(|| perr("malformed `contracted`"))?,
                    hi.as_i64().ok_or_else(|| perr("malformed `contracted`"))?,
                )),
                _ => Err(perr("malformed `contracted`")),
            })
            .collect::<Result<_, _>>()?,
    };
    Ok(JacEntry {
        u: s("row")?,
        v: s("col")?,
        band: Band {
            rows: parse_pairs(d.get("rows"), "rows")?,
            ridx,
            cidx,
            coef: parse_expr(d.get("coef").ok_or_else(|| perr("band without `coef`"))?)?,
            contracted,
        },
    })
}

/// A block's zero-parameter `expression_templates`, by name.
fn block_templates(block: &Value) -> Result<HashMap<String, Expr>, crate::bands::BandError> {
    let mut tpl: HashMap<String, Expr> = HashMap::new();
    if let Some(ts) = block.get("expression_templates").and_then(Value::as_object) {
        for (n, t) in ts {
            let body = t
                .get("body")
                .ok_or_else(|| perr(format!("template `{n}` without body")))?;
            tpl.insert(n.clone(), parse_expr(body)?);
        }
    }
    Ok(tpl)
}

/// A block's entries through `parse`, template references in each
/// coefficient expanded.
fn block_entries<T>(
    block: &Value,
    parse: impl Fn(&Value) -> Result<T, crate::bands::BandError>,
    coef: impl Fn(&mut T) -> &mut Expr,
) -> Result<Vec<T>, crate::bands::BandError> {
    let tpl = block_templates(block)?;
    block
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| perr("block without `entries`"))?
        .iter()
        .map(|d| {
            let mut en = parse(d)?;
            if !tpl.is_empty() {
                let c = coef(&mut en);
                *c = expand_templates(c, &tpl)?;
            }
            Ok(en)
        })
        .collect()
}

/// A block's `wrt`: a string names one axis, a two-name array an axis pair
/// (spec §4.4). Required, as in the reference: a block without one is not
/// read as a state block.
fn block_wrt(block: &Value) -> Result<BlockWrt, crate::bands::BandError> {
    let axis = |v: &Value| {
        let w = v.as_str().ok_or_else(|| perr("malformed `wrt`"))?;
        Wrt::parse(w).ok_or_else(|| perr(format!("unknown wrt `{w}`")))
    };
    match block.get("wrt") {
        None => Err(perr("block without `wrt`")),
        Some(Value::Array(p)) => match p.as_slice() {
            [a1, a2] => {
                let (a1, a2) = crate::second_order::axis_pair((axis(a1)?, axis(a2)?))?;
                Ok(BlockWrt::Pair(a1, a2))
            }
            _ => Err(perr("a pair `wrt` names exactly two axes")),
        },
        Some(w) => Ok(BlockWrt::Axis(axis(w)?)),
    }
}

/// One `jacobians` block back to `(wrt, entries)`, its template references
/// expanded (the round trip is exact with or without CSE). A pair block
/// (second-order entries) is [`parse_jacobian_block2`]'s.
pub fn parse_jacobian_block(
    block: &Value,
) -> Result<(Wrt, Vec<JacEntry>), crate::bands::BandError> {
    let BlockWrt::Axis(wrt) = block_wrt(block)? else {
        return Err(perr(
            "a block whose `wrt` is a pair holds second-order entries \
             (parse_jacobian_block2)",
        ));
    };
    Ok((
        wrt,
        block_entries(block, parse_band, |en| &mut en.band.coef)?,
    ))
}

/// One pair block (spec §4.4) back to `((a1, a2), entries)`, its template
/// references expanded.
pub fn parse_jacobian_block2(
    block: &Value,
) -> Result<((Wrt, Wrt), Vec<JacEntry2>), crate::bands::BandError> {
    let BlockWrt::Pair(a1, a2) = block_wrt(block)? else {
        return Err(perr(
            "a block whose `wrt` is one axis holds first-order entries \
             (parse_jacobian_block)",
        ));
    };
    Ok((
        (a1, a2),
        block_entries(block, crate::second_order::parse_band2, |en| {
            &mut en.band.coef
        })?,
    ))
}

/// One parsed block of a model's `jacobians` entry: first-order entries of
/// one axis, or second-order entries of an axis pair (spec §4.4).
#[derive(Clone, Debug)]
pub enum ParsedBlock {
    Axis(Wrt, Vec<JacEntry>),
    Pair((Wrt, Wrt), Vec<JacEntry2>),
}

impl ParsedBlock {
    /// The block's `wrt`.
    pub fn wrt(&self) -> BlockWrt {
        match self {
            ParsedBlock::Axis(w, _) => BlockWrt::Axis(*w),
            ParsedBlock::Pair((a1, a2), _) => BlockWrt::Pair(*a1, *a2),
        }
    }
}

/// [`parse_jacobian_block`] / [`parse_jacobian_block2`] over a model's
/// `jacobians` entry, which is either one block or an array of blocks with
/// distinct `wrt` (spec §4); one [`ParsedBlock`] per block, in order.
pub fn parse_jacobian_blocks(x: &Value) -> Result<Vec<ParsedBlock>, crate::bands::BandError> {
    let one = |b: &Value| -> Result<ParsedBlock, crate::bands::BandError> {
        Ok(match block_wrt(b)? {
            BlockWrt::Axis(_) => {
                let (w, es) = parse_jacobian_block(b)?;
                ParsedBlock::Axis(w, es)
            }
            BlockWrt::Pair(..) => {
                let (w, es) = parse_jacobian_block2(b)?;
                ParsedBlock::Pair(w, es)
            }
        })
    };
    let out: Vec<ParsedBlock> = match x {
        Value::Array(bs) => bs.iter().map(one).collect::<Result<_, _>>()?,
        _ => vec![one(x)?],
    };
    let distinct: HashSet<BlockWrt> = out.iter().map(ParsedBlock::wrt).collect();
    if distinct.len() != out.len() {
        return Err(perr("a model's jacobian blocks must have distinct `wrt`"));
    }
    Ok(out)
}

// ── the derived evaluation document (emit.jl `_evaluation_document`) ────────
//
// The source's observed definitions with its state derivatives ZEROED, plus
// one band-field state `J_k` per entry whose RHS is the band's coefficient
// field: `D(J_k) = makearray(0, band region ← faq(coef))`. Byte-gated by the
// `*.evaldoc.json` goldens.

/// The source model as a document: every variable (sorted) and every
/// equation verbatim (emit.jl `_base_document`).
fn base_document(sv: &SysView, name: &str) -> (Map<String, Value>, Map<String, Value>) {
    let mut vars = Map::new();
    let mut names: Vec<&String> = sv.variables.keys().collect();
    names.sort();
    for n in names {
        let var = &sv.variables[n];
        let mut d = Map::new();
        d.insert(
            "type".into(),
            json!(if var.var_type == VariableType::Parameter {
                "parameter"
            } else {
                "unknown"
            }),
        );
        // index-set names, written as stored (the schema allows only strings)
        if let Some(sh) = var.shape.as_ref().filter(|s| !s.is_empty()) {
            d.insert("shape".into(), json!(sh));
        }
        if let Some(def) = &var.default {
            d.insert(
                "default".into(),
                serde_json::to_value(def).expect("default serializes"),
            );
        }
        vars.insert(n.clone(), Value::Object(d));
    }
    let eqs: Vec<Value> = sv
        .equations
        .iter()
        .map(|eq| json!({"lhs": ser_expr(&eq.lhs), "rhs": ser_expr(&eq.rhs)}))
        .collect();
    let isets: Map<String, Value> = sv
        .index_sets
        .iter()
        .filter_map(|(n, s)| {
            s.size
                .map(|sz| (n.clone(), json!({"kind": "interval", "size": sz})))
        })
        .collect();
    let mut doc = Map::new();
    // 1.1.0: the band coefficients may carry `faq` nodes.
    doc.insert("esm".into(), json!("1.1.0"));
    doc.insert("metadata".into(), json!({"name": name}));
    doc.insert("index_sets".into(), Value::Object(isets));
    let mut model = Map::new();
    model.insert("variables".into(), Value::Object(vars));
    model.insert("equations".into(), Value::Array(eqs));
    (doc, model)
}

/// Every source `D(u)` equation of `model` (as `base_document` serialized
/// it) becomes `D(u) = 0` (a zero `makearray` over an array state's shape),
/// in place of the first and with any further ones dropped — or, without
/// `zeroed`, is dropped outright (the states become read-only inputs). The states keep
/// their slots, so the observed and band fields read them unchanged, but
/// evaluating the document no longer recomputes `f`. Observed the
/// coefficients do not read (directly or through other observed) are dropped
/// with their declarations: they fed only `f`. (Carrying the states as
/// parameters instead is unsafe: EarthSciAST caches observed that read only
/// parameters, so an in-place `u` update goes stale.)
fn zero_source_derivatives(
    model: &mut Map<String, Value>,
    sv: &SysView,
    ctx: &Ctx,
    entries: &[JacEntry],
    zeroed: bool,
) -> Result<(), crate::bands::BandError> {
    let states: HashSet<String> = sv.equations.iter().filter_map(lhs_state).collect();
    let defined: Vec<Option<String>> = sv
        .equations
        .iter()
        .map(|eq| {
            if lhs_state(eq).is_none() {
                defined_name(eq)
            } else {
                None
            }
        })
        .collect();
    let mut reads: HashMap<&str, Vec<&Expr>> = HashMap::new(); // observed → its RHS
    for (eq, n) in sv.equations.iter().zip(&defined) {
        if let Some(n) = n.as_ref().filter(|n| !states.contains(*n)) {
            reads.entry(n.as_str()).or_default().push(&eq.rhs);
        }
    }
    let mut live: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = entries
        .iter()
        .flat_map(|en| free_variables(&en.band.coef))
        .collect();
    while let Some(n) = stack.pop() {
        let Some(rs) = reads.get(n.as_str()) else {
            continue;
        };
        if !live.insert(n) {
            continue;
        }
        for r in rs {
            stack.extend(free_variables(r));
        }
    }
    let Some(Value::Array(ser)) = model.remove("equations") else {
        unreachable!()
    };
    let Some(Value::Object(vars)) = model.get_mut("variables") else {
        unreachable!()
    };
    let mut done: HashSet<String> = HashSet::new();
    let mut eqs = Vec::new();
    for ((eq, ser), n) in sv.equations.iter().zip(ser).zip(&defined) {
        match lhs_state(eq) {
            None => {
                if is_derivative(eq) {
                    return Err(unsupported_derivative(eq));
                }
                if let Some(n) = n
                    .as_ref()
                    .filter(|n| reads.contains_key(n.as_str()) && !live.contains(*n))
                {
                    vars.remove(n);
                    continue;
                }
                eqs.push(ser);
            }
            Some(u) if zeroed && done.insert(u.clone()) => {
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
    model.insert("equations".into(), Value::Array(eqs));
    Ok(())
}

/// The `jcax<size>` axis index-set name for extent `sz`, `x`-suffixed past
/// any name `taken(nm)` (an existing set of another size). The one naming
/// rule for contracted band-field axes and hoisted-observed axes
/// (`eval_cse`), so an extent gets the same name in both.
pub(crate) fn jcax_name(taken: impl Fn(&str) -> bool, sz: i64) -> String {
    let mut nm = format!("jcax{sz}");
    while taken(&nm) {
        nm.push('x');
    }
    nm
}

/// A literal-size axis index set of the derived document (`jcax<size>`,
/// reused when present with that size).
fn axis_name(isets: &mut Map<String, Value>, sz: i64) -> String {
    let nm = jcax_name(
        |n| {
            isets
                .get(n)
                .is_some_and(|s| s.get("size") != Some(&json!(sz)))
        },
        sz,
    );
    if !isets.contains_key(&nm) {
        isets.insert(nm.clone(), json!({"kind": "interval", "size": sz}));
    }
    nm
}

/// The derived evaluation document and its band-state names (`J_1`, …).
/// With `cse`, repeated coefficient subtrees are first hoisted into observed
/// ([`crate::eval_cse::hoist_observed`]).
pub fn evaluation_document(
    sv: &SysView,
    entries: &[JacEntry],
    cse: bool,
    cse_min_nodes: usize,
) -> Result<(Value, Vec<String>), crate::bands::BandError> {
    evaluation_document_with(sv, entries, cse, cse_min_nodes, true)
}

/// [`evaluation_document`], with the source `D(u)` equations zeroed
/// (`zeroed`, the golden form) or dropped (the band-field program's form:
/// a `D(u) = 0` would make a `select`ed program keep writing `u`).
pub(crate) fn evaluation_document_with(
    sv: &SysView,
    entries: &[JacEntry],
    cse: bool,
    cse_min_nodes: usize,
    zeroed: bool,
) -> Result<(Value, Vec<String>), crate::bands::BandError> {
    let ctx = crate::system::ctx_of(sv)?;
    let (mut doc, mut model) = base_document(sv, "JacobianEval");
    zero_source_derivatives(&mut model, sv, &ctx, entries, zeroed)?;
    let Some(Value::Array(mut eqs)) = model.remove("equations") else {
        unreachable!()
    };
    let mut isets = match doc.remove("index_sets") {
        Some(Value::Object(m)) => m,
        _ => Map::new(),
    };
    let Some(Value::Object(vars)) = model.get_mut("variables") else {
        unreachable!()
    };
    let h = if cse {
        hoist_observed(entries, sv, cse_min_nodes)
    } else {
        None
    };
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
                sv.variables[&en.u]
                    .shape
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|x| json!(x))
                    .collect()
            };
            shape.extend(
                cr.iter()
                    .map(|(_, _, hi)| json!(axis_name(&mut isets, *hi))),
            );
            vars.insert(
                nm.clone(),
                json!({"type": "unknown", "default": 0.0, "shape": shape}),
            );
            let free: Vec<(usize, &String)> = b
                .ridx
                .iter()
                .enumerate()
                .filter_map(|(k2, r)| match r {
                    RIdx::Name(s) => Some((k2, s)),
                    RIdx::Fixed(_) => None,
                })
                .collect();
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
    doc.insert(
        "models".into(),
        json!({"JacobianEval": Value::Object(model)}),
    );
    Ok((Value::Object(doc), names))
}
