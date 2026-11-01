//! §4 band-entry serialization, template CSE, structure classification,
//! factorization plans, and the block emission — golden-gated against the
//! reference byte-for-byte.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::{free_variables, Expr, ExpressionNode, VariableType};
use serde_json::{json, Map, Value};

use crate::bands::{Band, Ctx};
use crate::branches::canon_lits;
use crate::clip::RIdx;
use crate::eval_cse::hoist_observed_bands;
use crate::helpers::{map_expr_children, ser_expr, stable_json};
use crate::jacobian::unsupported_derivative;
use crate::jacobian::{FirstOrder, JacEntry, Wrt};
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
    if let Some(g) = &b.guard {
        d.insert("where".into(), ser_expr(g));
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

/// [`detect_structure`] given an optional column axis (an index-set name,
/// e.g. from an integrator's locality pass: a name carries no vertical
/// meaning by itself): `"column_block"` in place of `"banded"` or
/// `"general"` when every coupling stays within one line along that axis
/// ([`column_lines`]). The empty and block-diagonal classes are kept.
pub fn detect_structure_along(
    entries: &[JacEntry],
    sv: &SysView,
    ctx: &Ctx,
    column_axis: Option<&str>,
) -> Result<&'static str, crate::bands::BandError> {
    let st = detect_structure(entries, ctx);
    refine_structure(st, entries, sv, ctx, column_axis)
}

fn refine_structure(
    st: &'static str,
    entries: &[JacEntry],
    sv: &SysView,
    ctx: &Ctx,
    column_axis: Option<&str>,
) -> Result<&'static str, crate::bands::BandError> {
    let Some(axis) = column_axis else {
        return Ok(st);
    };
    if st == "empty" || st == "block_diagonal" {
        return Ok(st);
    }
    Ok(if column_lines(entries, sv, ctx, axis)?.is_some() {
        "column_block"
    } else {
        st
    })
}

/// The per-line structure of a state Jacobian whose couplings all stay
/// within one line along an index set (structure.jl `column_lines`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnLines {
    /// The column axis (an index-set name).
    pub axis: String,
    /// Its 1-based dimension in the common state shape.
    pub dim: usize,
    /// The axis extent.
    pub line_length: i64,
    /// Every row and column variable, sorted.
    pub block_vars: Vec<String>,
    /// The largest distance below the diagonal any point reaches.
    pub kl: i64,
    /// The largest distance above the diagonal any point reaches.
    pub ku: i64,
    /// `"dense"`, `"lower"`, `"upper"`, `"banded"` or `"diag_plus_rank1"`.
    pub pattern: &'static str,
    /// The rank-one terms of a `"diag_plus_rank1"` line block (empty for
    /// every other pattern).
    pub rank1: Vec<Rank1>,
}

/// One rank-one term of a `"diag_plus_rank1"` line block (spec §6): entry
/// `entry` (1-based, in block order) contributes `u(row) · v(κ)` at every
/// point, `u` over the entry's row index names and `v` over its one
/// contracted name `κ` — the column's position on the line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rank1 {
    pub entry: usize,
    pub u: Value,
    pub v: Value,
}

/// The factors of a product: numerator and denominator factors, `*` and
/// `/` flattened and a unary minus taken out as `-1`.
fn factors(e: &Expr, num: &mut Vec<Expr>, den: &mut Vec<Expr>) {
    if let Expr::Operator(n) = e {
        match (&n.op[..], n.args.len()) {
            ("*", _) => {
                for a in &n.args {
                    factors(a, num, den);
                }
                return;
            }
            ("/", 2) => {
                factors(&n.args[0], num, den);
                factors(&n.args[1], den, num);
                return;
            }
            ("-", 1) => {
                num.push(Expr::Integer(-1));
                factors(&n.args[0], num, den);
                return;
            }
            _ => {}
        }
    }
    num.push(e.clone());
}

/// `coef = u · v` with `u` free of the contracted name `kap` and `v` free of
/// every row name, or `None` when a factor reads both.
fn split_rank1(coef: &Expr, kap: &str, rows: &HashSet<String>) -> Option<(Expr, Expr)> {
    let (mut num, mut den) = (Vec::new(), Vec::new());
    factors(coef, &mut num, &mut den);
    let (mut un, mut ud, mut vn, mut vd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (fs, is_num) in [(num, true), (den, false)] {
        for f in fs {
            let fv = crate::helpers::free_vars(&f);
            let (k, r) = (fv.contains(kap), fv.iter().any(|x| rows.contains(x)));
            match (k, r, is_num) {
                (true, true, _) => return None,
                (true, false, true) => vn.push(f),
                (true, false, false) => vd.push(f),
                (false, _, true) => un.push(f),
                (false, _, false) => ud.push(f),
            }
        }
    }
    let build = |n: Vec<Expr>, d: Vec<Expr>| -> Expr {
        let n = crate::helpers::mul(n);
        if d.is_empty() {
            n
        } else {
            crate::helpers::op("/", vec![n, crate::helpers::mul(d)])
        }
    };
    Some((build(un, ud), build(vn, vd)))
}

/// A band whose every point reads one other cell of its row's line — the
/// line position being its one contracted name, under no `where` — with a
/// coefficient that splits as `u(row) · v(κ)`: a rank-one term of the line
/// block.
fn rank1_term(b: &Band, dim: usize) -> Option<(Expr, Expr)> {
    if b.guard.is_some() || b.contracted.len() != 1 {
        return None;
    }
    let kap = &b.contracted[0].0;
    if !matches!(b.cidx.get(dim), Some(Expr::Variable(x)) if x == kap) {
        return None;
    }
    let rows: HashSet<String> = b
        .ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(n) => Some(n.clone()),
            RIdx::Fixed(_) => None,
        })
        .collect();
    split_rank1(&b.coef, kap, &rows)
}

/// The per-line structure of a state Jacobian along index set `axis`, or
/// `None` when a coupling leaves its line: every band couples two array
/// states of one common shape that holds `axis` once, and at every band
/// point the column equals the row in every other dimension. A point
/// reading past the axis is a ghost read and does not count. `kl` / `ku`
/// and their `pattern` name are read off the exact band points (guards
/// applied), never probed.
pub fn column_lines(
    entries: &[JacEntry],
    sv: &SysView,
    ctx: &Ctx,
    axis: &str,
) -> Result<Option<ColumnLines>, crate::bands::BandError> {
    if entries.is_empty() {
        return Ok(None);
    }
    let mut shape: Option<&Vec<String>> = None;
    for en in entries {
        for n in [&en.u, &en.v] {
            let Some(sh) = sv
                .variables
                .get(n)
                .and_then(|v| v.shape.as_ref())
                .filter(|s| !s.is_empty())
            else {
                return Ok(None);
            };
            match shape {
                None => shape = Some(sh),
                Some(s0) if s0 != sh => return Ok(None),
                _ => {}
            }
        }
    }
    let shape = shape.expect("entries are not empty");
    let dims: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|(_, x)| *x == axis)
        .map(|(k, _)| k)
        .collect();
    if dims.len() != 1 {
        return Ok(None);
    }
    let dim = dims[0];
    let nn = *ctx
        .index_sets
        .get(axis)
        .ok_or_else(|| crate::bands::BandError(format!("unknown index set `{axis}`")))?;
    let (mut kl, mut ku) = (0i64, 0i64);
    // per entry: whether every point sits at distance 0 on the axis
    let mut diag = vec![true; entries.len()];
    for (k, en) in entries.iter().enumerate() {
        let ok = crate::points::all_points(&en.band, &mut |cell, _, _, cidx| {
            for d in 0..cidx.len() {
                if d != dim && cidx[d] != cell[d] {
                    return Ok(false);
                }
            }
            if cidx[dim] < 1 || cidx[dim] > nn {
                return Ok(true); // ghost read
            }
            let off = cidx[dim] - cell[dim];
            kl = kl.max(-off);
            ku = ku.max(off);
            diag[k] = diag[k] && off == 0;
            Ok(true)
        })?;
        if !ok {
            return Ok(None);
        }
    }
    // every entry at distance 0 or a rank-one term, and at least one of
    // those: D + Σ u vᵀ per line, for a Sherman–Morrison solve
    let mut rank1 = Vec::new();
    let mut lowrank = (kl, ku) != (0, 0);
    for (k, en) in entries.iter().enumerate() {
        if diag[k] {
            continue;
        }
        match rank1_term(&en.band, dim) {
            Some((u, v)) => rank1.push(Rank1 {
                entry: k + 1,
                u: ser_expr(&canon_lits(&u)),
                v: ser_expr(&canon_lits(&v)),
            }),
            None => lowrank = false,
        }
    }
    if !lowrank {
        rank1.clear();
    }
    let pattern = if !rank1.is_empty() {
        "diag_plus_rank1"
    } else if kl == nn - 1 && ku == nn - 1 {
        "dense"
    } else if ku == 0 && kl == nn - 1 {
        "lower"
    } else if kl == 0 && ku == nn - 1 {
        "upper"
    } else {
        "banded"
    };
    let mut vars: Vec<String> = entries
        .iter()
        .flat_map(|en| [en.u.clone(), en.v.clone()])
        .collect();
    vars.sort();
    vars.dedup();
    Ok(Some(ColumnLines {
        axis: axis.to_string(),
        dim: dim + 1,
        line_length: nn,
        block_vars: vars,
        kl,
        ku,
        pattern,
        rank1,
    }))
}

/// The §6 line-by-line factorization plan of a `"column_block"` Jacobian
/// (factorization.jl `ColumnBlockPlan`, serialized).
pub fn column_block_plan(cl: &ColumnLines) -> Value {
    serialize_plan(&FactorizationPlan::ColumnBlock(cl.clone()))
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// The serialized per-cell block LU of a block-diagonal band list: the
/// `factorization` field of a state block ([`serialize_plan`] of
/// [`block_diagonal_lu`]).
pub fn block_diagonal_plan(entries: &[JacEntry]) -> Value {
    serialize_plan(&FactorizationPlan::BlockDiagonal(block_diagonal_lu(
        entries,
    )))
}

/// The structure classes [`detect_structure`] reports (spec §4).
pub const STRUCTURE_CLASSES: [&str; 5] = [
    "empty",
    "block_diagonal",
    "banded",
    "general",
    "column_block",
];

/// Symbolic factorization info for one structure class (factorization.jl's
/// `FactorizationPlan` subtypes). Non-exhaustive: a plan for another class
/// is a new variant.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FactorizationPlan {
    /// The per-cell block LU of a `"block_diagonal"` Jacobian.
    BlockDiagonal(BlockLu),
    /// The line-by-line plan of a `"column_block"` Jacobian.
    ColumnBlock(ColumnLines),
}

/// Symbolic factorization info for the band list's structure class: the
/// reference's `plan_factorization(entries, sv; structure)`.
///
/// `structure` defaults (`None`) to [`detect_structure`] of `entries` over
/// `ctx` (the view's context, [`crate::ctx_of`]). A `"block_diagonal"`
/// class gives its per-cell block LU ([`FactorizationPlan::BlockDiagonal`]);
/// `"column_block"` needs its column axis ([`plan_factorization_along`]);
/// every other class gives `None`, since no plan is implemented for it yet
/// (the Jacobian is still fully usable: a plan is an optimization, never a
/// requirement).
///
/// Errors: `structure` is not one of [`STRUCTURE_CLASSES`], or `entries`
/// has a parameter column (a parameters or forcing band list is a
/// rectangular matrix, which no plan factorizes).
pub fn plan_factorization(
    entries: &[JacEntry],
    sv: &SysView,
    ctx: &Ctx,
    structure: Option<&str>,
) -> Result<Option<FactorizationPlan>, crate::bands::BandError> {
    plan_factorization_along(entries, sv, ctx, structure, None)
}

/// [`plan_factorization`] given an optional column axis (the one
/// [`detect_structure_along`] was given): `structure` then defaults to that
/// refined class, and a `"column_block"` class gives its line-by-line plan
/// ([`FactorizationPlan::ColumnBlock`]).
///
/// Errors, besides [`plan_factorization`]'s: a `"column_block"` class with
/// no `column_axis`, or one the band list is not column-block along.
pub fn plan_factorization_along(
    entries: &[JacEntry],
    sv: &SysView,
    ctx: &Ctx,
    structure: Option<&str>,
    column_axis: Option<&str>,
) -> Result<Option<FactorizationPlan>, crate::bands::BandError> {
    let structure = match structure {
        Some(s) if STRUCTURE_CLASSES.contains(&s) => s,
        Some(s) => {
            return Err(crate::bands::BandError(format!(
                "unknown structure class `{s}` (expected one of {STRUCTURE_CLASSES:?})"
            )))
        }
        None => detect_structure_along(entries, sv, ctx, column_axis)?,
    };
    if let Some(en) = entries.iter().find(|en| {
        sv.variables
            .get(&en.v)
            .is_some_and(|v| v.var_type == VariableType::Parameter)
    }) {
        return Err(crate::bands::BandError(format!(
            "no factorization plan for a rectangular band list: column `{}` is a \
             parameter (plan a states or time band list)",
            en.v
        )));
    }
    match structure {
        "block_diagonal" => Ok(Some(FactorizationPlan::BlockDiagonal(block_diagonal_lu(
            entries,
        )))),
        "column_block" => {
            let axis = column_axis.ok_or_else(|| {
                crate::bands::BandError("a `column_block` plan needs its column axis".into())
            })?;
            let cl = column_lines(entries, sv, ctx, axis)?.ok_or_else(|| {
                crate::bands::BandError(format!(
                    "the Jacobian is not `column_block` along `{axis}`"
                ))
            })?;
            Ok(Some(FactorizationPlan::ColumnBlock(cl)))
        }
        _ => Ok(None),
    }
}

/// A factorization plan as the `factorization` field of a `jacobians`
/// block (spec §4): the reference's `serialize_plan`. A block LU is an
/// object with `type` (`"block_diagonal_lu"`), `block_vars` (the block's
/// variables in elimination order), `ordering` (the 1-based Markowitz
/// ordering of the name-sorted variables), and `pattern` and `lu_pattern`
/// (the permuted block's pattern and its LU pattern including fill-in, as
/// sorted 1-based `[row, column]` pairs). Serialized with
/// [`crate::stable_json`] it is byte-identical to the Julia and Python
/// bindings' output.
pub fn serialize_plan(plan: &FactorizationPlan) -> Value {
    match plan {
        FactorizationPlan::BlockDiagonal(b) => json!({
            "type": "block_diagonal_lu",
            "block_vars": b.block_vars,
            "ordering": b.ordering,
            "pattern": nzpairs(&b.pattern),
            "lu_pattern": nzpairs(&b.lu),
        }),
        FactorizationPlan::ColumnBlock(cl) => {
            let mut p = json!({
                "type": "column_block",
                "axis": cl.axis,
                "line_length": cl.line_length,
                "block_vars": cl.block_vars,
                "pattern": cl.pattern,
                "bandwidth": [cl.kl, cl.ku],
            });
            if !cl.rank1.is_empty() {
                p["rank1"] = Value::Array(
                    cl.rank1
                        .iter()
                        .map(|r| json!({"entry": r.entry, "u": r.u, "v": r.v}))
                        .collect(),
                );
            }
            p
        }
    }
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

/// The options of a block emission ([`jacobian_block_with`],
/// [`jacobian_blocks_with`]).
#[derive(Clone, Debug)]
pub struct BlockOptions<'a> {
    /// The forcing set (`None` infers it).
    pub forcing: Option<&'a [String]>,
    /// Template-CSE the coefficients.
    pub cse: bool,
    /// The smallest subtree (in nodes) hoisted into a template.
    pub cse_min_nodes: usize,
    /// An index-set name along which the state block may report the
    /// `"column_block"` structure class with its line-by-line factorization
    /// plan, when every coupling stays within one line along it
    /// ([`column_lines`]). Ignored on every other axis.
    pub column_axis: Option<&'a str>,
}

impl Default for BlockOptions<'_> {
    fn default() -> Self {
        BlockOptions {
            forcing: None,
            cse: true,
            cse_min_nodes: 12,
            column_axis: None,
        }
    }
}

/// [`jacobian_block`] with every emission option ([`BlockOptions`]).
pub fn jacobian_block_with(
    file: &earthsci_ast::EsmFile,
    model_name: Option<&str>,
    wrt: Wrt,
    opts: &BlockOptions,
) -> Result<Value, crate::bands::BandError> {
    let sv: SysView = crate::system::sysview(file, model_name)?;
    let ctx = crate::system::ctx_of(&sv)?;
    let obs = crate::system::observed_defs(&sv)?;
    let mut memo = FirstOrder::default();
    axis_block(&sv, &ctx, &obs, wrt, opts, &mut memo)
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
    let opts = BlockOptions {
        forcing,
        cse,
        cse_min_nodes,
        column_axis: None,
    };
    jacobian_blocks_with(file, model_name, wrt, &opts)
}

/// [`jacobian_blocks`] with every emission option ([`BlockOptions`]); a
/// `column_axis` refines the state block only (a pair block has no
/// structure class to refine).
pub fn jacobian_blocks_with<W: Into<BlockWrt> + Copy>(
    file: &earthsci_ast::EsmFile,
    model_name: Option<&str>,
    wrt: &[W],
    opts: &BlockOptions,
) -> Result<Value, crate::bands::BandError> {
    let (forcing, cse, cse_min_nodes) = (opts.forcing, opts.cse, opts.cse_min_nodes);
    let wrt: Vec<BlockWrt> = wrt.iter().map(|w| (*w).into()).collect();
    let distinct: HashSet<BlockWrt> = wrt.iter().copied().collect();
    if wrt.is_empty() || distinct.len() != wrt.len() {
        return Err(crate::bands::BandError(format!(
            "wrt must name one or more distinct axes, got {:?}",
            wrt.iter().map(BlockWrt::label).collect::<Vec<_>>()
        )));
    }
    let sv = crate::system::sysview(file, model_name)?;
    let ctx = crate::system::ctx_of(&sv)?;
    let obs = crate::system::observed_defs(&sv)?;
    let mut memo = FirstOrder::default();
    Ok(Value::Array(
        wrt.iter()
            .map(|w| match *w {
                BlockWrt::Axis(w) => axis_block(&sv, &ctx, &obs, w, opts, &mut memo),
                BlockWrt::Pair(a1, a2) => crate::second_order::pair_block(
                    &sv,
                    &ctx,
                    &obs,
                    (a1, a2),
                    forcing,
                    cse,
                    cse_min_nodes,
                    &mut memo,
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
    let opts = BlockOptions {
        forcing,
        cse,
        cse_min_nodes,
        column_axis: None,
    };
    jacobian_block_with(file, model_name, wrt, &opts)
}

/// [`jacobian_block_with_forcing`] over a view, sharing its first-order
/// pass through `memo` with the other blocks of one emission.
fn axis_block(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: Wrt,
    opts: &BlockOptions,
    memo: &mut FirstOrder,
) -> Result<Value, crate::bands::BandError> {
    let (forcing, cse, cse_min_nodes) = (opts.forcing, opts.cse, opts.cse_min_nodes);
    let entries = memo.bands(sv, ctx, obs, wrt, None, forcing, &[])?;
    // A parameter or forcing block describes a rectangular matrix, which the
    // structure classes and the factorization plan do not: it carries its
    // column space instead, so a consumer can size and order the columns
    // without the source document.
    let (columns, structure) = if wrt.has_columns() {
        (Some(columns_field(sv, wrt, forcing)?), None)
    } else {
        (None, Some(memo.structure(wrt, None, &entries, ctx)))
    };
    // Line-by-line structure is a property of the square state matrix.
    let cax = if wrt == Wrt::States {
        opts.column_axis
    } else {
        None
    };
    // The one point walk both the class and its plan read.
    let lines = match (structure, cax) {
        (Some(st), Some(axis)) if st != "empty" && st != "block_diagonal" => {
            column_lines(&entries, sv, ctx, axis)?
        }
        _ => None,
    };
    let structure = if lines.is_some() {
        Some("column_block")
    } else {
        structure
    };
    let mut block = block_fields(
        json!(wrt.as_str()),
        &entries,
        serialize_band,
        columns,
        structure,
        cse,
        cse_min_nodes,
    );
    let aux = crate::recurrence::auxiliary(sv, obs, ctx, &entries)?;
    if !aux.is_empty() {
        block.insert("auxiliary".into(), auxiliary_field(&aux));
    }
    if let Some(cl) = lines {
        // The plan from the point walk above, not a second one.
        let plan = FactorizationPlan::ColumnBlock(cl);
        block.insert("factorization".into(), serialize_plan(&plan));
    } else if let Some(st) = structure {
        if let Some(plan) = plan_factorization(&entries, sv, ctx, Some(st))? {
            block.insert("factorization".into(), serialize_plan(&plan));
        }
    }
    Ok(Value::Object(block))
}

/// The fields every block shares, in their emitted order: `wrt`, the
/// entries (coefficients template-CSE'd when `cse` is set), the `columns`
/// of a parameter or forcing axis and the `structure` class (each when
/// given), and the block-local `expression_templates`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_fields<T: Clone + BandEntry>(
    wrt: Value,
    entries: &[T],
    serialize: impl Fn(&T) -> Value,
    columns: Option<Value>,
    structure: Option<&str>,
    cse: bool,
    cse_min_nodes: usize,
) -> Map<String, Value> {
    let mut ser_entries: Vec<Value> = entries.iter().map(&serialize).collect();
    let mut templates: Vec<(String, Expr)> = Vec::new();
    if cse {
        let coefs: Vec<Expr> = entries.iter().map(|en| en.band().coef.clone()).collect();
        let (tpl, rewritten) = cse_templates(&coefs, cse_min_nodes, "jt");
        if !tpl.is_empty() {
            ser_entries = entries
                .iter()
                .zip(rewritten)
                .map(|(en, coef)| {
                    let mut en2 = en.clone();
                    en2.band_mut().coef = coef;
                    serialize(&en2)
                })
                .collect();
            templates = tpl;
        }
    }
    let mut block = Map::new();
    block.insert("wrt".into(), wrt);
    block.insert("entries".into(), Value::Array(ser_entries));
    if let Some(c) = columns {
        block.insert("columns".into(), c);
    }
    if let Some(st) = structure {
        block.insert("structure".into(), json!(st));
    }
    if !templates.is_empty() {
        let mut t = Map::new();
        for (n, body) in &templates {
            t.insert(n.clone(), json!({"params": [], "body": ser_expr(body)}));
        }
        block.insert("expression_templates".into(), Value::Object(t));
    }
    block
}

/// A first- or second-order entry, for [`block_fields`].
pub(crate) trait BandEntry {
    fn band(&self) -> &Band;
    fn band_mut(&mut self) -> &mut Band;
}

impl BandEntry for JacEntry {
    fn band(&self) -> &Band {
        &self.band
    }
    fn band_mut(&mut self) -> &mut Band {
        &mut self.band
    }
}

impl BandEntry for JacEntry2 {
    fn band(&self) -> &Band {
        &self.band
    }
    fn band_mut(&mut self) -> &mut Band {
        &mut self.band
    }
}

/// A parameter or forcing block's `columns` field.
pub(crate) fn columns_field(
    sv: &SysView,
    wrt: Wrt,
    forcing: Option<&[String]>,
) -> Result<Value, crate::bands::BandError> {
    let cols = crate::jacobian::jacobian_columns(sv, wrt, forcing)?;
    Ok(Value::Array(
        cols.iter()
            .map(|(n, shape)| json!({"name": n, "shape": shape}))
            .collect(),
    ))
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
            guard: d.get("where").map(parse_expr).transpose()?,
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
pub(crate) fn base_document(sv: &SysView, name: &str) -> (Map<String, Value>, Map<String, Value>) {
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
    // Provisional: `derived_version` settles it once the document is complete.
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
/// in place of the first and with any further ones dropped. The states keep
/// their slots, so the observed and band fields read them unchanged, but
/// evaluating the document no longer recomputes `f`. Observed the
/// coefficients do not read (directly or through other observed) are dropped
/// with their declarations: they fed only `f`. (Carrying the states as
/// parameters instead is unsafe: EarthSciAST caches observed that read only
/// parameters, so an in-place `u` update goes stale.) `roots` are the
/// expressions the derived model evaluates: the coefficients, and for a
/// product document also the column indices and `where` guards (spec §9).
pub(crate) fn zero_source_derivatives(
    model: &mut Map<String, Value>,
    sv: &SysView,
    ctx: &Ctx,
    roots: &[&Expr],
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
    let mut stack: Vec<String> = roots.iter().flat_map(|r| free_variables(r)).collect();
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
            Some(u) if done.insert(u.clone()) => {
                let shape = ctx.shapes.get(&u).cloned().unwrap_or_default();
                let rhs = zeros(&shape);
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

/// With `cse`, repeated coefficient subtrees are hoisted into observed of
/// the derived model ([`crate::hoist_observed`]): its index sets, declarations and
/// bare-LHS equations go into `doc` and `model`. The coefficients the model
/// reads (emit.jl `_hoist!`).
pub(crate) fn hoist(
    doc: &mut Map<String, Value>,
    model: &mut Map<String, Value>,
    sv: &SysView,
    bands: &[&Band],
    cse: bool,
    cse_min_nodes: usize,
) -> Vec<Expr> {
    let Some(h) = cse
        .then(|| hoist_observed_bands(bands, sv, cse_min_nodes))
        .flatten()
    else {
        return bands.iter().map(|b| b.coef.clone()).collect();
    };
    if let Some(Value::Object(isets)) = doc.get_mut("index_sets") {
        for (nm, sz) in &h.axes {
            isets.insert(nm.clone(), json!({"kind": "interval", "size": sz}));
        }
    }
    if let Some(Value::Object(vars)) = model.get_mut("variables") {
        for (nm, vd) in &h.obs {
            vars.insert(nm.clone(), vd.clone());
        }
    }
    if let Some(Value::Array(eqs)) = model.get_mut("equations") {
        for (nm, rhs) in &h.obs_eqs {
            eqs.push(json!({"lhs": nm, "rhs": rhs}));
        }
    }
    h.coefs
}

/// The zero value of a variable of `shape` (§9.1): `0.0`, or a zero
/// `makearray` over the shape.
pub(crate) fn zeros(shape: &[i64]) -> Value {
    if shape.is_empty() {
        json!(0.0)
    } else {
        json!({"op": "makearray", "args": [],
               "regions": [shape.iter().map(|n| json!([1, n])).collect::<Vec<_>>()],
               "values": [0.0]})
    }
}

/// The `auxiliary` definitions (spec §4.5) as observed of a derived model:
/// each a declared unknown of its shape defined by a bare-LHS equation,
/// after the source's own equations.
pub(crate) fn add_auxiliary(model: &mut Map<String, Value>, aux: &[(String, Vec<String>, Expr)]) {
    if let Some(Value::Object(vars)) = model.get_mut("variables") {
        for (n, shape, _) in aux {
            vars.insert(n.clone(), json!({"type": "unknown", "shape": shape}));
        }
    }
    if let Some(Value::Array(eqs)) = model.get_mut("equations") {
        for (n, _, d) in aux {
            eqs.push(json!({"lhs": n, "rhs": ser_expr(d)}));
        }
    }
}

/// The block's `auxiliary` field (spec §4.5): `[{name, shape, rhs}]` in
/// recurrence order.
pub(crate) fn auxiliary_field(aux: &[(String, Vec<String>, Expr)]) -> Value {
    Value::Array(
        aux.iter()
            .map(|(n, shape, d)| json!({"name": n, "shape": shape, "rhs": ser_expr(d)}))
            .collect(),
    )
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
    let ctx = crate::system::ctx_of(sv)?;
    let (mut doc, mut model) = base_document(sv, "JacobianEval");
    let obs = crate::system::observed_defs(sv)?;
    let aux = crate::recurrence::auxiliary(sv, &obs, &ctx, entries)?;
    let mut roots: Vec<&Expr> = entries.iter().map(|en| &en.band.coef).collect();
    roots.extend(aux.iter().map(|(_, _, d)| d));
    zero_source_derivatives(&mut model, sv, &ctx, &roots)?;
    add_auxiliary(&mut model, &aux);
    let bands: Vec<&Band> = entries.iter().map(|en| &en.band).collect();
    let coefs = hoist(&mut doc, &mut model, sv, &bands, cse, cse_min_nodes);
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
    let mut models = Map::new();
    models.insert("JacobianEval".into(), Value::Object(model));
    component_tree(&mut models, "JacobianEval")?;
    doc.insert("models".into(), Value::Object(models));
    derived_version(&mut doc);
    Ok((Value::Object(doc), names))
}

// ── the component tree (spec §8.1) ──────────────────────────────────────────
//
// A flattened system's variables carry scoped names (`SimpleOzone.O3`).
// Declared in the one derived model, EarthSciAST reads such a name as a
// reference into a component `SimpleOzone` that does not exist, and refuses
// the document. So each scoped name `P.v` is declared as `v` in the model at
// component path `P` (the first segment a top-level model, each further one a
// `subsystems` entry), and each equation goes to the model of its subject.
// Every reference stays as written, fully scoped, which EarthSciAST resolves
// from any model, so loading and flattening the document gives back the
// source's flattened names. The band fields, hoisted observed and recurrence
// auxiliaries keep their unscoped names in the derived model, so a source
// with no scoped name (one model of a document) gives the document it always
// did.

/// The variable an equation is about, from its serialized left-hand side:
/// the `D` or `ic` argument, a bare name, or the variable an `index` (inside
/// a lifted `faq`) writes.
fn lhs_subject(lhs: &Value) -> Option<&str> {
    match lhs {
        Value::String(s) => Some(s),
        Value::Object(m) => match m.get("op").and_then(Value::as_str) {
            Some("faq") => m.get("expr").and_then(lhs_subject),
            Some("D" | "ic" | "index") => m
                .get("args")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(lhs_subject),
            _ => None,
        },
        _ => None,
    }
}

/// The model at component `path` of `models`, created (with its parents)
/// when missing.
fn component_node<'m>(
    models: &'m mut Map<String, Value>,
    path: &str,
    derived: &str,
) -> Result<&'m mut Map<String, Value>, crate::bands::BandError> {
    fn node() -> Value {
        json!({"variables": {}, "equations": []})
    }
    let mut segs = path.split('.');
    let top = segs.next().unwrap_or_default();
    if top == derived {
        return Err(crate::bands::BandError(format!(
            "the source has a component named `{derived}`, which is the derived model's name"
        )));
    }
    let mut cur = models
        .entry(top)
        .or_insert_with(node)
        .as_object_mut()
        .expect("a model is an object");
    for s in segs {
        cur = cur
            .entry("subsystems")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("subsystems is an object")
            .entry(s)
            .or_insert_with(node)
            .as_object_mut()
            .expect("a subsystem is an object");
    }
    Ok(cur)
}

/// Move the scoped declarations and the equations whose subject is scoped
/// out of the derived model `derived` into the component tree, in order.
pub(crate) fn component_tree(
    models: &mut Map<String, Value>,
    derived: &str,
) -> Result<(), crate::bands::BandError> {
    let Some(Value::Object(mut m)) = models.remove(derived) else {
        unreachable!()
    };
    let Some(Value::Object(vars)) = m.remove("variables") else {
        unreachable!()
    };
    let Some(Value::Array(eqs)) = m.remove("equations") else {
        unreachable!()
    };
    let mut keep_vars = Map::new();
    let mut scoped: Vec<(String, Value)> = Vec::new();
    for (n, v) in vars {
        if n.contains('.') {
            scoped.push((n, v));
        } else {
            keep_vars.insert(n, v);
        }
    }
    scoped.sort_by(|a, b| a.0.cmp(&b.0));
    for (n, v) in scoped {
        let (path, local) = n.rsplit_once('.').expect("a scoped name has a dot");
        let node = component_node(models, path, derived)?;
        if let Some(Value::Object(vs)) = node.get_mut("variables") {
            vs.insert(local.to_string(), v);
        }
    }
    let mut keep_eqs = Vec::new();
    for eq in eqs {
        let path = lhs_subject(&eq["lhs"])
            .and_then(|s| s.rsplit_once('.'))
            .map(|(p, _)| p.to_string());
        match path {
            Some(p) => {
                let node = component_node(models, &p, derived)?;
                if let Some(Value::Array(es)) = node.get_mut("equations") {
                    es.push(eq);
                }
            }
            None => keep_eqs.push(eq),
        }
    }
    m.insert("variables".into(), Value::Object(keep_vars));
    m.insert("equations".into(), Value::Array(keep_eqs));
    models.insert(derived.to_string(), Value::Object(m));
    Ok(())
}

/// The `esm` version a finished derived document declares: the lowest that
/// admits what it holds. 1.1.0 for its `faq` nodes; 1.2.0 when any
/// expression node carries `units` (a unit-bearing `const` copied from a
/// 1.2.0 source). Read off the content, as the Julia reference does.
pub(crate) fn derived_version(doc: &mut Map<String, Value>) {
    fn has_units(v: &Value) -> bool {
        match v {
            Value::Object(m) => {
                (m.contains_key("units") && m.contains_key("op")) || m.values().any(has_units)
            }
            Value::Array(a) => a.iter().any(has_units),
            _ => false,
        }
    }
    let v = if doc.get("models").is_some_and(has_units) {
        "1.2.0"
    } else {
        "1.1.0"
    };
    doc.insert("esm".into(), json!(v));
}
