//! Causal self-reference recurrences (spec §4.5; esm-spec §4.3.1.1): an
//! observed `V` whose defining `faq` reads `V` itself at a strictly earlier
//! cell along one of its output axes. Such an observed has no inlining
//! fixpoint, so it is differentiated as an INTERMEDIATE: the band calculus
//! runs on its definition like on an equation (rows `V`), and on every
//! reader with `V` as an extra column. The chain rule through the sweep is
//! then either materialized — composed with the transfer recurrence
//! `T = (I − G)⁻¹` of the line (the block's `auxiliary`, §4.5) — or left
//! matrix-free as tangent and adjoint sweeps in the product documents (§9.7).
//! Everything here is an AST: the sweeps are ESM recurrences that
//! EarthSciAST executes.

use std::collections::{HashMap, HashSet};

use earthsci_ast::extension::types::RangeSpec;
use earthsci_ast::{Expr, ExpressionNode};

use crate::bands::{
    bands, merge_bands, normalize_band, range_of, subst_free, Band, BandError, Ctx,
};
use crate::branches::{canon_lits, simp};
use crate::clip::RIdx;
use crate::helpers::{free_vars, mul, op};
use crate::inline::{all_names, binders, inline_observed};
use crate::jacobian::JacEntry;
use crate::system::{defined_name, SysView};

type R<T> = Result<T, BandError>;

/// One recurrence: the observed `name`, defined by a `faq` over the cell
/// frame `frame` (its `output_idx`) with ranges `rows`, folding along frame
/// dimension `dim` (0-based). `shape` is the declared shape of `name`, as
/// written (index-set names).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recurrence {
    pub name: String,
    pub frame: Vec<String>,
    pub rows: Vec<(i64, i64)>,
    pub dim: usize,
    pub shape: Vec<String>,
}

impl Recurrence {
    /// `lo + hi` of the recurrence axis: the reversal `k ↦ lo + hi − k`.
    fn flip(&self) -> i64 {
        self.rows[self.dim].0 + self.rows[self.dim].1
    }

    fn lo(&self) -> i64 {
        self.rows[self.dim].0
    }
}

fn var(n: &str) -> Expr {
    Expr::Variable(n.to_string())
}

fn int(i: i64) -> Expr {
    Expr::Integer(i)
}

fn index(v: &str, idx: Vec<Expr>) -> Expr {
    if idx.is_empty() {
        return var(v);
    }
    let mut args = vec![var(v)];
    args.extend(idx);
    op("index", args)
}

fn and_all(mut cs: Vec<Expr>) -> Option<Expr> {
    match cs.len() {
        0 => None,
        1 => cs.pop(),
        _ => Some(op("and", cs)),
    }
}

/// A `faq` node with the given output indices and ranges (in that order;
/// ranges serialize sorted) and body.
fn faq(output_idx: Vec<String>, ranges: Vec<(String, i64, i64)>, expr: Expr) -> Expr {
    Expr::operator(ExpressionNode {
        op: "faq".into(),
        output_idx: Some(output_idx),
        ranges: Some(
            ranges
                .into_iter()
                .map(|(n, lo, hi)| (n, RangeSpec::Interval([lo, hi])))
                .collect(),
        ),
        expr: Some(Box::new(expr)),
        ..Default::default()
    })
}

/// A left fold of binary `+` (§9.1), `zero` for no term.
pub(crate) fn plus_fold(terms: Vec<Expr>, zero: Expr) -> Expr {
    let mut it = terms.into_iter();
    let Some(first) = it.next() else {
        return zero;
    };
    it.fold(first, |a, b| op("+", vec![a, b]))
}

// ── recognition ─────────────────────────────────────────────────────────────

/// The self-reads of `v` in `e` (each read's index arguments) and whether
/// `v` also occurs bare. A faq that binds `v` as an index hides it: there
/// the name is that index, never a read of the observed.
fn self_reads(e: &Expr, v: &str, reads: &mut Vec<Vec<Expr>>, bare: &mut bool) {
    match e {
        Expr::Variable(n) if n == v => *bare = true,
        Expr::Operator(n) => {
            if n.op == "faq" && binders(n).iter().any(|b| b == v) {
                return;
            }
            if n.op == "index" && matches!(n.args.first(), Some(Expr::Variable(t)) if t == v) {
                reads.push(n.args[1..].to_vec());
                for x in &n.args[1..] {
                    self_reads(x, v, reads, bare);
                }
                return;
            }
            n.for_each_child(&mut |c: &Expr| self_reads(c, v, reads, bare));
        }
        _ => {}
    }
}

/// Whether observed definition `def` reads `v` itself: a recurrence
/// candidate (esm-spec §4.3.1.1), whatever its well-foundedness.
fn is_candidate(def: &Expr, v: &str) -> bool {
    let (mut reads, mut bare) = (Vec::new(), false);
    self_reads(def, v, &mut reads, &mut bare);
    !reads.is_empty() || bare
}

/// Every recurrence of the view, in equation order: each observed whose
/// definition reads itself, with its frame and axis read off the
/// definition. A self-read the producer cannot place on one axis is
/// refused (EarthSciAST rejects such a document anyway; the producer never
/// guesses).
pub fn recurrences(sv: &SysView, obs: &HashMap<String, Expr>, ctx: &Ctx) -> R<Vec<Recurrence>> {
    let mut out: Vec<Recurrence> = Vec::new();
    for eq in sv.equations {
        let Some(n) = defined_name(eq) else { continue };
        let Some(def) = obs.get(&n) else { continue };
        if out.iter().any(|r| r.name == n) || !is_candidate(def, &n) {
            continue;
        }
        out.push(recognize(&n, def, sv, ctx)?);
    }
    Ok(out)
}

fn recognize(v: &str, def: &Expr, sv: &SysView, ctx: &Ctx) -> R<Recurrence> {
    let fail = |why: &str| BandError(format!("recurrence `{v}`: {why} (esm-spec §4.3.1.1)"));
    let shape: Vec<String> = sv
        .variables
        .get(v)
        .and_then(|x| x.shape.clone())
        .unwrap_or_default();
    let ext = ctx.shapes.get(v).cloned().unwrap_or_default();
    let Expr::Operator(n) = def else {
        return Err(fail("its definition is not a faq over its frame"));
    };
    let frame = match (&n.op[..], &n.output_idx) {
        ("faq", Some(o)) if !o.is_empty() && o.len() == ext.len() => o.clone(),
        _ => return Err(fail("its definition is not a faq over its frame")),
    };
    if frame.iter().any(|x| x.parse::<i64>().is_ok()) {
        return Err(fail("a singleton output index is not a frame symbol"));
    }
    let empty = HashMap::new();
    let ranges = n.ranges.as_ref().unwrap_or(&empty);
    let rows: Vec<(i64, i64)> = frame
        .iter()
        .enumerate()
        .map(|(m, f)| match ranges.get(f) {
            Some(r) => range_of(r, ctx),
            None => Ok((1, ext[m])),
        })
        .collect::<R<_>>()?;
    let (mut reads, mut bare) = (Vec::new(), false);
    self_reads(def, v, &mut reads, &mut bare);
    if bare {
        return Err(fail("a bare self-reference has no cell"));
    }
    let mut dims: Vec<usize> = Vec::new();
    for args in &reads {
        if args.len() != frame.len() {
            return Err(fail("a self-read does not index every frame dimension"));
        }
        for (m, a) in args.iter().enumerate() {
            if !matches!(a, Expr::Variable(x) if *x == frame[m]) && !dims.contains(&m) {
                dims.push(m);
            }
        }
    }
    match dims[..] {
        [d] => Ok(Recurrence {
            name: v.to_string(),
            frame,
            rows,
            dim: d,
            shape,
        }),
        [] => Err(fail("a self-read reads its own cell")),
        _ => Err(fail("self-reads move along more than one axis")),
    }
}

/// The names reachable from `seeds` through the observed definitions, read
/// backwards: every observed whose definition (transitively) reads a seed.
pub(crate) fn reader_closure(
    obs: &HashMap<String, Expr>,
    seeds: &HashSet<String>,
) -> HashSet<String> {
    let deps: HashMap<&String, HashSet<String>> =
        obs.iter().map(|(n, e)| (n, free_vars(e))).collect();
    let mut keep: HashSet<String> = HashSet::new();
    loop {
        let before = keep.len();
        for (n, fv) in &deps {
            if !keep.contains(*n) && fv.iter().any(|x| seeds.contains(x) || keep.contains(x)) {
                keep.insert((*n).clone());
            }
        }
        if keep.len() == before {
            return keep;
        }
    }
}

/// The recurrences that move with `seeds` (an axis's targets, or a pair's
/// two target sets): the active intermediates of a pass.
pub(crate) fn active(
    recs: &[Recurrence],
    obs: &HashMap<String, Expr>,
    seeds: &HashSet<String>,
) -> Vec<Recurrence> {
    let moving = reader_closure(obs, seeds);
    recs.iter()
        .filter(|r| moving.contains(&r.name))
        .cloned()
        .collect()
}

// ── the lag of a self band ──────────────────────────────────────────────────

/// `lag = k − (column index along the axis)` of a band of `V`'s definition
/// with respect to `V` itself (a self band), over its row index `k` (a name
/// or a pinned literal). The column index must be affine in `k` with
/// coefficient 1, and every other dimension the identity (esm-spec
/// §4.3.1.1), or the band is refused.
pub(crate) fn lag(rec: &Recurrence, b: &Band) -> R<Expr> {
    let fail = || {
        BandError(format!(
            "recurrence `{}`: a self-read is not `{}` minus a lag along its axis \
             (esm-spec §4.3.1.1)",
            rec.name, rec.frame[rec.dim]
        ))
    };
    for (m, (r, c)) in b.ridx.iter().zip(&b.cidx).enumerate() {
        if m == rec.dim {
            continue;
        }
        let same = match r {
            RIdx::Name(n) => matches!(c, Expr::Variable(x) if x == n),
            RIdx::Fixed(i) => matches!(c, Expr::Integer(j) if j == i),
        };
        if !same {
            return Err(BandError(format!(
                "recurrence `{}`: a self-read moves along dimension {} besides its axis \
                 (esm-spec §4.3.1.1)",
                rec.name,
                m + 1
            )));
        }
    }
    let c = &b.cidx[rec.dim];
    match &b.ridx[rec.dim] {
        RIdx::Fixed(k) => {
            if free_vars(c).iter().any(|x| rec.frame.contains(x)) {
                return Err(fail());
            }
            Ok(simp(&op("-", vec![int(*k), c.clone()])))
        }
        RIdx::Name(k) => {
            let has = |e: &Expr| free_vars(e).contains(k);
            let Expr::Operator(n) = c else {
                return Err(fail());
            };
            match (&n.op[..], &n.args[..]) {
                ("-", [Expr::Variable(x), a]) if x == k && !has(a) => Ok(a.clone()),
                ("+", args) => {
                    let (ks, rest): (Vec<&Expr>, Vec<&Expr>) = args
                        .iter()
                        .partition(|a| matches!(a, Expr::Variable(x) if x == k));
                    if ks.len() != 1 || rest.iter().any(|a| has(a)) {
                        return Err(fail());
                    }
                    let rest: Vec<Expr> = rest.into_iter().cloned().collect();
                    Ok(simp(&op("-", vec![crate::helpers::add(rest)])))
                }
                _ => Err(fail()),
            }
        }
    }
}

/// Whether `e` is an integer literal `≥ lo`.
fn lit_at_least(e: &Expr, lo: i64) -> bool {
    matches!(e, Expr::Integer(i) if *i >= lo)
}

// ── sweep terms ─────────────────────────────────────────────────────────────

/// A band of `V`'s definition placed on the sweep frame: the substitution of
/// its row names by frame expressions and the conditions that keep the
/// frame point inside its row box. `at[m]` is the frame expression of
/// dimension `m`, `plain[m]` whether it is the frame symbol itself (ranging
/// over `rec.rows[m]`, so a full-range row box needs no condition).
fn place(
    rec: &Recurrence,
    b: &Band,
    at: &[Expr],
    plain: &[bool],
) -> (HashMap<String, Expr>, Vec<Expr>) {
    let mut sigma = HashMap::new();
    let mut conds = Vec::new();
    for (m, r) in b.ridx.iter().enumerate() {
        match r {
            RIdx::Name(n) => {
                sigma.insert(n.clone(), at[m].clone());
                let (lo, hi) = b.rows[m];
                let (flo, fhi) = rec.rows[m];
                if !plain[m] || lo > flo {
                    conds.push(op(">=", vec![at[m].clone(), int(lo)]));
                }
                if !plain[m] || hi < fhi {
                    conds.push(op("<=", vec![at[m].clone(), int(hi)]));
                }
            }
            RIdx::Fixed(c) => conds.push(op("==", vec![at[m].clone(), int(*c)])),
        }
    }
    (sigma, conds)
}

/// One band's term in a sweep body: `coef · read` at the frame point,
/// under `conds` (the row box, the band's `where`, the read's own
/// conditions), summed over the band's contracted names. The conditions
/// gate the read through `ifelse`, which a recurrence body evaluates
/// branch-first (esm-spec §4.3.1.1), so a guarded self-read is never
/// reached outside its axis.
fn term(
    b: &Band,
    sigma: &HashMap<String, Expr>,
    conds: Vec<Expr>,
    read: Expr,
    zero: &Expr,
) -> Expr {
    let mut conds = conds;
    if let Some(g) = &b.guard {
        conds.push(subst_free(g, sigma));
    }
    let body = op("*", vec![subst_free(&b.coef, sigma), read]);
    let body = match and_all(conds) {
        None => body,
        Some(c) => op("ifelse", vec![c, body, zero.clone()]),
    };
    if b.contracted.is_empty() {
        return body;
    }
    faq(Vec::new(), b.contracted.clone(), body)
}

/// The forward read of a self band at frame `frame` (a frame symbol per
/// dimension): `self[frame with axis − lag, extra…]`, with the conditions
/// that the lag is positive and the read cell is on the axis.
fn self_read(
    rec: &Recurrence,
    b: &Band,
    sigma: &HashMap<String, Expr>,
    target: &str,
    extra: &[Expr],
) -> R<(Expr, Vec<Expr>)> {
    let l = subst_free(&lag(rec, b)?, sigma);
    let d = rec.dim;
    let pos = op("-", vec![var(&rec.frame[d]), l.clone()]);
    let mut conds = Vec::new();
    if !lit_at_least(&l, 1) {
        conds.push(op(">=", vec![l.clone(), int(1)]));
    }
    let row_lo = match &b.ridx[d] {
        RIdx::Name(_) => b.rows[d].0,
        RIdx::Fixed(c) => *c,
    };
    let provable = matches!(&l, Expr::Integer(i) if row_lo - i >= rec.lo());
    if !provable {
        conds.push(op(">=", vec![pos.clone(), int(rec.lo())]));
    }
    let mut idx: Vec<Expr> = rec.frame.iter().map(|f| var(f)).collect();
    idx[d] = pos;
    idx.extend(extra.iter().cloned());
    Ok((index(target, idx), conds))
}

/// How a forward sweep reads a column of one of `V`'s bands: `Some((read,
/// conds))` given the column's substituted index expressions; the self
/// column is handled by the sweep.
pub(crate) type ColRead<'a> = dyn Fn(&Band, &str, Vec<Expr>) -> R<Option<(Expr, Vec<Expr>)>> + 'a;

/// The forward (tangent) sweep of recurrence `rec`: one `faq` over its
/// frame whose body adds, per band of `V`'s definition (`own`, rows `V`),
/// `coef · col(read)`; a self band reads the sweep variable `target` itself
/// one lag back. A recurrence by esm-spec §4.3.1.1: well-founded because
/// every self-read is guarded to a strictly earlier cell on the axis.
pub(crate) fn forward_sweep(
    rec: &Recurrence,
    own: &[&Band],
    own_cols: &[&str],
    target: &str,
    col: &ColRead,
    zero: &Expr,
) -> R<Expr> {
    let at: Vec<Expr> = rec.frame.iter().map(|f| var(f)).collect();
    let plain = vec![true; at.len()];
    let mut terms = Vec::new();
    for (b, c) in own.iter().zip(own_cols) {
        let (sigma, mut conds) = place(rec, b, &at, &plain);
        let read = if *c == rec.name {
            let (r, cs) = self_read(rec, b, &sigma, target, &[])?;
            conds.extend(cs);
            r
        } else {
            let ci: Vec<Expr> = b.cidx.iter().map(|x| subst_free(x, &sigma)).collect();
            match col(b, c, ci)? {
                None => continue,
                Some((r, cs)) => {
                    conds.extend(cs);
                    r
                }
            }
        };
        terms.push(term(b, &sigma, conds, read, zero));
    }
    let ranges = rec
        .frame
        .iter()
        .zip(&rec.rows)
        .map(|(f, (lo, hi))| (f.clone(), *lo, *hi))
        .collect();
    Ok(faq(
        rec.frame.clone(),
        ranges,
        plus_fold(terms, zero.clone()),
    ))
}

/// The adjoint sweep of recurrence `rec`, written ascending on the REVERSED
/// axis (the spec admits no backward lag): `target[o, k̃]` is the
/// accumulated cotangent of `V[o, lo + hi − k̃]`,
///
/// ```text
/// V̂[m] = src[m] + Σ_{self points p, col(p) = m} coef(p) · V̂[row(p)]
/// ```
///
/// with `row(p)` on the axis `m + lag(p)`, i.e. `k̃ − lag` reversed. `src`
/// reads the cotangent arriving from the readers at the actual cell (a
/// function of the actual axis position expression).
pub(crate) fn adjoint_sweep(
    rec: &Recurrence,
    own_self: &[&Band],
    target: &str,
    src: Expr,
    zero: &Expr,
) -> R<Expr> {
    let d = rec.dim;
    let actual = op("-", vec![int(rec.flip()), var(&rec.frame[d])]);
    let mut terms = vec![src];
    for b in own_self {
        let l0 = lag(rec, b)?;
        // the other frame dimensions first: the lag may read them
        let mut at: Vec<Expr> = rec.frame.iter().map(|f| var(f)).collect();
        let mut plain = vec![true; at.len()];
        let o_sigma: HashMap<String, Expr> = b
            .ridx
            .iter()
            .enumerate()
            .filter(|(m, _)| *m != d)
            .filter_map(|(m, r)| match r {
                RIdx::Name(n) => Some((n.clone(), at[m].clone())),
                RIdx::Fixed(_) => None,
            })
            .collect();
        let l = subst_free(&l0, &o_sigma);
        at[d] = op("+", vec![actual.clone(), l.clone()]);
        plain[d] = false;
        let (sigma, mut conds) = place(rec, b, &at, &plain);
        let pos = op("-", vec![var(&rec.frame[d]), l.clone()]);
        if !lit_at_least(&l, 1) {
            conds.push(op(">=", vec![l.clone(), int(1)]));
        }
        conds.push(op(">=", vec![pos.clone(), int(rec.lo())]));
        let mut idx: Vec<Expr> = rec.frame.iter().map(|f| var(f)).collect();
        idx[d] = pos;
        terms.push(term(b, &sigma, conds, index(target, idx), zero));
    }
    let ranges = rec
        .frame
        .iter()
        .zip(&rec.rows)
        .map(|(f, (lo, hi))| (f.clone(), *lo, *hi))
        .collect();
    Ok(faq(
        rec.frame.clone(),
        ranges,
        plus_fold(terms, zero.clone()),
    ))
}

/// The actual cell of `rec` a reversed sweep `target` holds at row point
/// `rho` (one index expression per frame dimension): `target[rho with the
/// axis reversed]`.
pub(crate) fn reversed_read(rec: &Recurrence, target: &str, rho: Vec<Expr>) -> Expr {
    let mut idx = rho;
    let d = rec.dim;
    idx[d] = op("-", vec![int(rec.flip()), idx[d].clone()]);
    index(target, idx)
}

// ── the transfer recurrence and materialization ─────────────────────────────

/// Every name the source holds — variables, index sets and every name its
/// equations use — so a generated name stays clear of them.
fn source_names(sv: &SysView) -> HashSet<String> {
    let mut t: HashSet<String> = sv.variables.keys().cloned().collect();
    t.extend(sv.index_sets.keys().cloned());
    for eq in sv.equations {
        all_names(&eq.lhs, &mut t);
        all_names(&eq.rhs, &mut t);
    }
    t
}

/// The transfer variable of each recurrence: `_tr_<V>`, `_` appended until
/// clear of every name of the source and of the names before it.
pub(crate) fn transfer_names(sv: &SysView, recs: &[Recurrence]) -> HashMap<String, String> {
    let mut taken = source_names(sv);
    recs.iter()
        .map(|r| {
            let mut nm = format!("_tr_{}", r.name);
            while taken.contains(&nm) {
                nm.push('_');
            }
            taken.insert(nm.clone());
            (r.name.clone(), nm)
        })
        .collect()
}

/// The self bands of `rec`, from its definition with every other observed
/// inlined (no recurrence): the line operator `G`, independent of the axis
/// a pass differentiates for.
fn self_bands(
    rec: &Recurrence,
    obs: &HashMap<String, Expr>,
    recs: &[Recurrence],
    ctx: &Ctx,
) -> R<Vec<Band>> {
    let mut map = obs.clone();
    for r in recs {
        map.remove(&r.name);
    }
    let def = inline_observed(&obs[&rec.name], &map).map_err(|e| BandError(e.0))?;
    let shape = ctx.shapes.get(&rec.name).cloned().unwrap_or_default();
    Ok(merge_bands(bands(&def, &rec.name, ctx, &shape)?)
        .iter()
        .map(normalize_band)
        .collect())
}

/// The output index name of the transfer recurrence's extra axis: `_tj`,
/// `_j` appended until clear of the frame and of every name in `bs`.
fn transfer_j(rec: &Recurrence, bs: &[Band]) -> String {
    let mut taken: HashSet<String> = rec.frame.iter().cloned().collect();
    for b in bs {
        all_names(&b.coef, &mut taken);
        taken.extend(b.contracted.iter().map(|(n, _, _)| n.clone()));
    }
    let mut j = "_tj".to_string();
    while taken.contains(&j) {
        j.push_str("_j");
    }
    j
}

/// The transfer recurrence `T` of `rec` (spec §4.5): over the frame plus one
/// output index `j` on the axis,
///
/// ```text
/// T[o, k, j] = [k == j] + Σ_{self bands} coef · T[o, k − lag, j]
/// ```
///
/// so `T[o, ·, ·] = (I − G)⁻¹` on each line: `∂V[o, k] / ∂(source at [o, j])`.
/// Lower triangular on each line (zero for `j > k`).
pub fn transfer_definition(
    rec: &Recurrence,
    obs: &HashMap<String, Expr>,
    recs: &[Recurrence],
    ctx: &Ctx,
    tname: &str,
) -> R<Expr> {
    let bs = self_bands(rec, obs, recs, ctx)?;
    let j = transfer_j(rec, &bs);
    let d = rec.dim;
    let at: Vec<Expr> = rec.frame.iter().map(|f| var(f)).collect();
    let plain = vec![true; at.len()];
    let zero = int(0);
    let mut terms = vec![op(
        "ifelse",
        vec![op("==", vec![var(&rec.frame[d]), var(&j)]), int(1), int(0)],
    )];
    for b in &bs {
        let (sigma, mut conds) = place(rec, b, &at, &plain);
        let (r, cs) = self_read(rec, b, &sigma, tname, &[var(&j)])?;
        conds.extend(cs);
        terms.push(term(b, &sigma, conds, r, &zero));
    }
    let mut ranges: Vec<(String, i64, i64)> = rec
        .frame
        .iter()
        .zip(&rec.rows)
        .map(|(f, (lo, hi))| (f.clone(), *lo, *hi))
        .collect();
    ranges.push((j.clone(), rec.rows[d].0, rec.rows[d].1));
    let mut out = rec.frame.clone();
    out.push(j);
    Ok(canon_lits(&faq(out, ranges, plus_fold(terms, zero))))
}

/// The transfer band of `rec`: row `V` over its whole frame, column `V`'s
/// own source cell on the axis (contracted `κ ≤ k`), coefficient
/// `T[frame…, κ]`. Composed with a band of `V`'s definition it carries that
/// band's direct derivative along the line.
fn transfer_band(rec: &Recurrence, tname: &str) -> Band {
    let d = rec.dim;
    let mut kap = format!("{}_t", rec.frame[d]);
    while rec.frame.contains(&kap) {
        kap.push_str("_t");
    }
    let mut cidx: Vec<Expr> = rec.frame.iter().map(|f| var(f)).collect();
    cidx[d] = var(&kap);
    let mut tidx = cidx.clone();
    tidx[d] = var(&rec.frame[d]);
    tidx.push(var(&kap));
    Band {
        rows: rec.rows.clone(),
        ridx: rec.frame.iter().map(|f| RIdx::Name(f.clone())).collect(),
        cidx,
        coef: index(tname, tidx),
        contracted: vec![(kap.clone(), rec.rows[d].0, rec.rows[d].1)],
        guard: Some(op("<=", vec![var(&kap), var(&rec.frame[d])])),
    }
}

/// Every faq binder in `e`.
fn binder_names(e: &Expr, acc: &mut HashSet<String>) {
    if let Expr::Operator(n) = e {
        if n.op == "faq" {
            acc.extend(binders(n));
        }
        n.for_each_child(&mut |c: &Expr| binder_names(c, acc));
    }
}

/// The names a band holds: its index names and every name its expressions
/// use.
fn band_names(b: &Band) -> HashSet<String> {
    let mut t: HashSet<String> = HashSet::new();
    for r in &b.ridx {
        if let RIdx::Name(n) = r {
            t.insert(n.clone());
        }
    }
    t.extend(b.contracted.iter().map(|(n, _, _)| n.clone()));
    all_names(&b.coef, &mut t);
    for c in &b.cidx {
        all_names(c, &mut t);
    }
    if let Some(g) = &b.guard {
        all_names(g, &mut t);
    }
    t
}

/// The chain rule on two bands: `outer` (rows `R`, column `V` at `φ`)
/// followed by `inner` (rows `V`, column `x`). Each inner row name becomes
/// the outer column index it is read at; the inner contracted names are
/// renamed apart (`_c` appended) and become contracted names of the
/// composite. The inner row box and `V`'s extent (`shape_v`: a ghost read
/// of `V` contributes nothing) restrict the composite exactly: a bare outer
/// name tightens its own range, any other index expression adds `where`
/// conjuncts. `None` when the composite is empty.
pub(crate) fn compose(outer: &Band, inner: &Band, shape_v: &[i64]) -> R<Option<Band>> {
    let mut taken = band_names(outer);
    taken.extend(band_names(inner));
    let mut sigma: HashMap<String, Expr> = HashMap::new();
    let mut extra = Vec::new();
    for (n, lo, hi) in &inner.contracted {
        let outer_names = band_names(outer);
        let nm = if outer_names.contains(n) {
            let mut f = format!("{n}_c");
            while taken.contains(&f) {
                f.push_str("_c");
            }
            taken.insert(f.clone());
            sigma.insert(n.clone(), var(&f));
            f
        } else {
            n.clone()
        };
        extra.push((nm, *lo, *hi));
    }
    let mut rows = outer.rows.clone();
    let mut ctr = outer.contracted.clone();
    let mut conds: Vec<Expr> = Vec::new();
    // `lo ≤ e ≤ hi` exactly: tighten the range of a bare outer name, else
    // a conjunct per violated side
    let mut restrict = |e: &Expr, lo: i64, hi: i64, conds: &mut Vec<Expr>| {
        if let Expr::Variable(nm) = e {
            for (k, r) in outer.ridx.iter().enumerate() {
                if matches!(r, RIdx::Name(x) if x == nm) {
                    rows[k] = (rows[k].0.max(lo), rows[k].1.min(hi));
                    return;
                }
            }
            for c in ctr.iter_mut() {
                if c.0 == *nm {
                    c.1 = c.1.max(lo);
                    c.2 = c.2.min(hi);
                    return;
                }
            }
        }
        if lo == hi {
            conds.push(op("==", vec![e.clone(), int(lo)]));
        } else {
            conds.push(op("<=", vec![int(lo), e.clone()]));
            conds.push(op("<=", vec![e.clone(), int(hi)]));
        }
    };
    for (m, e) in outer.cidx.iter().enumerate() {
        let n = shape_v.get(m).copied().unwrap_or(1);
        let (lo, hi) = match &inner.ridx[m] {
            RIdx::Name(_) => inner.rows[m],
            RIdx::Fixed(c) => (*c, *c),
        };
        // V's extent and the inner box together
        let (lo, hi) = (lo.max(1), hi.min(n));
        if !in_range(e, outer, lo, hi) {
            restrict(e, lo, hi, &mut conds);
        }
        if let RIdx::Name(nm) = &inner.ridx[m] {
            sigma.insert(nm.clone(), e.clone());
        }
    }
    if rows.iter().any(|(lo, hi)| hi < lo) || ctr.iter().any(|(_, lo, hi)| hi < lo) {
        return Ok(None);
    }
    // a capture check: no faq inside the inner band may bind a name the
    // substituted index expressions read
    let mut bound = HashSet::new();
    binder_names(&inner.coef, &mut bound);
    for c in &inner.cidx {
        binder_names(c, &mut bound);
    }
    if let Some(g) = &inner.guard {
        binder_names(g, &mut bound);
    }
    let read: HashSet<String> = sigma.values().flat_map(free_vars).collect();
    if let Some(n) = bound.iter().find(|n| read.contains(*n)) {
        return Err(BandError(format!(
            "composing through a recurrence would capture `{n}` under a nested sum"
        )));
    }
    let guard = crate::reductions::conj(
        crate::reductions::conj(outer.guard.clone(), and_all(conds)),
        inner.guard.as_ref().map(|g| subst_free(g, &sigma)),
    );
    let b = Band {
        rows,
        ridx: outer.ridx.clone(),
        cidx: inner.cidx.iter().map(|c| subst_free(c, &sigma)).collect(),
        coef: simp(&mul(vec![
            outer.coef.clone(),
            subst_free(&inner.coef, &sigma),
        ])),
        contracted: [ctr, extra].concat(),
        guard,
    };
    Ok(tidy(b))
}

/// The span of index expression `e` over `b`'s box: a literal, a bare row
/// or contracted name, or one of those plus or minus a literal, or a
/// literal minus a name; `None` for anything else.
fn span(e: &Expr, b: &Band) -> Option<(i64, i64)> {
    let range = |nm: &str| -> Option<(i64, i64)> {
        for (k, r) in b.ridx.iter().enumerate() {
            if matches!(r, RIdx::Name(x) if x == nm) {
                return Some(b.rows[k]);
            }
        }
        b.contracted.iter().find(|c| c.0 == nm).map(|c| (c.1, c.2))
    };
    match e {
        Expr::Integer(i) => Some((*i, *i)),
        Expr::Variable(nm) => range(nm),
        Expr::Operator(n) if n.args.len() == 2 && (n.op == "+" || n.op == "-") => {
            match (&n.args[0], &n.args[1]) {
                (Expr::Variable(nm), Expr::Integer(c)) => {
                    let s = if n.op == "+" { *c } else { -*c };
                    range(nm).map(|(a, b)| (a + s, b + s))
                }
                (Expr::Integer(c), Expr::Variable(nm)) => {
                    let (a, b) = range(nm)?;
                    Some(if n.op == "+" {
                        (a + c, b + c)
                    } else {
                        (c - b, c - a)
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether index expression `e` provably lies in `[lo, hi]` at every point
/// of `b`'s box.
fn in_range(e: &Expr, b: &Band, lo: i64, hi: i64) -> bool {
    span(e, b).is_some_and(|(a, z)| lo <= a && z <= hi)
}

/// Tighten the range of the bare name `nm` of `b` to `[lo, hi]`; whether
/// `nm` is one of its row or contracted names.
fn tighten(b: &mut Band, nm: &str, lo: i64, hi: i64) -> bool {
    for (k, r) in b.ridx.iter().enumerate() {
        if matches!(r, RIdx::Name(x) if x == nm) {
            b.rows[k] = (b.rows[k].0.max(lo), b.rows[k].1.min(hi));
            return true;
        }
    }
    for c in b.contracted.iter_mut() {
        if c.0 == nm {
            c.1 = c.1.max(lo);
            c.2 = c.2.min(hi);
            return true;
        }
    }
    false
}

/// Decide what the box decides of a composite's `where` guard: a conjunct
/// that holds at every point is dropped, one that holds at none empties the
/// band (`None`), and a bound on a bare name tightens that name's range
/// instead. Repeated to a fixpoint, since a tightened range decides more;
/// the remaining conjuncts are kept once each, in order.
fn tidy(b: Band) -> Option<Band> {
    let mut b = b;
    loop {
        let Some(g) = b.guard.take() else { break };
        let mut keep: Vec<Expr> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut changed = false;
        for c in crate::reductions::conjuncts(&g) {
            let c = canon_lits(&c);
            if let Expr::Integer(i) = c {
                if i == 0 {
                    return None;
                }
                changed = true;
                continue;
            }
            if let Expr::Operator(n) = &c {
                if n.args.len() == 2 && matches!(&n.op[..], "<=" | "<" | ">=" | ">" | "==") {
                    // as `a ≤ z − off`
                    let (a, z, off) = match &n.op[..] {
                        "<=" | "==" => (&n.args[0], &n.args[1], 0),
                        "<" => (&n.args[0], &n.args[1], 1),
                        ">=" => (&n.args[1], &n.args[0], 0),
                        _ => (&n.args[1], &n.args[0], 1),
                    };
                    if let (Some(sa), Some(sz)) = (span(a, &b), span(z, &b)) {
                        if n.op == "==" {
                            if sa.0 == sa.1 && sz.0 == sz.1 {
                                if sa.0 != sz.0 {
                                    return None;
                                }
                                changed = true;
                                continue;
                            }
                            let pin = match (a, z) {
                                (Expr::Variable(x), Expr::Integer(v))
                                | (Expr::Integer(v), Expr::Variable(x)) => Some((x, *v)),
                                _ => None,
                            };
                            if let Some((x, v)) = pin {
                                if tighten(&mut b, x, v, v) {
                                    changed = true;
                                    continue;
                                }
                            }
                        } else {
                            if sa.1 <= sz.0 - off {
                                changed = true;
                                continue;
                            }
                            if sa.0 > sz.1 - off {
                                return None;
                            }
                            let bound = match (a, z) {
                                (Expr::Variable(x), Expr::Integer(v)) => {
                                    Some((x, i64::MIN, *v - off))
                                }
                                (Expr::Integer(v), Expr::Variable(x)) => {
                                    Some((x, *v + off, i64::MAX))
                                }
                                _ => None,
                            };
                            if let Some((x, lo, hi)) = bound {
                                if tighten(&mut b, x, lo, hi) {
                                    changed = true;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
            if seen.insert(crate::helpers::skey(&c)) {
                keep.push(c);
            } else {
                changed = true;
            }
        }
        if b.rows.iter().any(|(lo, hi)| hi < lo) || b.contracted.iter().any(|c| c.2 < c.1) {
            return None;
        }
        b.guard = and_all(keep);
        if !changed {
            break;
        }
    }
    Some(normalize_band(&b))
}

/// The extended first-order system of one pass: the active recurrences, the
/// bands of the state equations (columns: the axis's targets and the active
/// recurrences) and the bands of the recurrences' own definitions (rows: a
/// recurrence). Without an active recurrence `rows` is the ordinary band
/// list and the rest is empty.
#[derive(Clone, Debug, Default)]
pub struct ExtendedBands {
    pub recurrences: Vec<Recurrence>,
    pub rows: Vec<JacEntry>,
    pub rec_rows: Vec<JacEntry>,
}

impl ExtendedBands {
    pub fn is_recurrence(&self, v: &str) -> bool {
        self.recurrences.iter().any(|r| r.name == v)
    }

    pub fn recurrence(&self, v: &str) -> Option<&Recurrence> {
        self.recurrences.iter().find(|r| r.name == v)
    }
}

/// The materialized band list of an extended system (spec §4.5): every state
/// band whose column is a target, plus, for each state band reading a
/// recurrence `V`, its composition with `dV/dx` — the transfer band of `V`
/// composed with each band of `V`'s definition on a target (through any
/// recurrence it reads in turn). Per row equation the entries keep the
/// sorted column order; a column's composites follow its direct bands.
pub(crate) fn materialize(sv: &SysView, ctx: &Ctx, ext: &ExtendedBands) -> R<Vec<JacEntry>> {
    let names = transfer_names(sv, &ext.recurrences);
    let mut memo: HashMap<String, Vec<JacEntry>> = HashMap::new();
    let mut out: Vec<JacEntry> = Vec::new();
    let mut group: Vec<JacEntry> = Vec::new();
    let flush = |group: &mut Vec<JacEntry>, out: &mut Vec<JacEntry>| {
        group.sort_by(|a, b| a.v.cmp(&b.v));
        out.append(group);
    };
    for (k, en) in ext.rows.iter().enumerate() {
        if k > 0 && ext.rows[k - 1].u != en.u {
            flush(&mut group, &mut out);
        }
        match ext.recurrence(&en.v) {
            None => group.push(en.clone()),
            Some(rec) => {
                let shape = ctx.shapes.get(&rec.name).cloned().unwrap_or_default();
                for d in total(rec, ext, ctx, &names, &mut memo)? {
                    if let Some(b) = compose(&en.band, &d.band, &shape)? {
                        group.push(JacEntry {
                            u: en.u.clone(),
                            v: d.v.clone(),
                            band: b,
                        });
                    }
                }
            }
        }
    }
    flush(&mut group, &mut out);
    Ok(out)
}

/// `dV/dx` of recurrence `rec` as bands with rows `V` and target columns:
/// the transfer band composed with each band of `V`'s definition on a
/// target, and with each band on another recurrence `W` composed with
/// `dW/dx` first. Memoized per recurrence.
fn total(
    rec: &Recurrence,
    ext: &ExtendedBands,
    ctx: &Ctx,
    names: &HashMap<String, String>,
    memo: &mut HashMap<String, Vec<JacEntry>>,
) -> R<Vec<JacEntry>> {
    if let Some(t) = memo.get(&rec.name) {
        return Ok(t.clone());
    }
    let tb = transfer_band(rec, &names[&rec.name]);
    let shape = ctx.shapes.get(&rec.name).cloned().unwrap_or_default();
    let mut out = Vec::new();
    for en in ext.rec_rows.iter().filter(|en| en.u == rec.name) {
        if en.v == rec.name {
            continue; // the line operator itself: inside T
        }
        let direct: Vec<JacEntry> = match ext.recurrence(&en.v) {
            None => vec![en.clone()],
            Some(w) => {
                let sw = ctx.shapes.get(&w.name).cloned().unwrap_or_default();
                let mut v = Vec::new();
                for d in total(w, ext, ctx, names, memo)? {
                    if let Some(b) = compose(&en.band, &d.band, &sw)? {
                        v.push(JacEntry {
                            u: en.u.clone(),
                            v: d.v.clone(),
                            band: b,
                        });
                    }
                }
                v
            }
        };
        for d in direct {
            if let Some(b) = compose(&tb, &d.band, &shape)? {
                out.push(JacEntry {
                    u: rec.name.clone(),
                    v: d.v,
                    band: b,
                });
            }
        }
    }
    memo.insert(rec.name.clone(), out.clone());
    Ok(out)
}

/// The `auxiliary` definitions a band list reads (spec §4.5): for each
/// recurrence whose transfer variable some coefficient reads, `(name,
/// shape, definition)` — the shape is the recurrence's followed by its axis
/// once more. In recurrence order.
pub fn auxiliary(
    sv: &SysView,
    obs: &HashMap<String, Expr>,
    ctx: &Ctx,
    entries: &[JacEntry],
) -> R<Vec<(String, Vec<String>, Expr)>> {
    let recs = recurrences(sv, obs, ctx)?;
    if recs.is_empty() {
        return Ok(Vec::new());
    }
    let names = transfer_names(sv, &recs);
    let read: HashSet<String> = entries
        .iter()
        .flat_map(|en| free_vars(&en.band.coef))
        .collect();
    let mut out = Vec::new();
    for rec in &recs {
        let t = &names[&rec.name];
        if !read.contains(t) {
            continue;
        }
        let mut shape = rec.shape.clone();
        shape.push(rec.shape[rec.dim].clone());
        out.push((
            t.clone(),
            shape,
            transfer_definition(rec, obs, &recs, ctx, t)?,
        ));
    }
    Ok(out)
}
