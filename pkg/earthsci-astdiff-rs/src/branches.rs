//! Branch-aware simplification + wire-canonical literals (the reference's
//! simplify_branches.jl). `simp` is THE coefficient simplifier.

use std::collections::HashMap;

use earthsci_ast::Expr;

use crate::hashcons::{map_children_shared, plain_node, same_expr, StructHash, StructIndex};
use crate::helpers::{is_zero, op};
use crate::simplify_port::simplify_jl;

/// `ifelse`-aware simplification, the swell control for coefficients of
/// limiter-heavy schemes: equal branches collapse and same-condition tests
/// are pruned.
///
/// Conditions and branches are compared by structure at every `ifelse` the
/// pass meets, so the comparison must not cost the subtree's size (a
/// limiter nests its tests, and each level would re-serialize everything
/// below it). As in the reference (simplify_branches.jl): a structural hash
/// ([`StructHash`]) settles the common unequal case in one compare, a match
/// is confirmed by structural id ([`StructIndex`]), the rewrites keep every
/// unchanged node ([`map_children_shared`]) so the identity memos of both
/// see each node once, and a census of the conditions already built skips
/// the same-condition descent when there is nothing to prune.
pub fn simplify_branches(e: &Expr) -> Expr {
    sbr(e, &mut SbrCtx::default())
}

#[derive(Default)]
struct SbrCtx {
    ix: StructIndex,
    hx: StructHash,
    /// condition hash → `ifelse` nodes on it the pass has already built
    seen: HashMap<u64, usize>,
}

impl SbrCtx {
    /// Structural equality: identity, then the hash filter, then the ids.
    fn same(&mut self, x: &Expr, y: &Expr) -> bool {
        same_expr(x, y)
            || (self.hx.shash(x) == self.hx.shash(y) && self.ix.sid(x) == self.ix.sid(y))
    }
}

fn sbr(e: &Expr, cx: &mut SbrCtx) -> Expr {
    if !matches!(e, Expr::Operator(_)) {
        return e.clone();
    }
    let e2 = map_children_shared(e, &mut |c| sbr(c, cx));
    let Expr::Operator(n2) = &e2 else {
        unreachable!()
    };
    if !(n2.op == "ifelse" && n2.args.len() == 3) {
        return e2;
    }
    let (c, a, b) = (&n2.args[0], &n2.args[1], &n2.args[2]);
    if matches!(c, Expr::Integer(_) | Expr::Number(_)) {
        return if is_zero(c) { b.clone() } else { a.clone() };
    }
    let hck = cx.hx.shash(c);
    // The pass is bottom-up, so every `ifelse` that survived into either
    // branch has been built and counted: a condition with no count has no
    // nested twin to collapse, and `assume` would return both branches
    // unchanged. Counting by hash keeps the census O(1); a collision only
    // takes the descent that would always have been taken.
    let (a2, b2) = if cx.seen.get(&hck).copied().unwrap_or(0) == 0 {
        (a.clone(), b.clone())
    } else {
        let ck = cx.ix.sid(c);
        (assume(a, hck, ck, true, cx), assume(b, hck, ck, false, cx))
    };
    if cx.same(&a2, &b2) {
        return a2;
    }
    *cx.seen.entry(hck).or_insert(0) += 1;
    // a node with no field but `op` and `args` is what `op` would build
    if same_expr(&a2, a) && same_expr(&b2, b) && plain_node(n2) {
        return e2;
    }
    op("ifelse", vec![c.clone(), a2, b2])
}

/// `e` under the assumption that the condition identified by (`hck`, `ck`)
/// has the truth value `val`: every `ifelse` on that same condition
/// collapses to the corresponding branch. `hck` filters, `ck` decides.
fn assume(e: &Expr, hck: u64, ck: usize, val: bool, cx: &mut SbrCtx) -> Expr {
    let Expr::Operator(n) = e else {
        return e.clone();
    };
    if n.op == "ifelse"
        && n.args.len() == 3
        && cx.hx.shash(&n.args[0]) == hck
        && cx.ix.sid(&n.args[0]) == ck
    {
        return assume(&n.args[if val { 1 } else { 2 }], hck, ck, val, cx);
    }
    let r = map_children_shared(e, &mut |x| assume(x, hck, ck, val, cx));
    // An `ifelse` whose condition held a test of this one now has a new
    // condition, which no count covers: count it, or an enclosing test of
    // that new condition would skip a descent that has something to prune.
    if let (Expr::Operator(r2), Expr::Operator(n)) = (&r, e) {
        if r2.op == "ifelse" && r2.args.len() == 3 && !same_expr(&r2.args[0], &n.args[0]) {
            let h = cx.hx.shash(&r2.args[0]);
            *cx.seen.entry(h).or_insert(0) += 1;
        }
    }
    r
}

const INT64_MAXF: f64 = 9_223_372_036_854_775_807.0;
const INT64_MINF: f64 = -9_223_372_036_854_775_808.0;

/// The wire-canonical literal (simplify_branches.jl `_canon_lit`): an
/// integral Int64-representable float is an integer, and a NON-FINITE float,
/// which JSON cannot write, is the quotient ESM spells it with (the esm:std
/// calendar templates' `0/0`): NaN is `0/0`, ±Inf is `±1/0`.
pub fn canon_lit(e: &Expr) -> Expr {
    if let Expr::Number(x) = e {
        if x.is_nan() {
            return op("/", vec![Expr::Integer(0), Expr::Integer(0)]);
        }
        if x.is_infinite() {
            let s = if *x > 0.0 { 1 } else { -1 };
            return op("/", vec![Expr::Integer(s), Expr::Integer(0)]);
        }
        if x.is_finite()
            && *x == x.trunc()
            && *x >= INT64_MINF
            && *x <= INT64_MAXF
            && (*x as i64) as f64 == *x
        {
            return Expr::Integer(*x as i64);
        }
    }
    e.clone()
}

pub fn canon_lits(e: &Expr) -> Expr {
    match e {
        Expr::Operator(_) => map_children_shared(e, &mut canon_lits),
        _ => canon_lit(e),
    }
}

/// Algebraic pass, branch pass, final algebraic pass, literal normalization.
pub fn simp(e: &Expr) -> Expr {
    canon_lits(&simplify_jl(&simplify_branches(&simplify_jl(e))))
}

/// [`simp`] memoized on structure, for the coefficients of one band
/// computation (the reference's `_SimpMemo`). A clipped body yields one
/// coefficient per region and site, and the regions of a stencil operator
/// mostly agree (ReSEACT's advection term: ~6000 coefficients, ~500 bands
/// once merged), so most are a coefficient already simplified. `simp` is a
/// function of structure alone, so a structural id decides the repeat, and
/// the repeat gets the same coefficient object back.
#[derive(Default)]
pub(crate) struct SimpMemo {
    ix: crate::hashcons::StructIndex,
    out: std::collections::HashMap<usize, Expr>,
}

impl SimpMemo {
    pub(crate) fn simp(&mut self, e: &Expr) -> Expr {
        if !matches!(e, Expr::Operator(_)) {
            return simp(e);
        }
        let id = self.ix.sid(e);
        self.out.entry(id).or_insert_with(|| simp(e)).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::skey;
    use earthsci_ast::ExpressionNode;

    fn times(x: f64) -> Expr {
        Expr::operator(ExpressionNode {
            op: "*".into(),
            args: vec![Expr::Variable("x".into()), Expr::Number(x)],
            ..Default::default()
        })
    }

    #[test]
    fn memo_tells_non_finite_literals_apart() {
        // `skey` writes Inf, -Inf and NaN alike (JSON null), so a memo keyed
        // on it handed `x*-Inf` the cached `simp(x*Inf)`. `Debug` shows the
        // three apart.
        let mut sm = SimpMemo::default();
        let mut seen = Vec::new();
        for x in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
            let e = times(x);
            let (got, want) = (format!("{:?}", sm.simp(&e)), format!("{:?}", simp(&e)));
            assert_eq!(got, want, "simp(x*{x})");
            seen.push(got);
        }
        assert_ne!(seen[0], seen[1]);
        assert_ne!(seen[0], seen[2]);
        assert_ne!(seen[1], seen[2]);
    }

    fn v(s: &str) -> Expr {
        Expr::Variable(s.into())
    }

    fn ife(c: Expr, a: Expr, b: Expr) -> Expr {
        op("ifelse", vec![c, a, b])
    }

    #[test]
    fn prunes_a_condition_that_assume_rewrote() {
        // `assume` rewrites `ifelse(q, x, y) < y` to `x < y` inside `q`'s
        // true side; the rebuilt test must be counted so the enclosing test
        // of `x < y` still prunes it (simplify_branches.jl, cse_test.jl).
        let c = op("<", vec![v("x"), v("y")]);
        let q = op(">", vec![v("z"), Expr::Integer(0)]);
        let cq = op("<", vec![ife(q.clone(), v("x"), v("y")), v("y")]);
        let e = ife(
            c.clone(),
            ife(q.clone(), ife(cq, v("a"), v("b")), v("d")),
            v("f"),
        );
        let want = ife(c, ife(q, v("a"), v("d")), v("f"));
        assert_eq!(skey(&simplify_branches(&e)), skey(&want));
    }
}
