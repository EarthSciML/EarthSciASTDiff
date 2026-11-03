//! Structural interning and identity memo keys (the reference's hashcons.jl).
//!
//! [`skey`] answers "are these two subtrees equal?" by serializing the whole
//! subtree. A pass that asks it once per node, or once per band of a lowered
//! stencil operator whose coefficients are ~10⁴–10⁵ nodes each, pays the
//! subtree size every time. [`StructIndex`] gives each structurally distinct
//! subtree an integer id instead: a node's table key is its SHELL (the node
//! with every child replaced by a placeholder carrying that child's id), so
//! keying a tree is Θ(nodes), and a node reached again by identity is O(1).
//!
//! Operator payloads are `Arc`s, so node identity is the payload's address.
//! [`NodeRef`] keys a memo on it and holds a strong reference, so a memoized
//! node can never be freed and its address reused for another node while
//! the memo is alive.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use earthsci_ast::{Expr, ExpressionNode};

use crate::helpers::skey;

/// An operator node, compared and hashed by identity (the `Arc` address).
/// Holding it keeps the node alive, so the address stays unique.
#[derive(Clone)]
pub(crate) struct NodeRef(pub(crate) Arc<ExpressionNode>);

impl PartialEq for NodeRef {
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.0, &o.0)
    }
}

impl Eq for NodeRef {}

impl Hash for NodeRef {
    fn hash<H: Hasher>(&self, h: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(h);
    }
}

/// Whether `a` and `b` are the same node (operators) or equal leaves.
pub(crate) fn same_expr(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Operator(x), Expr::Operator(y)) => Arc::ptr_eq(x, y),
        (Expr::Operator(_), _) | (_, Expr::Operator(_)) => false,
        _ => a == b,
    }
}

/// `e` with `f` applied to every child, returning `e` itself (the same
/// node) when `f` returned every child unchanged. Structurally the result is
/// what rebuilding the node would give; keeping the node keeps the sharing
/// that identity memos rely on.
pub(crate) fn map_children_shared(e: &Expr, f: &mut impl FnMut(&Expr) -> Expr) -> Expr {
    let Expr::Operator(n) = e else {
        return e.clone();
    };
    let mut changed = false;
    let m = n.map_children(&mut |c: &Expr| {
        let r = f(c);
        if !same_expr(&r, c) {
            changed = true;
        }
        r
    });
    if changed {
        Expr::operator(m)
    } else {
        e.clone()
    }
}

/// Interning table for structural equality: `sid(a) == sid(b)` iff
/// `skey(a) == skey(b)`. Ids are meaningful only within one index.
#[derive(Default)]
pub(crate) struct StructIndex {
    ids: HashMap<String, usize>,
    memo: HashMap<NodeRef, usize>,
}

impl StructIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn intern_key(&mut self, k: String) -> usize {
        let n = self.ids.len() + 1;
        *self.ids.entry(k).or_insert(n)
    }

    /// The structural id of `e`.
    pub(crate) fn sid(&mut self, e: &Expr) -> usize {
        let Expr::Operator(a) = e else {
            // A leaf's key is its own `skey`, behind a prefix no shell key
            // (which always starts with `{`) can share.
            return self.intern_key(format!("L{}", skey(e)));
        };
        let k = NodeRef(a.clone());
        if let Some(&id) = self.memo.get(&k) {
            return id;
        }
        let mut kids = Vec::new();
        a.for_each_child(&mut |c: &Expr| kids.push(self.sid(c)));
        let mut i = 0usize;
        // Every child position holds a placeholder, so a placeholder can
        // never be confused with a real variable of the same name.
        let shell = a.map_children(&mut |_c: &Expr| {
            i += 1;
            Expr::Variable(format!("__sid⟦{}⟧", kids[i - 1]))
        });
        let id = self.intern_key(skey(&Expr::Operator(Arc::new(shell))));
        self.memo.insert(k, id);
        id
    }
}

/// One object per distinct subtree (hashcons.jl `_Interner`): the band
/// calculus builds every coefficient on its own, so equal subtrees of
/// different coefficients are distinct allocations. Interning maps each
/// subtree to one representative per structural id, so the coefficients
/// share it. Structure (and serialization) is unchanged; only `Arc` identity
/// is.
#[derive(Default)]
pub(crate) struct Interner {
    ix: StructIndex,
    canon: HashMap<usize, Expr>,
    memo: HashMap<NodeRef, Expr>,
}

impl Interner {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn intern(&mut self, e: &Expr) -> Expr {
        let Expr::Operator(a) = e else {
            return e.clone();
        };
        let k = NodeRef(a.clone());
        if let Some(r) = self.memo.get(&k) {
            return r.clone();
        }
        let id = self.ix.sid(e);
        let c = match self.canon.get(&id) {
            Some(c) => c.clone(),
            None => {
                let c = map_children_shared(e, &mut |x| self.intern(x));
                self.canon.insert(id, c.clone());
                c
            }
        };
        self.memo.insert(k, c.clone());
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ex(v: serde_json::Value) -> Expr {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn sid_agrees_with_skey() {
        let u = |i: &str| json!({"op": "index", "args": ["u", i]});
        let es: Vec<Expr> = [
            json!(1),
            json!(1.0),
            json!(-0.0),
            json!(0.0),
            json!("x"),
            json!({"op": "+", "args": [1, "x"]}),
            json!({"op": "+", "args": [1.0, "x"]}),
            json!({"op": "+", "args": ["x", 1]}),
            json!({"op": "*", "args": [1, "x"]}),
            json!({"op": "faq", "args": [], "output_idx": ["i"],
                   "ranges": {"i": [1, 3]}, "expr": u("i")}),
            json!({"op": "faq", "args": [], "output_idx": ["i"],
                   "ranges": {"i": [1, 4]}, "expr": u("i")}),
            json!({"op": "faq", "args": [], "output_idx": ["i"],
                   "ranges": {"i": [1, 3]}, "expr": u("j")}),
            json!({"op": "ifelse", "args": [{"op": "<", "args": ["i", 2]}, u("i"), 0]}),
        ]
        .into_iter()
        .map(ex)
        .collect();
        // Fresh copies, so equal structures are distinct nodes.
        let copies: Vec<Expr> = es
            .iter()
            .map(|e| ex(serde_json::to_value(e).unwrap()))
            .collect();
        let mut ix = StructIndex::new();
        for a in es.iter().chain(&copies) {
            for b in es.iter().chain(&copies) {
                assert_eq!(
                    ix.sid(a) == ix.sid(b),
                    skey(a) == skey(b),
                    "{} vs {}",
                    skey(a),
                    skey(b)
                );
            }
        }
    }

    #[test]
    fn interning_shares_equal_subtrees() {
        let a = ex(json!({"op": "+", "args": [{"op": "*", "args": [2, "x"]}, 1]}));
        let b = ex(json!({"op": "-", "args": [{"op": "*", "args": [2, "x"]}]}));
        let mut t = Interner::new();
        let (a2, b2) = (t.intern(&a), t.intern(&b));
        assert_eq!(skey(&a2), skey(&a));
        assert_eq!(skey(&b2), skey(&b));
        let (Expr::Operator(na), Expr::Operator(nb)) = (&a2, &b2) else {
            panic!()
        };
        assert!(same_expr(&na.args[0], &nb.args[0]));
    }

    #[test]
    fn unchanged_children_keep_the_node() {
        let e = ex(json!({"op": "+", "args": [1, {"op": "*", "args": [2, "x"]}]}));
        let same = map_children_shared(&e, &mut |c| c.clone());
        assert!(same_expr(&same, &e));
        let other = map_children_shared(&e, &mut |c| match c {
            Expr::Integer(1) => Expr::Integer(3),
            _ => c.clone(),
        });
        assert!(!same_expr(&other, &e));
        assert_eq!(
            skey(&other),
            skey(&ex(
                json!({"op": "+", "args": [3, {"op": "*", "args": [2, "x"]}]})
            ))
        );
    }
}
