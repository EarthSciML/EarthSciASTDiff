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

    /// The id of a leaf: its own `skey`, behind a prefix no shell key (which
    /// always starts with `{`) can share.
    fn leaf_id(&mut self, e: &Expr) -> usize {
        self.intern_key(format!("L{}", skey(e)))
    }

    /// The id of operator node `a` whose children have ids `kids`: the key
    /// of its SHELL, every child position holding a placeholder that carries
    /// the child's id (so a placeholder can never be confused with a real
    /// variable of the same name). O(node fields), not O(subtree).
    fn shell_id(&mut self, a: &ExpressionNode, kids: &[usize]) -> usize {
        let mut i = 0usize;
        let shell = a.map_children(&mut |_c: &Expr| {
            i += 1;
            Expr::Variable(format!("__sid⟦{}⟧", kids[i - 1]))
        });
        self.intern_key(skey(&Expr::Operator(Arc::new(shell))))
    }

    /// The structural id of `e`.
    pub(crate) fn sid(&mut self, e: &Expr) -> usize {
        let Expr::Operator(a) = e else {
            return self.leaf_id(e);
        };
        let k = NodeRef(a.clone());
        if let Some(&id) = self.memo.get(&k) {
            return id;
        }
        let mut kids = Vec::new();
        a.for_each_child(&mut |c: &Expr| kids.push(self.sid(c)));
        let id = self.shell_id(a, &kids);
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
///
/// One bottom-up pass: a node's id is keyed on its shell over its children's
/// ids (as [`StructIndex::sid`] keys it), and its representative is built
/// from the children's representatives, so every node is visited and
/// shell-serialized once. The walk is iterative: the deep `ifelse` chains
/// of `index(makearray…)` lowering would otherwise cost a stack frame per
/// level.
#[derive(Default)]
pub(crate) struct Interner {
    ix: StructIndex,
    canon: HashMap<usize, Expr>,
    // node identity → (representative, structural id)
    memo: HashMap<NodeRef, (Expr, usize)>,
}

impl Interner {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn intern(&mut self, e: &Expr) -> Expr {
        let Expr::Operator(root) = e else {
            return e.clone();
        };
        let mut stack: Vec<(Arc<ExpressionNode>, bool)> = vec![(root.clone(), false)];
        while let Some((n, expanded)) = stack.pop() {
            let k = NodeRef(n.clone());
            if self.memo.contains_key(&k) {
                continue;
            }
            if !expanded {
                // revisit after every operator child is memoized
                stack.push((n.clone(), true));
                let memo = &self.memo;
                n.for_each_child(&mut |c: &Expr| {
                    if let Expr::Operator(ca) = c {
                        if !memo.contains_key(&NodeRef(ca.clone())) {
                            stack.push((ca.clone(), false));
                        }
                    }
                });
                continue;
            }
            let mut kids: Vec<(Expr, usize)> = Vec::new();
            n.for_each_child(&mut |c: &Expr| {
                kids.push(match c {
                    Expr::Operator(ca) => self.memo[&NodeRef(ca.clone())].clone(),
                    _ => (c.clone(), self.ix.leaf_id(c)),
                })
            });
            let ids: Vec<usize> = kids.iter().map(|(_, id)| *id).collect();
            let id = self.ix.shell_id(&n, &ids);
            let c = match self.canon.get(&id) {
                Some(c) => c.clone(),
                None => {
                    let mut i = 0usize;
                    let c = map_children_shared(&Expr::Operator(n.clone()), &mut |_x| {
                        i += 1;
                        kids[i - 1].0.clone()
                    });
                    self.canon.insert(id, c.clone());
                    c
                }
            };
            self.memo.insert(k, (c, id));
        }
        self.memo[&NodeRef(root.clone())].0.clone()
    }

    /// `b` with its coefficient, guard and column indices interned.
    pub(crate) fn intern_band(&mut self, b: &mut crate::bands::Band) {
        b.coef = self.intern(&b.coef);
        if let Some(g) = &b.guard {
            b.guard = Some(self.intern(g));
        }
        for c in &mut b.cidx {
            *c = self.intern(c);
        }
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
    fn interning_keeps_a_canonical_tree_and_reuses_it() {
        let a = ex(json!({"op": "+", "args": [{"op": "*", "args": [2, "x"]}, 1]}));
        let mut t = Interner::new();
        let a1 = t.intern(&a);
        // the first occurrence of a subtree is its own representative
        assert!(same_expr(&a1, &a));
        // a structurally equal copy maps onto it, root and all
        let copy = ex(serde_json::to_value(&a).unwrap());
        assert!(!same_expr(&copy, &a));
        assert!(same_expr(&t.intern(&copy), &a));
    }

    #[test]
    fn interning_a_deep_chain_needs_no_stack_per_level() {
        // an `ifelse` chain as deep as index(makearray…) lowering builds,
        // interned on a thread whose stack a recursive walk would overflow
        let depth = 50_000;
        // one shared read, so every level is already canonical and the
        // representative of the chain is the chain itself
        let read = ex(json!({"op": "index", "args": ["u", "i"]}));
        let mut e = Expr::Variable("x".into());
        for k in 0..depth {
            e = Expr::operator(earthsci_ast::ExpressionNode {
                op: "ifelse".into(),
                args: vec![ex(json!({"op": "<", "args": ["i", k]})), read.clone(), e],
                ..Default::default()
            });
        }
        let h = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let mut t = Interner::new();
                let r = t.intern(&e);
                let ok = same_expr(&r, &e);
                // dropping a deep tree recurses; keep it off this thread
                std::mem::forget((e, r, t));
                ok
            })
            .unwrap();
        assert!(h.join().unwrap());
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
