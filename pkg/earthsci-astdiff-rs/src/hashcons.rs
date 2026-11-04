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
//! [`WeakRef`] keys a memo on it without keeping the node's subtree alive:
//! the `Weak` pins only the node's own allocation, so its address cannot be
//! reused for another node while the key exists.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use earthsci_ast::{Expr, ExpressionNode};

use crate::helpers::skey;

/// An operator node compared and hashed by identity, NOT keeping it alive:
/// the `Weak` pins only the node's allocation (so the address cannot be
/// reused while the key exists) and lets its children be freed with it. A
/// key whose node is gone ([`WeakRef::is_dead`]) can never be reached again,
/// so a memo may drop it.
pub(crate) struct WeakRef(Weak<ExpressionNode>);

impl WeakRef {
    pub(crate) fn of(a: &Arc<ExpressionNode>) -> WeakRef {
        WeakRef(Arc::downgrade(a))
    }

    pub(crate) fn is_dead(&self) -> bool {
        self.0.strong_count() == 0
    }
}

impl PartialEq for WeakRef {
    fn eq(&self, o: &Self) -> bool {
        Weak::ptr_eq(&self.0, &o.0)
    }
}

impl Eq for WeakRef {}

impl Hash for WeakRef {
    fn hash<H: Hasher>(&self, h: &mut H) {
        (self.0.as_ptr() as usize).hash(h);
    }
}

/// Whether `a` and `b` are the same node (operators) or the same leaf. A
/// float leaf compares by its bits, not `==`: [`map_children_shared`] then
/// never drops a rewrite of a float (`0.0` → `-0.0`, which `==` calls equal),
/// and a NaN leaf is the same as itself instead of forcing a rebuild.
pub(crate) fn same_expr(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Operator(x), Expr::Operator(y)) => Arc::ptr_eq(x, y),
        (Expr::Number(x), Expr::Number(y)) => x.to_bits() == y.to_bits(),
        (Expr::Operator(_), _) | (_, Expr::Operator(_)) => false,
        _ => a == b,
    }
}

/// `e` with `f` applied to every child, returning `e` itself (the same
/// node) when `f` returned every child unchanged. Structurally the result is
/// what rebuilding the node would give; keeping the node keeps the sharing
/// that identity memos rely on.
///
/// The children are mapped first and the node is copied only when one
/// changed: `ExpressionNode::map_children` clones every field (ranges,
/// regions, the child `Arc`s), which on an unchanged node is a copy thrown
/// away. A mapped child is matched back to its slot by the slot's address
/// (`for_each_child` and `map_children` both visit the node's own fields),
/// not by visit order, which differs for the hash-map children (`axes`,
/// `bindings`).
pub(crate) fn map_children_shared(e: &Expr, f: &mut impl FnMut(&Expr) -> Expr) -> Expr {
    let Expr::Operator(n) = e else {
        return e.clone();
    };
    let mut changed = false;
    let mut out: Vec<(*const Expr, Expr)> = Vec::new();
    n.for_each_child(&mut |c: &Expr| {
        let r = f(c);
        if !same_expr(&r, c) {
            changed = true;
        }
        out.push((c as *const Expr, r));
    });
    if !changed {
        return e.clone();
    }
    let mut next = 0usize;
    Expr::operator(n.map_children(&mut |c: &Expr| {
        let k = slot_of(&out, &mut next, c);
        out[k].1.clone()
    }))
}

/// The entry of `slots` (child slot address, value; in `for_each_child`
/// order) for the child `c` that `map_children` is visiting. The walks agree
/// on order except inside a hash-map field, so the entry after the last one
/// matched is tried first.
fn slot_of<T>(slots: &[(*const Expr, T)], next: &mut usize, c: &Expr) -> usize {
    let at = c as *const Expr;
    let k = if slots.get(*next).is_some_and(|x| x.0 == at) {
        *next
    } else {
        slots
            .iter()
            .position(|x| x.0 == at)
            .expect("map_children visits the children for_each_child does")
    };
    *next = k + 1;
    k
}

/// Interning table for structural equality: `sid(a) == sid(b)` iff `a` and
/// `b` are the same structure, which is `skey(a) == skey(b)` except that a
/// non-finite float leaf (which `skey` writes as JSON `null`, as it does
/// every `Inf`, `-Inf` and `NaN`) is told apart by its bits. Ids are
/// meaningful only within one index.
///
/// The identity memo is keyed by [`WeakRef`], so the index does not keep a
/// keyed tree alive; the keys of freed nodes are swept out whenever the memo
/// has doubled since the last sweep.
pub(crate) struct StructIndex {
    ids: HashMap<String, usize>,
    memo: HashMap<WeakRef, usize>,
    next_sweep: usize,
}

const MIN_SWEEP: usize = 1 << 12;

impl Default for StructIndex {
    fn default() -> Self {
        StructIndex {
            ids: HashMap::new(),
            memo: HashMap::new(),
            next_sweep: MIN_SWEEP,
        }
    }
}

impl StructIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Drop the memo keys of freed nodes once the memo has doubled since the
    /// last sweep (amortized O(1) per key). A freed node's key can never be
    /// looked up again: its address stays pinned by the `Weak` until then.
    fn maybe_sweep(&mut self) {
        if self.memo.len() < self.next_sweep {
            return;
        }
        self.memo.retain(|k, _| !k.is_dead());
        self.next_sweep = (2 * self.memo.len()).max(MIN_SWEEP);
    }

    fn intern_key(&mut self, k: String) -> usize {
        let n = self.ids.len() + 1;
        *self.ids.entry(k).or_insert(n)
    }

    /// The structural id of `e`.
    pub(crate) fn sid(&mut self, e: &Expr) -> usize {
        let Expr::Operator(a) = e else {
            // A leaf's key is its own `skey`, behind a prefix no shell key
            // (which always starts with `{`) can share. `skey` writes every
            // non-finite float as `null`, so `x*Inf`, `x*-Inf` and `x*NaN`
            // would share an id (and a memoized simplification); such a
            // leaf is keyed by its bits instead.
            if let Expr::Number(x) = e {
                if !x.is_finite() {
                    return self.intern_key(format!("F{:016x}", x.to_bits()));
                }
            }
            return self.intern_key(format!("L{}", skey(e)));
        };
        self.maybe_sweep();
        let k = WeakRef::of(a);
        if let Some(&id) = self.memo.get(&k) {
            return id;
        }
        let mut kids: Vec<(*const Expr, usize)> = Vec::new();
        a.for_each_child(&mut |c: &Expr| kids.push((c as *const Expr, self.sid(c))));
        // Every child position holds a placeholder, so a placeholder can
        // never be confused with a real variable of the same name. A child
        // is matched to its id by its slot's address, as in
        // `map_children_shared` (the two walks order hash-map children
        // differently).
        let mut next = 0usize;
        let shell = a.map_children(&mut |c: &Expr| {
            let k = slot_of(&kids, &mut next, c);
            Expr::Variable(format!("__sid⟦{}⟧", kids[k].1))
        });
        let id = self.intern_key(skey(&Expr::Operator(Arc::new(shell))));
        self.memo.insert(k, id);
        id
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
    fn non_finite_leaves_are_distinct() {
        let times = |x: f64| {
            Expr::operator(ExpressionNode {
                op: "*".into(),
                args: vec![Expr::Variable("x".into()), Expr::Number(x)],
                ..Default::default()
            })
        };
        let es = [
            times(f64::INFINITY),
            times(f64::NEG_INFINITY),
            times(f64::NAN),
            times(1.0),
        ];
        // `skey` cannot tell the first three apart; `sid` must.
        assert_eq!(skey(&es[0]), skey(&es[1]));
        let mut ix = StructIndex::new();
        let ids: Vec<usize> = es.iter().map(|e| ix.sid(e)).collect();
        for a in 0..ids.len() {
            for b in a + 1..ids.len() {
                assert_ne!(ids[a], ids[b], "{a} vs {b}");
            }
        }
        // ... and the same structure is still one id.
        assert_eq!(ix.sid(&times(f64::NEG_INFINITY)), ids[1]);
        assert_eq!(ix.sid(&times(f64::NAN)), ids[2]);
    }

    #[test]
    fn index_does_not_keep_trees_alive() {
        let e = ex(json!({"op": "+", "args": [1, {"op": "*", "args": [2, "x"]}]}));
        let Expr::Operator(a) = &e else {
            unreachable!()
        };
        let w = Arc::downgrade(a);
        let mut ix = StructIndex::new();
        let id = ix.sid(&e);
        drop(e);
        assert_eq!(w.strong_count(), 0, "the index holds the tree");
        let again = ex(json!({"op": "+", "args": [1, {"op": "*", "args": [2, "x"]}]}));
        assert_eq!(ix.sid(&again), id);
    }

    #[test]
    fn mapped_children_land_in_their_slots() {
        // `axes` is a hash map: `for_each_child` visits it sorted,
        // `map_children` in hash order. Each mapped child must land in its
        // own slot whatever the order.
        let mut axes = HashMap::new();
        for k in ["a", "b", "c", "d", "e", "f", "g", "h"] {
            axes.insert(k.to_string(), Expr::Variable(k.to_string()));
        }
        let e = Expr::operator(ExpressionNode {
            op: "table_lookup".into(),
            args: vec![Expr::Variable("p".into())],
            axes: Some(axes),
            ..Default::default()
        });
        let m = map_children_shared(&e, &mut |c| match c {
            Expr::Variable(v) => Expr::Variable(format!("{v}{v}")),
            _ => c.clone(),
        });
        let Expr::Operator(mn) = &m else {
            unreachable!()
        };
        for (k, v) in mn.axes.as_ref().unwrap() {
            assert_eq!(v, &Expr::Variable(format!("{k}{k}")));
        }
        assert_eq!(mn.args[0], Expr::Variable("pp".into()));
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
        let z = ex(json!({"op": "+", "args": [0.0, "x"]}));
        let nz = map_children_shared(&z, &mut |c| match c {
            Expr::Number(x) if *x == 0.0 => Expr::Number(-0.0),
            _ => c.clone(),
        });
        assert!(!same_expr(&nz, &z));
        let nan = Expr::Number(f64::NAN);
        assert!(same_expr(&nan, &nan));
        assert_eq!(
            skey(&other),
            skey(&ex(
                json!({"op": "+", "args": [3, {"op": "*", "args": [2, "x"]}]})
            ))
        );
    }
}
