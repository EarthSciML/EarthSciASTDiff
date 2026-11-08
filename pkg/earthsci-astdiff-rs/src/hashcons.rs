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

#[cfg(test)]
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

/// An operator node compared and hashed by identity, holding a strong
/// reference: for a short-lived memo over a tree it walks (the [`Interner`],
/// a single scan), where keeping the node alive is what pins its address.
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
pub(crate) fn slot_of<T>(slots: &[(*const Expr, T)], next: &mut usize, c: &Expr) -> usize {
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
/// [`StructIndex::exact`] instead keys leaves as the reference does
/// (`isequal`): an integer is never a float (`1` is not `1.0`), `0.0` is not
/// `-0.0`, and every NaN is one leaf. Subtree sharing (esm-jacobian-spec
/// §4.1) and the band interner use it, so neither merges what the reference
/// keeps apart.
///
/// The identity memo is keyed by [`WeakRef`], so the index does not keep a
/// keyed tree alive; the keys of freed nodes are swept out whenever the memo
/// has doubled since the last sweep.
pub(crate) struct StructIndex {
    /// ids handed out so far; every table below draws from this one count
    n: usize,
    /// leaves by value as the reference keys them, not as `skey` writes them
    exact: bool,
    // leaves, by value (floats by bits)
    ints: HashMap<i64, usize>,
    floats: HashMap<u64, usize>,
    nonfinite: HashMap<u64, usize>,
    vars: HashMap<String, usize>,
    /// operator nodes with no field but `op` and `args`, by operator and
    /// argument ids — the bulk of a coefficient, keyed without serializing
    plain: HashMap<(String, Vec<usize>), usize>,
    /// every other operator node, by the `skey` of its shell
    shells: HashMap<String, usize>,
    // node address → (the `Weak` pinning that address, id)
    memo: HashMap<usize, (Weak<ExpressionNode>, usize)>,
    next_sweep: usize,
}

const MIN_SWEEP: usize = 1 << 12;

impl Default for StructIndex {
    fn default() -> Self {
        StructIndex {
            n: 0,
            exact: false,
            ints: HashMap::new(),
            floats: HashMap::new(),
            nonfinite: HashMap::new(),
            vars: HashMap::new(),
            plain: HashMap::new(),
            shells: HashMap::new(),
            memo: HashMap::new(),
            next_sweep: MIN_SWEEP,
        }
    }
}

/// A leaf by what its `skey` writes, without writing it. A float is
/// written as its serialized number: EarthSciAST writes an integral float as
/// an integer token (`1.0` and `-0.0` as `1` and `0`, which `skey` then
/// cannot tell from the integer leaves), any other finite one by its
/// shortest round-trip digits (so by its bits). A non-finite float, which
/// `skey` writes as `null`, keeps its bits.
pub(crate) enum LeafKey<'a> {
    Int(i64),
    Float(u64),
    NonFinite(u64),
    Var(&'a str),
}

impl<'a> LeafKey<'a> {
    pub(crate) fn of(e: &'a Expr) -> LeafKey<'a> {
        match e {
            Expr::Integer(i) => LeafKey::Int(*i),
            Expr::Number(x) if !x.is_finite() => LeafKey::NonFinite(x.to_bits()),
            Expr::Number(x) => match crate::helpers::ser_expr(e) {
                serde_json::Value::Number(n) => match (n.as_i64(), n.as_f64()) {
                    (Some(i), _) => LeafKey::Int(i),
                    (None, Some(f)) => LeafKey::Float(f.to_bits()),
                    _ => LeafKey::Float(x.to_bits()),
                },
                _ => LeafKey::Float(x.to_bits()),
            },
            Expr::Variable(v) => LeafKey::Var(v),
            Expr::Operator(_) => unreachable!("an operator is not a leaf"),
        }
    }
}

/// The id `k` has in `table`, or the next one.
fn id_in<K: std::hash::Hash + Eq>(table: &mut HashMap<K, usize>, n: &mut usize, k: K) -> usize {
    *table.entry(k).or_insert_with(|| {
        *n += 1;
        *n
    })
}

/// Whether `a` has no field but `op` and `args` set, i.e. serializes as
/// `{"op", "args"}` alone (every other field is skipped when unset).
pub(crate) fn plain_node(a: &ExpressionNode) -> bool {
    let mut rest = a.clone();
    rest.op.clear();
    rest.args.clear();
    rest == ExpressionNode::default()
}

impl StructIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// An index whose leaves are keyed as the reference keys them (see the
    /// type's docs).
    pub(crate) fn exact() -> Self {
        StructIndex {
            exact: true,
            ..Self::default()
        }
    }

    /// The id the identity memo holds for operator node `a`, if keyed.
    pub(crate) fn cached(&self, a: &Arc<ExpressionNode>) -> Option<usize> {
        self.memo.get(&(Arc::as_ptr(a) as usize)).map(|m| m.1)
    }

    /// Record `id` for operator node `a` in the identity memo.
    pub(crate) fn remember(&mut self, a: &Arc<ExpressionNode>, id: usize) {
        self.maybe_sweep();
        self.memo
            .insert(Arc::as_ptr(a) as usize, (Arc::downgrade(a), id));
    }

    /// Drop the memo keys of freed nodes once the memo has doubled since the
    /// last sweep (amortized O(1) per key). A freed node's key can never be
    /// looked up again: its address stays pinned by the `Weak` until then.
    fn maybe_sweep(&mut self) {
        if self.memo.len() < self.next_sweep {
            return;
        }
        self.memo.retain(|_, (w, _)| w.strong_count() > 0);
        self.next_sweep = (2 * self.memo.len()).max(MIN_SWEEP);
    }

    /// A leaf's id: two leaves share one iff `skey` writes them alike
    /// ([`LeafKey`]), except that the non-finite floats `skey` writes alike
    /// (`null`) are told apart by their bits — `x*Inf`, `x*-Inf` and `x*NaN`
    /// must not share an id (or a memoized simplification).
    pub(crate) fn leaf_id(&mut self, e: &Expr) -> usize {
        if self.exact {
            return match e {
                Expr::Integer(i) => id_in(&mut self.ints, &mut self.n, *i),
                Expr::Number(x) if x.is_nan() => {
                    id_in(&mut self.nonfinite, &mut self.n, f64::NAN.to_bits())
                }
                Expr::Number(x) => id_in(&mut self.floats, &mut self.n, x.to_bits()),
                Expr::Variable(v) => match self.vars.get(v.as_str()) {
                    Some(&id) => id,
                    None => id_in(&mut self.vars, &mut self.n, v.clone()),
                },
                Expr::Operator(_) => unreachable!("an operator is not a leaf"),
            };
        }
        match LeafKey::of(e) {
            LeafKey::Int(i) => id_in(&mut self.ints, &mut self.n, i),
            LeafKey::Float(b) => id_in(&mut self.floats, &mut self.n, b),
            LeafKey::NonFinite(b) => id_in(&mut self.nonfinite, &mut self.n, b),
            LeafKey::Var(v) => match self.vars.get(v) {
                Some(&id) => id,
                None => id_in(&mut self.vars, &mut self.n, v.to_string()),
            },
        }
    }

    /// The id of operator node `a` whose children have ids `kids` (in
    /// `for_each_child` order, with each child's slot address). A node with
    /// only `op` and `args` is keyed by its operator and argument ids. Any
    /// other is keyed by the `skey` of its SHELL, every child slot holding a
    /// placeholder carrying the child's id, matched to the child by slot
    /// address as in [`map_children_shared`] (the two walks order the
    /// hash-map children, `axes` and `bindings`, differently); a shell that
    /// serializes as `{"op", "args"}` alone (a set field that is never
    /// written) joins the plain table, so the two tables split nodes exactly
    /// where `skey` does. O(node fields), not O(subtree).
    pub(crate) fn shell_id(&mut self, a: &ExpressionNode, kids: &[(*const Expr, usize)]) -> usize {
        let plain_key = || (a.op.clone(), kids.iter().map(|k| k.1).collect::<Vec<_>>());
        if plain_node(a) {
            return id_in(&mut self.plain, &mut self.n, plain_key());
        }
        let mut next = 0usize;
        let shell = a.map_children(&mut |c: &Expr| {
            let k = slot_of(kids, &mut next, c);
            Expr::Variable(format!("__sid⟦{}⟧", kids[k].1))
        });
        let v = crate::helpers::ser_expr(&Expr::Operator(Arc::new(shell)));
        if v.as_object()
            .is_some_and(|m| m.len() == 2 && m.contains_key("op") && m.contains_key("args"))
        {
            return id_in(&mut self.plain, &mut self.n, plain_key());
        }
        id_in(
            &mut self.shells,
            &mut self.n,
            crate::helpers::stable_json(&v),
        )
    }

    /// The structural id of `e`.
    pub(crate) fn sid(&mut self, e: &Expr) -> usize {
        let Expr::Operator(a) = e else {
            return self.leaf_id(e);
        };
        // Looked up by address alone; the `Weak` stored with the id is what
        // keeps that address from being reused while the entry exists.
        let k = Arc::as_ptr(a) as usize;
        if let Some(&(_, id)) = self.memo.get(&k) {
            return id;
        }
        let mut kids: Vec<(*const Expr, usize)> = Vec::new();
        a.for_each_child(&mut |c: &Expr| kids.push((c as *const Expr, self.sid(c))));
        let id = self.shell_id(a, &kids);
        self.remember(a, id);
        id
    }
}

/// A structural hash used as a NEGATIVE filter (hashcons.jl `StructHash`):
/// equal structures always share it, so a mismatch proves two subtrees
/// differ in one `u64` compare, and only a match pays for the exact answer
/// from [`StructIndex::sid`]. It covers the operator and the children only,
/// not a node's other fields, so unequal trees may collide: a match proves
/// nothing, and it is never an identity.
///
/// Memoized on node identity (the memo's `Weak` pins each keyed address, as
/// in [`StructIndex`]); a leaf hashes its [`LeafKey`], as `sid` keys it.
pub(crate) struct StructHash {
    memo: HashMap<usize, (Weak<ExpressionNode>, u64)>,
    next_sweep: usize,
}

impl Default for StructHash {
    fn default() -> Self {
        StructHash {
            memo: HashMap::new(),
            next_sweep: MIN_SWEEP,
        }
    }
}

/// One round of a 64-bit mix (splitmix64's finalizer over `h ^ x`).
fn mix(h: u64, x: u64) -> u64 {
    let mut z = (h ^ x).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn str_hash(seed: u64, s: &str) -> u64 {
    let mut h = seed;
    for c in s.as_bytes().chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = mix(h, u64::from_le_bytes(w));
    }
    mix(h, s.len() as u64)
}

impl StructHash {
    /// The hash of `e`'s operator/child structure: `sid(a) == sid(b)`
    /// implies `shash(a) == shash(b)`, not the converse.
    pub(crate) fn shash(&mut self, e: &Expr) -> u64 {
        match e {
            // leaves hash their `LeafKey`, which `sid` equality implies
            Expr::Integer(_) | Expr::Number(_) | Expr::Variable(_) => match LeafKey::of(e) {
                LeafKey::Int(i) => mix(0xc2b2_ae3d_27d4_eb4f, i as u64),
                LeafKey::Float(b) => mix(0x1656_67b1_9e37_79f9, b),
                LeafKey::NonFinite(b) => mix(0x85eb_ca6b_c2b2_ae35, b),
                LeafKey::Var(v) => str_hash(0x27d4_eb2f_1656_67c5, v),
            },
            Expr::Operator(a) => {
                let k = Arc::as_ptr(a) as usize;
                if let Some(&(_, h)) = self.memo.get(&k) {
                    return h;
                }
                let mut h = str_hash(0x9e37_79b9_7f4a_7c11, &a.op);
                a.for_each_child(&mut |c: &Expr| h = mix(h, self.shash(c)));
                if self.memo.len() >= self.next_sweep {
                    self.memo.retain(|_, (w, _)| w.strong_count() > 0);
                    self.next_sweep = (2 * self.memo.len()).max(MIN_SWEEP);
                }
                self.memo.insert(k, (Arc::downgrade(a), h));
                h
            }
        }
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
pub(crate) struct Interner {
    ix: StructIndex,
    canon: HashMap<usize, Expr>,
    // node identity → (representative, structural id)
    memo: HashMap<NodeRef, (Expr, usize)>,
}

impl Interner {
    /// Leaves keyed exactly ([`StructIndex::exact`]): a representative never
    /// trades an integer leaf for a float one (or `-0.0` for `0.0`), so the
    /// subtree sharing that runs on the interned coefficients counts them as
    /// the reference does.
    pub(crate) fn new() -> Self {
        Interner {
            ix: StructIndex::exact(),
            canon: HashMap::new(),
            memo: HashMap::new(),
        }
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
            let mut ids: Vec<(*const Expr, usize)> = Vec::new();
            n.for_each_child(&mut |c: &Expr| {
                let kid = match c {
                    Expr::Operator(ca) => self.memo[&NodeRef(ca.clone())].clone(),
                    _ => (c.clone(), self.ix.leaf_id(c)),
                };
                ids.push((c as *const Expr, kid.1));
                kids.push(kid);
            });
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
    fn interner_keeps_integer_and_float_leaves_apart() {
        // Subtree sharing runs on the interned coefficients and must count
        // `x + 1` and `x + 1.0` as different subtrees, as the reference does;
        // so the interner must not make one the representative of the other.
        let node = |op: &str, args: Vec<Expr>| {
            Expr::operator(ExpressionNode {
                op: op.into(),
                args,
                ..Default::default()
            })
        };
        let x = || Expr::Variable("x".into());
        let mut t = Interner::new();
        let a = t.intern(&node("exp", vec![node("+", vec![x(), Expr::Integer(1)])]));
        let b = t.intern(&node("exp", vec![node("+", vec![x(), Expr::Number(1.0)])]));
        assert!(!same_expr(&a, &b));
        let leaf = |e: &Expr| match e {
            Expr::Operator(n) => match &n.args[0] {
                Expr::Operator(m) => m.args[1].clone(),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        assert!(matches!(leaf(&b), Expr::Number(_)));
        let c = t.intern(&node("exp", vec![node("+", vec![x(), Expr::Integer(1)])]));
        assert!(same_expr(&a, &c));
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
