//! Expression constructors, structural keys, and the differentiation site
//! (mirror of the reference's expr_helpers).

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use earthsci_ast::canonical_json;
use earthsci_ast::{Expr, ExpressionNode};
use serde_json::Value;

use crate::hashcons::WeakRef;

pub fn op(o: &str, args: Vec<Expr>) -> Expr {
    Expr::operator(ExpressionNode {
        op: o.to_string(),
        args,
        ..Default::default()
    })
}

pub fn is_zero(e: &Expr) -> bool {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Number(x) if *x == 0.0)
}

pub fn is_one(e: &Expr) -> bool {
    matches!(e, Expr::Integer(1)) || matches!(e, Expr::Number(x) if *x == 1.0)
}

/// n-ary `+` dropping literal zeros, collapsing to a single argument.
pub fn add(args: Vec<Expr>) -> Expr {
    let a: Vec<Expr> = args.into_iter().filter(|x| !is_zero(x)).collect();
    match a.len() {
        0 => Expr::Integer(0),
        1 => a.into_iter().next().unwrap(),
        _ => op("+", a),
    }
}

/// n-ary `*` short-circuiting on a literal zero, dropping literal ones.
pub fn mul(args: Vec<Expr>) -> Expr {
    if args.iter().any(is_zero) {
        return Expr::Integer(0);
    }
    let a: Vec<Expr> = args.into_iter().filter(|x| !is_one(x)).collect();
    match a.len() {
        0 => Expr::Integer(1),
        1 => a.into_iter().next().unwrap(),
        _ => op("*", a),
    }
}

pub fn neg(x: Expr) -> Expr {
    if is_zero(&x) {
        Expr::Integer(0)
    } else {
        op("-", vec![x])
    }
}

/// A float in the Julia reference's layout (EarthSciAST CONFORMANCE_SPEC
/// §5.5.3.1 rule 2): shortest round-trip digits, positional iff the decimal
/// exponent of the first significant digit is in `-4..=5`, otherwise
/// `d.f…e<exp>` with at least one fraction digit (`1.0e-5`, `2.4405875e6`).
/// serde_json has the same digits but switches layout at other magnitudes
/// (`0.0000727…`, `2440587.5`). Not `earthsci_ast::format_canonical_float`,
/// which is the RFC §5.4.6 canonical-form layout.
pub fn canonical_float(f: f64) -> String {
    assert!(f.is_finite(), "non-finite float {f} has no JSON form");
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` is the shortest round-trip digits as `d[.ddd]e<exp>`
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').unwrap();
    let e10: i32 = exp.parse().unwrap();
    let ds: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    let body = if (-4..=5).contains(&e10) {
        if e10 < 0 {
            format!("0.{}{ds}", "0".repeat((-e10 - 1) as usize))
        } else {
            let k = (e10 + 1) as usize;
            let int = format!("{:0<k$}", &ds[..k.min(ds.len())]);
            let frac = if ds.len() > k { &ds[k..] } else { "0" };
            format!("{int}.{frac}")
        }
    } else {
        let frac = if ds.len() > 1 { &ds[1..] } else { "0" };
        format!("{}.{frac}e{e10}", &ds[..1])
    };
    format!("{sign}{body}")
}

/// Deterministic JSON: sorted object keys at every level, floats in the
/// reference's layout ([`canonical_float`]), every other scalar by
/// serde_json. A `const` node's `value` is written in canonical-number form
/// (CONFORMANCE_SPEC §5.5.3.1 rule 1: an integral, i64-representable number
/// as an integer, at any depth), as earthsci-ast ≥ 0.10 serializes it, so the
/// text does not depend on the earthsci-ast release.
pub fn stable_json(v: &Value) -> String {
    let mut out = String::new();
    write_stable(v, &mut out);
    out
}

fn write_stable(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            let is_const = m.get("op").and_then(Value::as_str) == Some("const");
            out.push('{');
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push(':');
                if is_const && k.as_str() == "value" {
                    write_stable(&canon_numbers(&m[*k]), out);
                } else {
                    write_stable(&m[*k], out);
                }
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_stable(x, out);
            }
            out.push(']');
        }
        Value::Number(n) if !(n.is_i64() || n.is_u64()) => {
            out.push_str(&canonical_float(n.as_f64().unwrap()))
        }
        _ => out.push_str(&serde_json::to_string(v).unwrap()),
    }
}

/// §5.5.3.1 rule 1 over a `const` value: every integral, i64-representable
/// float becomes an integer, nested arrays and objects walked.
fn canon_numbers(v: &Value) -> Value {
    match v {
        Value::Number(n) if !(n.is_i64() || n.is_u64()) => match n.as_f64() {
            Some(f)
                if f.is_finite()
                    && f.fract() == 0.0
                    && (-(2f64.powi(63))..2f64.powi(63)).contains(&f) =>
            {
                Value::from(f as i64)
            }
            _ => v.clone(),
        },
        Value::Array(a) => Value::Array(a.iter().map(canon_numbers).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), canon_numbers(x)))
                .collect(),
        ),
        _ => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::canonical_float;

    #[test]
    fn const_values_are_written_in_canonical_number_form() {
        let v = serde_json::json!({
            "op": "+",
            "args": [
                {"op": "const", "args": [], "value": [[1.0, 0.5], [-2.0, 1e300]]},
                2.0
            ]
        });
        // only inside a const `value`: the coefficient literal keeps its float
        assert_eq!(
            super::stable_json(&v),
            r#"{"args":[{"args":[],"op":"const","value":[[1,0.5],[-2,1.0e300]]},2.0],"op":"+"}"#
        );
    }

    #[test]
    fn canonical_float_matches_the_reference_layout() {
        // CONFORMANCE_SPEC §5.5.3.1's probe table, plus Julia `show` values
        for (v, want) in [
            (1.0e-9, "1.0e-9"),
            (1.5e-5, "1.5e-5"),
            (0.000123456, "0.000123456"),
            (2.5, "2.5"),
            (999999.5, "999999.5"),
            (1000000.5, "1.0000005e6"),
            (123456789.25, "1.2345678925e8"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (1.0e6, "1.0e6"),
            (1.0e22, "1.0e22"),
            (0.0001, "0.0001"),
            (1.0e-5, "1.0e-5"),
            (-2.5e-7, "-2.5e-7"),
            (2440587.5, "2.4405875e6"),
            (7.27220521664304e-5, "7.27220521664304e-5"),
            (5e-324, "5.0e-324"),
            (1.7976931348623157e308, "1.7976931348623157e308"),
        ] {
            assert_eq!(canonical_float(v), want, "for {v:e}");
        }
    }
}

pub fn ser_expr(e: &Expr) -> Value {
    serde_json::to_value(e).expect("expression serializes")
}

/// Full-structure key for ANY expression (array ops included).
pub fn skey(e: &Expr) -> String {
    stable_json(&ser_expr(e))
}

/// Term-identity key for SCALAR expressions (RFC §5.4 canonical form).
pub fn ckey(e: &Expr) -> String {
    canonical_json(e).expect("scalar-core canonical form")
}

/// A site's identity key (expr_helpers.jl `_site_key`): the canonical JSON,
/// or — for an `index` node whose subscripts hold an array op the canonical
/// form refuses (a level search `u[κ]` with κ a min-faq, spec §3.5) — `~`
/// followed by its stable JSON, which sorts after every canonical key.
pub fn site_key(e: &Expr) -> String {
    match canonical_json(e) {
        Ok(k) => k,
        Err(_) => format!("~{}", skey(e)),
    }
}

/// The free variables of `e` with every node's own binders subtracted at
/// that node (a faq's ranges and output indices, an integral's variable),
/// plus any `wrt` — EarthSciAST.jl's `free_variables`. (`earthsci_ast`'s
/// `free_variables` subtracts no binder.)
pub fn free_vars(e: &Expr) -> HashSet<String> {
    let mut out = HashSet::new();
    match e {
        Expr::Variable(n) => {
            out.insert(n.clone());
        }
        Expr::Operator(n) => {
            n.for_each_child(&mut |c: &Expr| out.extend(free_vars(c)));
            if let Some(v) = &n.int_var {
                out.remove(v);
            }
            for k in n.ranges.iter().flat_map(|r| r.keys()) {
                out.remove(k);
            }
            for k in n.output_idx.iter().flatten() {
                out.remove(k);
            }
            if let Some(w) = &n.wrt {
                out.insert(w.clone());
            }
        }
        _ => {}
    }
    out
}

/// A bare variable reference or one symbolic `index(v, i…)` cell.
#[derive(Clone, Debug)]
pub struct Site {
    pub expr: Expr,
    pub key: String,
}

impl Site {
    pub fn of(e: &Expr) -> Site {
        Site {
            expr: e.clone(),
            key: site_key(e),
        }
    }
}

pub fn is_site(e: &Expr, s: &Site) -> bool {
    let candidate =
        matches!(e, Expr::Variable(_)) || matches!(e, Expr::Operator(n) if n.op == "index");
    candidate && site_key(e) == s.key
}

/// Structural occurrence — THE sparsity primitive (never a zero test).
pub fn occurs(e: &Expr, s: &Site) -> bool {
    if is_site(e, s) {
        return true;
    }
    let Expr::Operator(n) = e else { return false };
    n.any_child(&mut |c: &Expr| occurs(c, s))
}

// ── every site of one variable at once ───────────────────────────────────────
//
// The band calculus differentiates a body once per site of `v`, and a faq
// body is clipped into one body per region first: on ReSEACT's lowered
// advection operator that is 343 regions × ~16 sites, each body ~4.5×10⁴
// nodes. `occurs` at every node of a derivative walk costs the subtree's
// size, so one scan is Θ(size × depth), and `is_site` re-serializes every
// `index` node it meets. `OccIndex` (the reference's `_OccIndex`) answers
// both for ALL sites of `v` from one walk: each node gets the set of site
// keys under it, and each `index` node of `v` its own key, both memoized on
// node identity. Clipping keeps the subtrees it does not touch
// (`hashcons::map_children_shared`), so one index serves every region.
//
// Exact: a node's key set holds a site's key iff `occurs` would find it, and
// `SiteScan::is_site` makes `is_site`'s decision with the key read from the
// memo. The derivative memo is exact because the derivative of a node with
// respect to a site depends on the two alone and nothing rewrites an
// expression in place.

/// A set of small integers (site key ids).
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct Bits(Vec<u64>);

impl Bits {
    fn one(i: usize) -> Bits {
        let mut b = Bits(vec![0; i / 64 + 1]);
        b.0[i / 64] |= 1 << (i % 64);
        b
    }
    pub(crate) fn contains(&self, i: usize) -> bool {
        self.0.get(i / 64).is_some_and(|w| w & (1 << (i % 64)) != 0)
    }
    fn is_empty(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }
    fn is_subset(&self, o: &Bits) -> bool {
        self.0
            .iter()
            .enumerate()
            .all(|(k, w)| w & !o.0.get(k).copied().unwrap_or(0) == 0)
    }
    fn union_with(&mut self, o: &Bits) {
        if o.0.len() > self.0.len() {
            self.0.resize(o.0.len(), 0);
        }
        for (k, w) in o.0.iter().enumerate() {
            self.0[k] |= w;
        }
    }
}

/// The variable `e` is a site of, if it is one: `v` itself, or an `index`
/// read `index(v, ...)`. The one definition of a site's variable, shared by
/// the derivative scan, the occurrence index and the nested-sum walk.
pub(crate) fn site_var(e: &Expr) -> Option<&str> {
    match e {
        Expr::Variable(n) => Some(n),
        Expr::Operator(n) => index_var(n),
        _ => None,
    }
}

/// The variable an `index` node reads, if `n` is one over a variable.
fn index_var(n: &ExpressionNode) -> Option<&str> {
    match n.args.first() {
        Some(Expr::Variable(t)) if n.op == "index" => Some(t),
        _ => None,
    }
}

type DerivMemo = Rc<RefCell<HashMap<WeakRef, Expr>>>;

/// The total entry count of an index's derivative memos, kept by the scans
/// that insert into them, so a sweep decision never walks every memo.
type Count = Rc<Cell<usize>>;

/// Occurrence of every site of variable `v`, memoized on node identity, for
/// the derivative scans of one band computation. See [`SiteScan`].
///
/// Keys are [`WeakRef`]s, so the index does not keep a region body alive
/// after its iteration; the entries of freed nodes are swept out whenever
/// the tables have doubled since the last sweep.
pub(crate) struct OccIndex {
    v: String,
    keyid: HashMap<String, usize>,
    nodekey: HashMap<WeakRef, usize>,
    /// A site node's derivative-memo key (its `skey`), so a site scanned
    /// again (once per region of a clipped body) is not re-serialized.
    sitekey: HashMap<WeakRef, Rc<str>>,
    bits: HashMap<WeakRef, Rc<Bits>>,
    none: Rc<Bits>,
    /// The set of a bare `Variable(v)`, keyed once: one shared `Rc`, so a
    /// parent already holding it skips the union by `Rc::ptr_eq`.
    var: Option<Rc<Bits>>,
    deriv: HashMap<Rc<str>, DerivMemo>,
    deriv_len: Count,
    next_sweep: usize,
}

const MIN_SWEEP: usize = 1 << 12;

impl OccIndex {
    pub(crate) fn new(v: &str) -> OccIndex {
        OccIndex {
            v: v.to_string(),
            keyid: HashMap::new(),
            nodekey: HashMap::new(),
            sitekey: HashMap::new(),
            bits: HashMap::new(),
            none: Rc::new(Bits::default()),
            var: None,
            deriv: HashMap::new(),
            deriv_len: Rc::new(Cell::new(0)),
            next_sweep: MIN_SWEEP,
        }
    }

    fn keyid(&mut self, k: &str) -> usize {
        if let Some(&i) = self.keyid.get(k) {
            return i;
        }
        let i = self.keyid.len();
        self.keyid.insert(k.to_string(), i);
        i
    }

    fn is_v_index(&self, n: &ExpressionNode) -> bool {
        index_var(n) == Some(self.v.as_str())
    }

    fn nodekey(&mut self, e: &Expr, a: &Arc<ExpressionNode>) -> usize {
        let k = WeakRef::of(a);
        if let Some(&i) = self.nodekey.get(&k) {
            return i;
        }
        let i = self.keyid(&site_key(e));
        self.nodekey.insert(k, i);
        i
    }

    /// O(1): the derivative memos' share is the running count.
    fn entries(&self) -> usize {
        self.nodekey.len() + self.sitekey.len() + self.bits.len() + self.deriv_len.get()
    }

    /// Drop the entries of nodes that have been freed, once the tables have
    /// doubled since the last sweep (amortized O(1) per entry). Called
    /// between scans only, so no derivative memo is borrowed.
    fn maybe_sweep(&mut self) {
        if self.entries() < self.next_sweep {
            return;
        }
        self.nodekey.retain(|k, _| !k.is_dead());
        self.sitekey.retain(|k, _| !k.is_dead());
        self.bits.retain(|k, _| !k.is_dead());
        let mut n = 0;
        for m in self.deriv.values() {
            let mut m = m.borrow_mut();
            m.retain(|k, _| !k.is_dead());
            n += m.len();
        }
        self.deriv_len.set(n);
        self.next_sweep = (2 * self.entries()).max(MIN_SWEEP);
    }

    /// The ids of the site keys of `v` at or under `e`. A stored set is
    /// never mutated: a node whose sites all come from one child shares
    /// that child's set.
    fn site_bits(&mut self, e: &Expr) -> Rc<Bits> {
        let a = match e {
            Expr::Variable(n) if *n == self.v => {
                if let Some(b) = &self.var {
                    return b.clone();
                }
                let b = Rc::new(Bits::one(self.keyid(&site_key(e))));
                self.var = Some(b.clone());
                return b;
            }
            Expr::Operator(a) => a,
            _ => return self.none.clone(),
        };
        let k = WeakRef::of(a);
        if let Some(b) = self.bits.get(&k) {
            return b.clone();
        }
        let mut out = if self.is_v_index(a) {
            Rc::new(Bits::one(self.nodekey(e, a)))
        } else {
            self.none.clone()
        };
        a.for_each_child(&mut |c: &Expr| {
            let b = self.site_bits(c);
            if b.is_empty() || Rc::ptr_eq(&b, &out) {
                return;
            }
            if out.is_empty() {
                out = b;
            } else if !b.is_subset(&out) {
                Rc::make_mut(&mut out).union_with(&b);
            }
        });
        self.bits.insert(k, out.clone());
        out
    }
}

/// The occurrence argument of one derivative scan ([`crate::rules::dscalar`])
/// for site `s`: answered from the shared [`OccIndex`] when `s` is a site of
/// its variable (else as [`occurs`] does), with a memo of the derivatives
/// already taken, so a subtree shared by several parents, or by several
/// regions of one clipped body, is differentiated once. The memo is keyed
/// by the site's own expression, so two spellings of one site never share.
pub(crate) struct SiteScan<'a> {
    ix: Option<&'a RefCell<OccIndex>>,
    bit: usize,
    deriv: DerivMemo,
    deriv_len: Count,
}

impl<'a> SiteScan<'a> {
    /// A scan with no shared index: [`is_site`] and [`occurs`] as they are,
    /// and a derivative memo private to this scan.
    pub(crate) fn plain() -> SiteScan<'static> {
        SiteScan {
            ix: None,
            bit: 0,
            deriv: Rc::new(RefCell::new(HashMap::new())),
            deriv_len: Rc::new(Cell::new(0)),
        }
    }

    pub(crate) fn new(ix: &'a RefCell<OccIndex>, s: &Site) -> SiteScan<'a> {
        let mut x = ix.borrow_mut();
        if site_var(&s.expr) != Some(x.v.as_str()) {
            return SiteScan::plain();
        }
        x.maybe_sweep();
        let bit = x.keyid(&s.key);
        let sk: Rc<str> = match &s.expr {
            Expr::Operator(a) => x
                .sitekey
                .entry(WeakRef::of(a))
                .or_insert_with(|| skey(&s.expr).into())
                .clone(),
            e => skey(e).into(),
        };
        let deriv = x.deriv.entry(sk).or_default().clone();
        let deriv_len = x.deriv_len.clone();
        drop(x);
        SiteScan {
            ix: Some(ix),
            bit,
            deriv,
            deriv_len,
        }
    }

    /// [`is_site`], with an `index` node's key read from the index.
    pub(crate) fn is_site(&self, e: &Expr, s: &Site) -> bool {
        let Some(ix) = self.ix else {
            return is_site(e, s);
        };
        match e {
            Expr::Operator(a) if a.op == "index" => {
                let mut x = ix.borrow_mut();
                // A site of `v` has an `index` node of `v` as its key's
                // only spelling among operator nodes.
                x.is_v_index(a) && x.nodekey(e, a) == self.bit
            }
            Expr::Operator(_) => false,
            _ => is_site(e, s),
        }
    }

    /// [`occurs`], answered from the index.
    pub(crate) fn occurs(&self, e: &Expr, s: &Site) -> bool {
        let Some(ix) = self.ix else {
            return occurs(e, s);
        };
        if !matches!(e, Expr::Operator(_)) {
            return self.is_site(e, s);
        }
        let b = ix.borrow_mut().site_bits(e);
        b.contains(self.bit)
    }

    pub(crate) fn memo_get(&self, n: &Arc<ExpressionNode>) -> Option<Expr> {
        self.deriv.borrow().get(&WeakRef::of(n)).cloned()
    }

    pub(crate) fn memo_put(&self, n: &Arc<ExpressionNode>, d: Expr) {
        if self.deriv.borrow_mut().insert(WeakRef::of(n), d).is_none() {
            self.deriv_len.set(self.deriv_len.get() + 1);
        }
    }
}

pub fn occurs_var(e: &Expr, v: &str) -> bool {
    earthsci_ast::free_variables(e).contains(v)
}

/// Rebuild an operator node with mapped children (Expr-level wrapper).
pub fn map_expr_children(e: &Expr, f: &mut impl FnMut(&Expr) -> Expr) -> Expr {
    match e {
        Expr::Operator(n) => Expr::operator(n.map_children(f)),
        _ => e.clone(),
    }
}
