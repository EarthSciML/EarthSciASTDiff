//! A CONTRACTING observed (issue #6; port of the reference's
//! test/contracting_observed_test.jl): `s[i] = Σ_k u[i,k]²` read inside a faq
//! that binds `k` too. Inlined as the bare body, the sum is dropped and the
//! leftover `k` is captured — the column sum collapses to the local cell,
//! with no error. The read must keep the sum, and the band calculus must
//! follow the contraction through it.

use std::collections::HashMap;

use earthsci_ast::Expr;
use earthsci_astdiff::numeric::Env;
use earthsci_astdiff::system::{lhs_state, observed_defs};
use earthsci_astdiff::{assemble_jacobian, inline_observed, jacobian_bands, skey,
                       sysview, Wrt};
use serde_json::{json, Value};

const NX: usize = 2;
const NZ: usize = 3;

fn uval(i: usize, k: usize) -> f64 {
    0.1 * (i + NX * k + 1) as f64
}

fn sval(i: usize) -> f64 {
    (0..NZ).map(|k| uval(i, k).powi(2)).sum()
}

fn idx(v: &str, i: &[&str]) -> Value {
    let mut args = vec![json!(v)];
    args.extend(i.iter().map(|x| json!(x)));
    json!({"op": "index", "args": args})
}

fn sq(x: Value) -> Value {
    json!({"op": "^", "args": [x, 2]})
}

fn neg(x: Value) -> Value {
    json!({"op": "-", "args": [x]})
}

fn sdef(filter: Option<Value>) -> Value {
    let mut d = json!({"op": "faq", "args": [], "output_idx": ["i"],
                       "ranges": {"i": [1, NX], "k": [1, NZ]}, "reduce": "+",
                       "expr": sq(idx("u", &["i", "k"]))});
    if let Some(f) = filter {
        d["filter"] = f;
    }
    d
}

fn doc(body: Value, oidx: [&str; 2], s: Value) -> earthsci_ast::EsmFile {
    doc_with(body, oidx, s, json!({}), vec![])
}

fn doc_with(body: Value, oidx: [&str; 2], s: Value, vars: Value,
            eqs: Vec<Value>) -> earthsci_ast::EsmFile {
    let mut ranges = serde_json::Map::new();
    ranges.insert(oidx[0].into(), json!([1, NX]));
    ranges.insert(oidx[1].into(), json!([1, NZ]));
    let mut variables = json!({
        "u": {"type": "unknown", "shape": ["x", "z"], "default": 0.5},
        "s": {"type": "unknown", "shape": ["x"]}});
    for (k, v) in vars.as_object().unwrap() {
        variables[k] = v.clone();
    }
    let mut equations = vec![json!({"lhs": "s", "rhs": s})];
    equations.extend(eqs);
    equations.push(json!({"lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                          "rhs": {"op": "faq", "args": [], "output_idx": oidx,
                                  "ranges": ranges, "expr": body}}));
    let d = json!({
        "esm": "1.1.0",
        "metadata": {"name": "ContractObs"},
        "index_sets": {"x": {"kind": "interval", "size": NX},
                       "z": {"kind": "interval", "size": NZ}},
        "models": {"ContractObs": {
            "variables": variables,
            "equations": equations}}});
    earthsci_ast::load_document(&d).expect("doc loads")
}

type Want = fn(usize, usize, usize) -> f64;

fn cases(o: [&str; 2]) -> Vec<(&'static str, Value, Want)> {
    let s = idx("s", &[o[0]]);
    vec![
        ("-s[i]", neg(s.clone()), |i, _, kp| -2.0 * uval(i, kp)),
        ("-s[i]^2", neg(sq(s.clone())),
         |i, _, kp| -2.0 * sval(i) * 2.0 * uval(i, kp)),
        ("-s[i]*u[i,k]",
         neg(json!({"op": "*", "args": [s, idx("u", &o)]})),
         |i, k, kp| -2.0 * uval(i, kp) * uval(i, k)
             - if k == kp { sval(i) } else { 0.0 }),
    ]
}

#[test]
fn contracting_observed_keeps_its_sum() {
    let u: Vec<Vec<f64>> = (0..NX).map(|i| (0..NZ).map(|k| uval(i, k)).collect()).collect();
    let env = Env {
        states: HashMap::from([("u".to_string(), json!(u))]),
        params: HashMap::new(),
        t: 0.0,
    };
    for oidx in [["i", "k"], ["i", "j"]] {
        for (label, body, want) in cases(oidx) {
            let file = doc(body, oidx, sdef(None));
            let j = assemble_jacobian(&file, None, Wrt::States, &env)
                .unwrap_or_else(|e| panic!("{label} over {oidx:?}: {}", e.0));
            assert_eq!(j.len(), NX * NZ * NZ,
                       "{label} over {oidx:?}: one dense block per column of u");
            for i in 0..NX {
                for k in 0..NZ {
                    for kp in 0..NZ {
                        let key = (format!("u[{},{}]", i + 1, k + 1),
                                   format!("u[{},{}]", i + 1, kp + 1));
                        let got = j[&key];
                        assert!((got - want(i, k, kp)).abs() <= 1e-13,
                                "{label} over {oidx:?} {key:?}: {got} vs {}",
                                want(i, k, kp));
                    }
                }
            }
        }
    }
}

#[test]
fn filtered_contracting_observed_is_refused() {
    // Refused, as the same filter written inline is — never dropped.
    let filt = json!({"op": "<", "args": ["k", 3]});
    let file = doc(neg(idx("s", &["i"])), ["i", "k"], sdef(Some(filt)));
    assert!(jacobian_bands(&file, None, Wrt::States).is_err());
}

#[test]
fn semiring_max_is_refused() {
    // `semiring` supersedes `reduce`: a max_product node with no `reduce` is
    // a max, not a sum.
    let mut s = sdef(None);
    s.as_object_mut().unwrap().remove("reduce");
    s["semiring"] = json!("max_product");
    let file = doc(neg(idx("s", &["i"])), ["i", "k"], s);
    assert!(jacobian_bands(&file, None, Wrt::States).is_err());
}

fn mul(a: Value, b: Value) -> Value {
    json!({"op": "*", "args": [a, b]})
}

fn ksum(body: Value) -> Value {
    json!({"op": "faq", "args": [], "output_idx": [], "ranges": {"k": [1, NZ]},
           "expr": body})
}

fn env() -> Env {
    let u: Vec<Vec<f64>> = (0..NX).map(|i| (0..NZ).map(|k| uval(i, k)).collect()).collect();
    Env { states: HashMap::from([("u".to_string(), json!(u))]),
          params: HashMap::new(), t: 0.0 }
}

/// Compare against a closed form ∂ D(u)[i,k] / ∂ u[a,b] (0-based), nonzeros
/// only.
fn check(file: &earthsci_ast::EsmFile, want: impl Fn(usize, usize, usize, usize) -> f64) {
    let j = assemble_jacobian(file, None, Wrt::States, &env())
        .unwrap_or_else(|e| panic!("{}", e.0));
    let mut n = 0;
    for i in 0..NX {
        for k in 0..NZ {
            for a in 0..NX {
                for b in 0..NZ {
                    let w = want(i, k, a, b);
                    if w == 0.0 {
                        continue;
                    }
                    n += 1;
                    let key = (format!("u[{},{}]", i + 1, k + 1),
                               format!("u[{},{}]", a + 1, b + 1));
                    let got = j.get(&key).copied()
                        .unwrap_or_else(|| panic!("missing {key:?}"));
                    assert!((got - w).abs() <= 1e-13, "{key:?}: {got} vs {w}");
                }
            }
        }
    }
    assert_eq!(j.len(), n, "pattern");
}

fn cvar() -> Value {
    json!({"c": {"type": "unknown"}})
}

#[test]
fn scalar_sum_observed_bare_read() {
    // c = Σ_k u[1,k], D(u)[i,k] = -c·u[i,k]
    let c: f64 = (0..NZ).map(|k| uval(0, k)).sum();
    let file = doc_with(neg(mul(json!("c"), idx("u", &["i", "k"]))), ["i", "k"],
                        sdef(None), cvar(),
                        vec![json!({"lhs": "c", "rhs": ksum(json!(
                            {"op": "index", "args": ["u", 1, "k"]}))})]);
    check(&file, |i, k, a, b| {
        -(if (a, b) == (i, k) { c } else { 0.0 }) - if a == 0 { uval(i, k) } else { 0.0 }
    });
}

#[test]
fn observed_with_a_sum_below_its_root() {
    // c = 2·Σ_k u[2,k]²: spliced as is, then renamed apart from the reader's k
    let c: f64 = 2.0 * (0..NZ).map(|k| uval(1, k).powi(2)).sum::<f64>();
    let file = doc_with(neg(mul(json!("c"), idx("u", &["i", "k"]))), ["i", "k"],
                        sdef(None), cvar(),
                        vec![json!({"lhs": "c", "rhs": mul(json!(2), ksum(sq(json!(
                            {"op": "index", "args": ["u", 2, "k"]}))))})]);
    check(&file, |i, k, a, b| {
        -(if (a, b) == (i, k) { c } else { 0.0 })
            - if a == 1 { uval(i, k) * 4.0 * uval(1, b) } else { 0.0 }
    });
}

#[test]
fn hand_written_shadowing_sum() {
    // -u[i,k]·Σ_k u[i,k]²: the same site outside and inside a sum that
    // shadows the reader's k
    let file = doc(neg(mul(idx("u", &["i", "k"]), ksum(sq(idx("u", &["i", "k"]))))),
                   ["i", "k"], sdef(None));
    check(&file, |i, k, a, b| {
        if a != i { 0.0 } else {
            -(if b == k { sval(i) } else { 0.0 }) - uval(i, k) * 2.0 * uval(i, b)
        }
    });
}

#[test]
fn index_named_like_an_observed() {
    // An index an enclosing faq binds is that index, not a read of the
    // observed of the same name.
    let file = doc_with(neg(idx("u", &["i", "k"])), ["i", "k"], sdef(None),
                        json!({"k": {"type": "unknown"}}),
                        vec![json!({"lhs": "k", "rhs": ksum(json!(
                            {"op": "index", "args": ["u", 1, "k"]}))})]);
    check(&file, |i, k, a, b| if (a, b) == (i, k) { -1.0 } else { 0.0 });
}

fn sums(e: &Expr, acc: &mut Vec<Expr>) {
    if let Expr::Operator(n) = e {
        if n.op == "faq" && n.output_idx.as_ref().map_or(true, |o| o.is_empty()) {
            acc.push(e.clone());
        }
        n.map_children(&mut |c: &Expr| {
            sums(c, acc);
            c.clone()
        });
    }
}

#[test]
fn repeated_reads_inline_to_one_tree() {
    // Two reads of one contracting observed stay structurally equal, so
    // template CSE and hoisting still share the sum.
    let s = idx("s", &["i"]);
    let file = doc(json!({"op": "-", "args": [neg(s.clone()), sq(s)]}), ["i", "k"],
                   sdef(None));
    let sv = sysview(&file, None).unwrap();
    let obs = observed_defs(&sv).unwrap();
    let eq = sv.equations.iter().find(|e| lhs_state(e).is_some()).unwrap();
    let mut acc = Vec::new();
    sums(&inline_observed(&eq.rhs, &obs).unwrap(), &mut acc);
    assert_eq!(acc.len(), 2);
    assert_eq!(skey(&acc[0]), skey(&acc[1]));
    let Expr::Operator(n) = &acc[0] else { unreachable!() };
    assert!(!n.ranges.as_ref().unwrap().contains_key("k"),
            "renamed apart from the reader's k");
}
