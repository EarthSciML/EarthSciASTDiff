//! A CONTRACTING observed (issue #6; port of the reference's
//! test/contracting_observed_test.jl): `s[i] = Σ_k u[i,k]²` read inside a faq
//! that binds `k` too. Inlined as the bare body, the sum is dropped and the
//! leftover `k` is captured — the column sum collapses to the local cell,
//! with no error. The read must keep the sum, and the band calculus must
//! follow the contraction through it.

use std::collections::HashMap;

use earthsci_astdiff::numeric::Env;
use earthsci_astdiff::{assemble_jacobian, jacobian_bands, Wrt};
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
    let mut ranges = serde_json::Map::new();
    ranges.insert(oidx[0].into(), json!([1, NX]));
    ranges.insert(oidx[1].into(), json!([1, NZ]));
    let d = json!({
        "esm": "1.1.0",
        "metadata": {"name": "ContractObs"},
        "index_sets": {"x": {"kind": "interval", "size": NX},
                       "z": {"kind": "interval", "size": NZ}},
        "models": {"ContractObs": {
            "variables": {
                "u": {"type": "unknown", "shape": ["x", "z"], "default": 0.5},
                "s": {"type": "unknown", "shape": ["x"]}},
            "equations": [
                {"lhs": "s", "rhs": s},
                {"lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                 "rhs": {"op": "faq", "args": [], "output_idx": oidx,
                         "ranges": ranges, "expr": body}}]}}});
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
