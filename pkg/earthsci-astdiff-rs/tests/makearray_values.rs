//! `makearray` region values read by position (esm-spec §4.3.2): a value
//! fills its region by position, so the value's element at its own index
//! `origin + p` lands on region index `lo + p`, where `origin` is a faq
//! value's range start (1 for any other array). A value may be FULL-RANK
//! (spanning every region dim, a pinned one included) or REDUCED-RANK (the
//! non-singleton dims only), and may be a faq node, a makearray, or an
//! array variable. Each case is checked against central differences of
//! earthsci-ast's own right-hand side, both through an observed (the
//! inliner's `index`-of-`makearray` lowering) and as the right-hand side
//! itself (the band calculus's `bands_makearray`), so the two paths cannot
//! disagree with the source or with each other.

use std::collections::{BTreeMap, HashMap};

use earthsci_ast::{esm_problem, rhs_program, ProblemOptions};
use earthsci_astdiff::Wrt;
use serde_json::{json, Value};

mod common;
use common::docvals::{doc_jacobian, Env};

fn u2(i: Value, j: Value) -> Value {
    json!({"op": "index", "args": ["u", i, j]})
}

/// `2·u[a, b]` as a faq node over `ranges`.
fn twice(oidx: &[&str], ranges: Value, a: Value, b: Value) -> Value {
    json!({"op": "faq", "args": [], "output_idx": oidx, "ranges": ranges,
           "expr": {"op": "*", "args": [2.0, u2(a, b)]}})
}

const U: [[f64; 2]; 3] = [[0.3, 0.7], [1.1, 1.9], [2.3, 2.9]];

/// `F = makearray(0 over [1,3]×[1,2]; value over region)` on a 3×2 state
/// `u`, read through an observed (`D(u) = -u + F`) or as `D(u) = F`; the
/// band Jacobian must equal central differences of the source.
fn check(label: &str, region: Value, value: Value, extra_vars: Value, extra_eqs: Value) {
    for observed in [true, false] {
        let fdef = json!({"op": "makearray", "args": [],
            "regions": [[[1, 3], [1, 2]], region], "values": [0.0, value]});
        let mut vars = json!({"u": {"type": "unknown", "shape": ["x", "y"], "default": 0.5}});
        let mut eqs = vec![];
        if observed {
            vars["F"] = json!({"type": "unknown", "shape": ["x", "y"]});
            eqs.push(json!({"lhs": "F", "rhs": fdef}));
            eqs.push(json!({"lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                "rhs": {"op": "faq", "args": [], "output_idx": ["i", "j"],
                    "ranges": {"i": [1, 3], "j": [1, 2]},
                    "expr": {"op": "+", "args": [
                        {"op": "-", "args": [u2(json!("i"), json!("j"))]},
                        {"op": "index", "args": ["F", "i", "j"]}]}}}));
        } else {
            eqs.push(json!({"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": fdef}));
        }
        for (k, v) in extra_vars.as_object().unwrap() {
            vars[k] = v.clone();
        }
        eqs.extend(extra_eqs.as_array().unwrap().iter().cloned());
        let d = json!({"esm": "1.1.0", "metadata": {"name": "S"},
            "index_sets": {"x": {"kind": "interval", "size": 3},
                           "y": {"kind": "interval", "size": 2},
                           "one": {"kind": "interval", "size": 1},
                           "two": {"kind": "interval", "size": 2}},
            "models": {"S": {"variables": vars, "equations": eqs}}});
        let file = earthsci_ast::load_document(&d).expect("doc loads");
        let label = format!("{label} (observed: {observed})");

        let src = rhs_program(&esm_problem(&file, (0.0, 1.0), ProblemOptions::default()).unwrap())
            .unwrap();
        let lay = src.layout();
        let ub = lay
            .u
            .iter()
            .find(|b| b.name.ends_with(".u") || b.name == "u")
            .unwrap();
        // Column-major storage of u[i, j].
        let cell = |k: usize| format!("u[{},{}]", k % 3 + 1, k / 3 + 1);
        let (mut y, aux) = src.default_state();
        for k in 0..6 {
            y[ub.base + k] = U[k % 3][k / 3];
        }
        let p: Vec<f64> = src
            .parameters()
            .iter()
            .map(|s| s.default.unwrap_or(0.0))
            .collect();
        let forcing = src.default_forcing();
        let mut ws = src.new_workspace();
        let mut f = |y: &[f64]| {
            let mut du = vec![0.0; y.len()];
            src.eval_into(y, &aux, &p, 0.0, &forcing, &mut du, &mut ws)
                .unwrap();
            du[ub.base..ub.base + 6].to_vec()
        };
        let h = 1e-6;
        let mut fd: BTreeMap<(String, String), f64> = BTreeMap::new();
        for c in 0..6 {
            let (mut yp, mut ym) = (y.clone(), y.clone());
            yp[ub.base + c] += h;
            ym[ub.base + c] -= h;
            let (fp, fm) = (f(&yp), f(&ym));
            for r in 0..6 {
                let d = (fp[r] - fm[r]) / (2.0 * h);
                if d.abs() > 1e-9 {
                    fd.insert((cell(r), cell(c)), d);
                }
            }
        }
        let env = Env {
            states: HashMap::from([("u".to_string(), json!(U))]),
            params: HashMap::new(),
            t: 0.0,
        };
        let jac = doc_jacobian(&file, None, Wrt::States, None, &env)
            .unwrap_or_else(|e| panic!("{label}: {}", e.0));
        for (k, want) in &fd {
            let got = jac.get(k).copied().unwrap_or(0.0);
            assert!((got - want).abs() <= 1e-6, "{label} {k:?}: {got} vs {want}");
        }
        for (k, got) in &jac {
            let want = fd.get(k).copied().unwrap_or(0.0);
            assert!((got - want).abs() <= 1e-6, "{label} {k:?}: {got} vs {want}");
        }
    }
}

#[test]
fn full_rank_faq_value_over_a_shifted_range() {
    // Region row 2, value range i ∈ [1,1] (and [3,3]): row 2 reads u[1,j]
    // (u[3,j]), NOT u[2,j]. The inliner once bound the value's output index
    // to the absolute row index and read u[2,j].
    for r in [1, 2, 3] {
        check(
            &format!("full-rank faq, i ∈ [{r},{r}]"),
            json!([[2, 2], [1, 2]]),
            twice(
                &["i", "j"],
                json!({"i": [r, r], "j": [1, 2]}),
                json!("i"),
                json!("j"),
            ),
            json!({}),
            json!([]),
        );
    }
}

#[test]
fn reduced_rank_faq_value_over_a_shifted_region() {
    // Region rows [2,3] of column 2, value k ∈ [1,2]: row i reads u[i−1,1].
    check(
        "reduced-rank faq",
        json!([[2, 3], [2, 2]]),
        twice(&["k"], json!({"k": [1, 2]}), json!("k"), json!(1)),
        json!({}),
        json!([]),
    );
}

#[test]
fn array_variable_values() {
    // A full-rank observed variable G (shape one×y) over the pinned row 2,
    // and a reduced-rank one H (shape two) over rows [2,3] of column 2. Both
    // were read at the region's free rank alone: G[j] (wrong arity) and H[i]
    // (unshifted, H[3] out of range).
    check(
        "full-rank variable",
        json!([[2, 2], [1, 2]]),
        json!("G"),
        json!({"G": {"type": "unknown", "shape": ["one", "y"]}}),
        json!([{"lhs": "G",
                "rhs": twice(&["i", "j"], json!({"i": [1, 1], "j": [1, 2]}), json!("i"), json!("j"))}]),
    );
    check(
        "reduced-rank variable",
        json!([[2, 3], [2, 2]]),
        json!("H"),
        json!({"H": {"type": "unknown", "shape": ["two"]}}),
        json!([{"lhs": "H",
                "rhs": twice(&["k"], json!({"k": [1, 2]}), json!("k"), json!(1))}]),
    );
}

#[test]
fn full_rank_makearray_value() {
    // A makearray value spanning both region dims over the pinned row 2:
    // its own row 1 lands on row 2. The inliner indexed it at the free rank
    // and panicked on the missing index.
    let inner = json!({"op": "makearray", "args": [], "regions": [[[1, 1], [1, 2]]],
        "values": [twice(&["j"], json!({"j": [1, 2]}), json!(1), json!("j"))]});
    check(
        "full-rank makearray",
        json!([[2, 2], [1, 2]]),
        inner,
        json!({}),
        json!([]),
    );
}
