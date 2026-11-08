//! The extraction rule of esm-jacobian-spec §4.1 (the reference's
//! cse_test.jl "the extraction rule" testset).
use earthsci_ast::{Expr, ExpressionNode};
use earthsci_astdiff::{cse_templates, skey};
use serde_json::{json, Value};

fn ex(v: Value) -> Expr {
    serde_json::from_value(v).unwrap()
}

fn rf(n: &str) -> Value {
    json!({"op": "apply_expression_template", "args": [], "name": n})
}

#[test]
fn counts_after_larger_extractions() {
    let b = json!({"op": "exp", "args": [{"op": "+", "args": ["x", "y"]}]});
    let c = json!({"op": "log", "args": [{"op": "*", "args": ["x", 2.0]}]});
    let a = json!({"op": "+", "args": [{"op": "*", "args": [b, b]}, c]});
    let e1 = ex(json!({"op": "-", "args": [a, "a"]}));
    let e2 = ex(json!({"op": "/", "args": [a, "b"]}));
    let (tmpl, rw) = cse_templates(&[e1, e2], 4, "jt");
    // A occurs twice; B occurs twice in A's body, which is kept once, so it
    // still counts 2; C occurs once in A's body, so it counts 1 and stays.
    let names: Vec<&str> = tmpl.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["jt1", "jt2"]);
    let want1 = json!({"op": "+", "args": [{"op": "*", "args": [rf("jt2"), rf("jt2")]}, c]});
    assert_eq!(skey(&tmpl[0].1), skey(&ex(want1)));
    assert_eq!(skey(&tmpl[1].1), skey(&ex(b)));
    assert_eq!(
        skey(&rw[0]),
        skey(&ex(json!({"op": "-", "args": [rf("jt1"), "a"]})))
    );
    assert_eq!(
        skey(&rw[1]),
        skey(&ex(json!({"op": "/", "args": [rf("jt1"), "b"]})))
    );
}

#[test]
fn equal_sizes_are_named_in_first_encounter_order() {
    let d = json!({"op": "sin", "args": [{"op": "+", "args": ["x", "z"]}]});
    let e = json!({"op": "cos", "args": [{"op": "+", "args": ["y", "z"]}]});
    let (tmpl, _) = cse_templates(
        &[
            ex(json!({"op": "*", "args": [e, d]})),
            ex(json!({"op": "-", "args": [d, e]})),
        ],
        4,
        "jt",
    );
    assert_eq!(tmpl.len(), 2);
    assert_eq!(skey(&tmpl[0].1), skey(&ex(e)));
    assert_eq!(skey(&tmpl[1].1), skey(&ex(d)));
}

fn node(op: &str, args: Vec<Expr>) -> Expr {
    Expr::operator(ExpressionNode {
        op: op.into(),
        args,
        ..Default::default()
    })
}

#[test]
fn an_integer_leaf_is_not_a_float_leaf() {
    let x = || Expr::Variable("x".into());
    let p = node("exp", vec![node("+", vec![x(), Expr::Integer(1)])]);
    let q = node("exp", vec![node("+", vec![x(), Expr::Number(1.0)])]);
    let (tmpl, _) = cse_templates(&[node("*", vec![p, q])], 4, "jt");
    assert!(tmpl.is_empty());
}

#[test]
fn map_slots_are_visited_by_ascending_key() {
    // `axes` written "b" then "a": the spec numbers the "a" entry first
    // (child slots, §4.1), so equal-size E (under "a") is jt1.
    let d = json!({"op": "sin", "args": [{"op": "+", "args": ["x", "z"]}]});
    let e = json!({"op": "cos", "args": [{"op": "+", "args": ["y", "z"]}]});
    let tl = json!({"op": "table_lookup", "args": [], "table": "t",
                    "axes": {"b": d, "a": e}});
    let (tmpl, _) = cse_templates(&[ex(tl), ex(json!({"op": "*", "args": [d, e]}))], 4, "jt");
    assert_eq!(tmpl.len(), 2);
    assert_eq!(skey(&tmpl[0].1), skey(&ex(e)));
    assert_eq!(skey(&tmpl[1].1), skey(&ex(d)));
}

#[test]
fn an_operator_with_no_children_is_extracted() {
    let z = json!({"op": "zarg", "args": []});
    let (tmpl, rw) = cse_templates(&[ex(json!({"op": "*", "args": [z, z]}))], 1, "jt");
    assert_eq!(tmpl.len(), 1);
    assert_eq!(skey(&tmpl[0].1), skey(&ex(z)));
    assert_eq!(
        skey(&rw[0]),
        skey(&ex(json!({"op": "*", "args": [rf("jt1"), rf("jt1")]})))
    );
}

#[test]
fn a_deep_chain_needs_no_stack_per_level() {
    let mut e = Expr::Variable("x".into());
    for _ in 0..200_000 {
        e = node("+", vec![e, Expr::Integer(1)]);
    }
    let (tmpl, rw) = cse_templates(&[e.clone(), e.clone()], 12, "jt");
    assert_eq!(tmpl.len(), 1);
    assert!(matches!(&rw[0], Expr::Operator(a) if a.op == "apply_expression_template"));
    // the body is the chain itself, not a copy
    assert!(
        matches!((&tmpl[0].1, &e), (Expr::Operator(a), Expr::Operator(b)) if std::sync::Arc::ptr_eq(a, b))
    );
    // dropping a 200k-deep chain recursively would overflow; leak it
    std::mem::forget((tmpl, rw, e));
}
