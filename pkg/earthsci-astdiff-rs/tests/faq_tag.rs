//! The `faq` node tag — the regression guard for a SILENT ZERO (port of the
//! reference's test/faq_tag_test.jl).
//!
//! EarthSciAST renamed the unified query node's wire tag from `aggregate` to
//! `faq` at esm 1.1.0, and esm 2.0.0 removed the old spelling (the loader
//! rejects it, `removed_op`). Matching the old tag threw nothing: `lhs_state` stops
//! recognising the pointwise-lifted `faq{ expr: D(index(u, …)) }` LHS, every
//! array state equation is classified as non-differential, and the band list
//! comes back EMPTY. So these assertions are about non-emptiness as much as
//! correctness.

use std::path::PathBuf;

use earthsci_ast::{Equation, EsmFile, Expr};
use earthsci_astdiff::system::{lhs_state, observed_defs, view_states};
use earthsci_astdiff::{jacobian_bands, sysview, Wrt};
use serde_json::{json, Value};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> EsmFile {
    earthsci_ast::load_path(root().join("tests/valid").join(format!("{name}.esm")))
        .expect("fixture loads")
}

fn expr(v: Value) -> Expr {
    serde_json::from_value(v).expect("expression parses")
}

fn eq(lhs: Value) -> Equation {
    serde_json::from_value(json!({"lhs": lhs, "rhs": 0})).expect("equation parses")
}

#[test]
fn lifted_lhs_is_an_array_state_equation() {
    let lifted = json!({"op": "faq", "args": [], "output_idx": ["i"],
                        "ranges": {"i": [1, 4]},
                        "expr": {"op": "D", "wrt": "t",
                                 "args": [{"op": "index", "args": ["u", "i"]}]}});
    assert!(matches!(expr(lifted.clone()), Expr::Operator(_)));
    assert_eq!(lhs_state(&eq(lifted)).as_deref(), Some("u"));
    let plain = json!({"op": "D", "wrt": "t", "args": ["u"]});
    assert_eq!(lhs_state(&eq(plain)).as_deref(), Some("u"));
}

/// The CI gate: every fixture whose states are arrays differentiates to a
/// non-empty band set. A tag mismatch makes all of these silently empty.
#[test]
fn array_models_differentiate_to_a_non_empty_band_set() {
    for name in [
        "adv_interior",
        "flux_form_adv",
        "minmod_adv",
        "contracted_ops",
        "adv_react_2d",
        "bd_chem",
        "contracting_observed",
        "forcing_bracket",
        "filtered_scan",
        "semiring_reduce",
        "pre_operator",
    ] {
        let file = fixture(name);
        let sv = sysview(&file, None).unwrap();
        assert!(
            !view_states(&sv, &observed_defs(&sv).unwrap()).is_empty(),
            "{name}: no states"
        );
        let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
        assert!(
            !entries.is_empty(),
            "{name}: EMPTY band list (the silent zero)"
        );
        assert!(
            entries.iter().any(|e| !e.band.rows.is_empty()),
            "{name}: no array-valued band"
        );
    }
}

fn alias_json(tag: &str, version: &str) -> serde_json::Value {
    let body = json!({"op": "*", "args": [-0.5, {"op": "index", "args": ["u", "i"]}]});
    json!({
        "esm": version,
        "metadata": {"name": "FaqAlias"},
        "index_sets": {"x": {"kind": "interval", "size": 4}},
        "models": {"FaqAlias": {
            "variables": {"u": {"type": "unknown", "shape": ["x"], "default": 1.0}},
            "equations": [{
                "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                "rhs": {"op": tag, "args": [], "output_idx": ["i"],
                        "ranges": {"i": [1, 4]}, "expr": body}}]}}})
}

fn alias_doc(tag: &str, version: &str) -> EsmFile {
    earthsci_ast::load_document(&alias_json(tag, version)).expect("document loads")
}

#[test]
fn removed_alias_is_refused_at_load() {
    assert!(
        !jacobian_bands(&alias_doc("faq", "1.1.0"), None, Wrt::States)
            .unwrap()
            .is_empty()
    );
    // Even a 1.0.0 document, where `aggregate` was the current spelling: the
    // 2.0.0 loader rejects every removed op whatever `esm` it declares.
    assert!(earthsci_ast::load_document(&alias_json("aggregate", "1.0.0")).is_err());
}
