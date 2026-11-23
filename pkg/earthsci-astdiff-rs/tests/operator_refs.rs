//! Operator references are differentiated without being expanded
//! (operator_refs.rs; the reference's operator_refs_test.jl). EarthSciAST's
//! Rust `load` still expands every reference, so the reference-preserving
//! flattened system comes from the reference binding's `flatten`
//! (`coupled_chem_advection.flat-refs.json`: its equations and merged
//! registry) laid over this binding's flattened system, whose variables and
//! index sets are the same. Its bands, structure and evaluation document must
//! match the reference's goldens byte for byte.

use std::fs;

use earthsci_ast::{Equation, FlattenedSystem};
use earthsci_astdiff::{
    evaluation_document, jacobian_bands_view, jacobian_structure_view, serialize_band,
    serialize_structure_band, stable_json, sysview_flat, Wrt,
};
use serde_json::Value;

mod common;
use common::{load, root};

const NAME: &str = "coupled_chem_advection";

fn golden(file: &str) -> String {
    fs::read_to_string(root().join("tests/goldens").join(file)).unwrap()
}

fn flat_refs() -> FlattenedSystem {
    let mut flat = earthsci_ast::flatten(&load(NAME)).unwrap();
    let refs: Value = serde_json::from_str(&golden(&format!("{NAME}.flat-refs.json"))).unwrap();
    flat.equations = refs["equations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| serde_json::from_value::<Equation>(e.clone()).unwrap())
        .collect();
    flat.template_registry = refs["template_registry"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    flat
}

fn count_refs(x: &Value) -> usize {
    match x {
        Value::Object(m) => {
            let own =
                matches!(m.get("op"), Some(Value::String(o)) if o == "apply_operator") as usize;
            own + m.values().map(count_refs).sum::<usize>()
        }
        Value::Array(a) => a.iter().map(count_refs).sum(),
        _ => 0,
    }
}

#[test]
fn bands_match_the_reference() {
    let flat = flat_refs();
    let sv = sysview_flat(&flat).unwrap();
    for (wrt, tag) in [(Wrt::States, ""), (Wrt::Parameters, ".parameters")] {
        let entries = jacobian_bands_view(&sv, wrt, None).unwrap();
        let ser: Vec<Value> = entries.iter().map(serialize_band).collect();
        assert!(
            count_refs(&Value::Array(ser.clone())) > 0,
            "{wrt:?}: no references kept"
        );
        assert_eq!(
            stable_json(&Value::Array(ser)),
            golden(&format!("{NAME}{tag}.bands.json")),
            "{wrt:?} diverges"
        );
    }
}

#[test]
fn structure_matches_the_reference() {
    let flat = flat_refs();
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_structure_view(&sv, Wrt::States, None).unwrap();
    let text = stable_json(&Value::Array(
        entries.iter().map(serialize_structure_band).collect(),
    ));
    assert_eq!(text, golden(&format!("{NAME}.structure.bands.json")));
}

#[test]
fn evaluation_document_carries_definitions() {
    let flat = flat_refs();
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
    let jm = &doc["models"]["JacobianEval"];
    let ops = jm["operators"].as_object().expect("an operators block");
    assert!(ops.keys().any(|k| k.contains(".d_f.b")));
    assert!(ops
        .iter()
        .filter(|(k, _)| k.contains(".d_"))
        .all(|(_, e)| e["reference_only"] == Value::Bool(true)));
    assert_eq!(stable_json(&doc), golden(&format!("{NAME}.evaldoc.json")));
}
