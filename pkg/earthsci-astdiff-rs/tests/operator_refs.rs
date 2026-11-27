//! Operator references are differentiated without being expanded
//! (operator_refs.rs; the reference's operator_refs_test.jl). The flattened
//! system is EarthSciAST's Rust `flatten`, which keeps references and closes
//! the merged registry; it is checked against the reference binding's
//! (`coupled_chem_advection.flat-refs.json`: its equations and merged
//! registry, recorded from Julia), and the bands, structure and evaluation
//! document of both inputs must match the reference's goldens byte for byte.

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

/// This binding's `flatten`, references kept.
fn flat_real() -> FlattenedSystem {
    earthsci_ast::flatten(&load(NAME)).unwrap()
}

/// This binding's flattened system with the reference binding's recorded
/// equations and merged registry laid over it.
fn flat_recorded() -> FlattenedSystem {
    let mut flat = flat_real();
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

fn inputs() -> [(FlattenedSystem, &'static str); 2] {
    [(flat_real(), "flatten"), (flat_recorded(), "recorded")]
}

/// The Rust `flatten` keeps the references the Julia one keeps, with the
/// same closed merged registry.
#[test]
fn flatten_matches_the_reference() {
    let (real, rec) = (flat_real(), flat_recorded());
    assert!(count_refs(&serde_json::to_value(&real.equations).unwrap()) > 0);
    assert_eq!(
        stable_json(&serde_json::to_value(&real.equations).unwrap()),
        stable_json(&serde_json::to_value(&rec.equations).unwrap()),
        "equations differ"
    );
    assert_eq!(
        stable_json(&serde_json::to_value(&real.template_registry).unwrap()),
        stable_json(&serde_json::to_value(&rec.template_registry).unwrap()),
        "merged registries differ"
    );
}

#[test]
fn bands_match_the_reference() {
    for (flat, input) in inputs() {
        let sv = sysview_flat(&flat).unwrap();
        for (wrt, tag) in [(Wrt::States, ""), (Wrt::Parameters, ".parameters")] {
            let entries = jacobian_bands_view(&sv, wrt, None).unwrap();
            let ser: Vec<Value> = entries.iter().map(serialize_band).collect();
            assert!(
                count_refs(&Value::Array(ser.clone())) > 0,
                "{input} {wrt:?}: no references kept"
            );
            assert_eq!(
                stable_json(&Value::Array(ser)),
                golden(&format!("{NAME}{tag}.bands.json")),
                "{input} {wrt:?} diverges"
            );
        }
    }
}

#[test]
fn structure_matches_the_reference() {
    for (flat, input) in inputs() {
        let sv = sysview_flat(&flat).unwrap();
        let entries = jacobian_structure_view(&sv, Wrt::States, None).unwrap();
        let text = stable_json(&Value::Array(
            entries.iter().map(serialize_structure_band).collect(),
        ));
        assert_eq!(
            text,
            golden(&format!("{NAME}.structure.bands.json")),
            "{input}"
        );
    }
}

#[test]
fn evaluation_document_carries_definitions() {
    for (flat, input) in inputs() {
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
        assert_eq!(
            stable_json(&doc),
            golden(&format!("{NAME}.evaldoc.json")),
            "{input}"
        );
    }
}

/// One stencil definition used at two binding shapes, with a structural
/// parameter as a range bound (`[1, "n"]`) and an expression argument
/// (spec §3.6): each shape has its own derived band definitions, named with
/// the shape, and the bands and evaluation document match the reference's
/// goldens byte for byte.
#[test]
fn one_stencil_at_two_binding_shapes() {
    let name = "two_shape_stencil";
    let flat = earthsci_ast::flatten(&load(name)).unwrap();
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let ser: Vec<Value> = entries.iter().map(serialize_band).collect();
    assert_eq!(
        stable_json(&Value::Array(ser)),
        golden(&format!("{name}.bands.json"))
    );
    let (doc, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
    let ops = doc["models"]["JacobianEval"]["operators"]
        .as_object()
        .unwrap();
    assert!(ops.keys().any(|k| k.ends_with(".d_f.b1@6,6")));
    assert!(ops.keys().any(|k| k.ends_with(".d_f.b1@2,2")));
    assert_eq!(stable_json(&doc), golden(&format!("{name}.evaldoc.json")));
}

/// A scalar definition read through its array parameters (directly, under a
/// sum in its body, and through a nested definition), at uses whose bindings
/// read the row index, once with an expression argument (spec §3.6): the
/// bands and evaluation document match the reference's goldens.
#[test]
fn scalar_definition_through_array_parameters() {
    let name = "ref_array_param";
    let flat = earthsci_ast::flatten(&load(name)).unwrap();
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let ser: Vec<Value> = entries.iter().map(serialize_band).collect();
    assert_eq!(
        stable_json(&Value::Array(ser)),
        golden(&format!("{name}.bands.json"))
    );
    let (doc, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
    let ops = doc["models"]["JacobianEval"]["operators"]
        .as_object()
        .unwrap();
    assert!(ops.keys().any(|k| k.starts_with("pair.d_w.s1@")));
    assert!(ops.keys().any(|k| k.starts_with("inner.d_w.s1@")));
    assert_eq!(stable_json(&doc), golden(&format!("{name}.evaldoc.json")));
}
