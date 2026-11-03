//! Data-fed parameters in the evaluation document (spec §8.1; the
//! reference's data_fed_test.jl): a parameter whose `update` has a `data` rule
//! keeps it, and the document carries the `data_sources` entries those rules
//! name, so EarthSciAST builds the parameter as a forcing slot of
//! `rhs_program`, not a constant. Byte-identical to the reference's
//! `data_fed_flat.{evaldoc,bands}.json`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::sync::Arc;

use earthsci_ast::{
    esm_problem, rhs_program, EsmFile, ForcingSet, ProblemOptions, RhsProgram, SlotKind,
};
use earthsci_astdiff::{
    evaluation_document, jacobian_bands_view, jvp_document, serialize_band, stable_json,
    sysview_flat, ProductOptions, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

const NAME: &str = "data_fed_flat";
const FED: [&str; 3] = ["Box.P", "Chem.E", "Chem.T"];

fn golden(tag: &str) -> String {
    fs::read_to_string(
        root()
            .join("tests/goldens")
            .join(format!("{NAME}.{tag}.json")),
    )
    .unwrap()
}

/// The fixture without its data rules and `data_sources`: the same model
/// with every data-fed parameter a constant at its default.
fn without_rules() -> EsmFile {
    let path = root().join("tests/valid").join(format!("{NAME}.esm"));
    let mut d: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    d.as_object_mut().unwrap().remove("data_sources");
    for m in d["models"].as_object_mut().unwrap().values_mut() {
        for v in m["variables"].as_object_mut().unwrap().values_mut() {
            v.as_object_mut().unwrap().remove("update");
        }
    }
    earthsci_ast::load_document(&d).unwrap()
}

fn program(doc: &Value) -> RhsProgram {
    let file = earthsci_ast::load_document(doc).expect("the evaluation document loads");
    rhs_program(&esm_problem(&file, (0.0, 1.0), ProblemOptions::default()).unwrap()).unwrap()
}

/// One right-hand-side call at a fixed state with forcing `fs`, `du` by
/// state block.
fn rhs(prog: &RhsProgram, fs: &ForcingSet) -> BTreeMap<String, Vec<f64>> {
    let (mut u, aux) = prog.default_state();
    for b in prog.state_layout() {
        let n = b.shape.iter().product::<usize>().max(1);
        for x in &mut u[b.flat_offset..b.flat_offset + n] {
            *x = 1.0 + b.name.len() as f64;
        }
    }
    let p = prog.parameter_vector(&HashMap::new()).unwrap();
    let mut du = vec![f64::NAN; u.len()];
    let mut ws = prog.new_workspace();
    prog.eval_into(&u, &aux, &p, 0.0, fs, &mut du, &mut ws)
        .unwrap();
    prog.state_layout()
        .iter()
        .map(|b| {
            let n = b.shape.iter().product::<usize>().max(1);
            (
                b.name.clone(),
                du[b.flat_offset..b.flat_offset + n].to_vec(),
            )
        })
        .collect()
}

#[test]
fn evaluation_document_carries_the_data_rules() {
    let file = load(NAME);
    let flat = earthsci_ast::flatten(&file).unwrap();
    let sv = sysview_flat(&flat)
        .unwrap()
        .with_data_sources(file.data_sources.as_ref());
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    for cse in [true, false] {
        let (doc, _) = evaluation_document(&sv, &entries, cse, 12).unwrap();
        assert_eq!(stable_json(&doc), golden("evaldoc"), "cse = {cse}");
    }
    let (doc, _) = evaluation_document(&sv, &entries, false, 0).unwrap();
    assert_eq!(
        doc["models"]["Chem"]["variables"]["T"]["update"],
        json!({"kind": "data", "source": "MET", "from": {"file_variable": "T2M"}})
    );
    assert_eq!(
        doc["models"]["Chem"]["variables"]["E"]["update"]["from"],
        json!({"file_variable": "NOX", "unit_conversion": 0.001})
    );
    assert!(doc["models"]["Chem"]["variables"]["k"]
        .get("update")
        .is_none());
    let keys: Vec<&String> = doc["data_sources"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["EMIS", "MET"]);
    // a flattened system carries no `data_sources`: without them the
    // document is refused, never written with a parameter baked in
    let bare = sysview_flat(&flat).unwrap();
    let err = evaluation_document(&bare, &entries, false, 0).unwrap_err();
    assert!(err.0.contains("data source `EMIS`"), "{}", err.0);
    assert!(jvp_document(&bare, &entries, Wrt::States, &ProductOptions::default()).is_err());
}

#[test]
fn data_fed_parameters_are_constants_of_the_state_jacobian() {
    for file in [load(NAME), without_rules()] {
        let flat = earthsci_ast::flatten(&file).unwrap();
        let entries =
            jacobian_bands_view(&sysview_flat(&flat).unwrap(), Wrt::States, None).unwrap();
        let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
        assert_eq!(text, golden("bands"));
    }
}

#[test]
fn data_fed_parameters_are_forcing_slots() {
    let file = load(NAME);
    let flat = earthsci_ast::flatten(&file).unwrap();
    let sv = sysview_flat(&flat)
        .unwrap()
        .with_data_sources(file.data_sources.as_ref());
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, _) = evaluation_document(&sv, &entries, false, 0).unwrap();
    let prog = program(&doc);
    let slots: Vec<(&str, &[usize], SlotKind)> = prog
        .forcing_slots()
        .iter()
        .map(|s| (s.name.as_str(), s.shape.as_slice(), s.kind))
        .collect();
    let want: Vec<(&str, &[usize], SlotKind)> = FED
        .iter()
        .map(|n| (*n, &[3usize][..], SlotKind::Forcing))
        .collect();
    assert_eq!(slots, want);
    let params: Vec<&str> = prog.parameters().iter().map(|s| s.name.as_str()).collect();
    assert_eq!(params, ["Chem.k"]);
    // Before: the same parameters were constants of the build, no slot.
    let old_file = without_rules();
    let old_flat = earthsci_ast::flatten(&old_file).unwrap();
    let old_sv = sysview_flat(&old_flat).unwrap();
    let (old, _) = evaluation_document(&old_sv, &entries, false, 0).unwrap();
    let oprog = program(&old);
    assert!(oprog.forcing_slots().is_empty());
    // At the default forcing the band fields are those of the document with
    // the parameters baked in; a new forcing set moves exactly the fields
    // that read it: `Chem.T` the chemistry band, `Box.P` the box band, the
    // additive `Chem.E` none.
    let d = prog.default_forcing();
    let base = rhs(&prog, &d);
    assert_eq!(base, rhs(&oprog, &oprog.default_forcing()));
    let changes = |name: &str| -> Vec<String> {
        let k = FED.iter().position(|n| *n == name).unwrap();
        let buf: Arc<[f64]> = d.buffer(k).iter().map(|x| 2.0 * x).collect();
        let moved = rhs(&prog, &d.replace(k, buf).unwrap());
        moved
            .into_iter()
            .filter(|(n, v)| *v != base[n])
            .map(|(n, _)| n)
            .collect()
    };
    assert_eq!(changes("Chem.T"), ["JacobianEval.J_1"]);
    assert_eq!(changes("Box.P"), ["JacobianEval.J_2"]);
    assert!(changes("Chem.E").is_empty());
}
