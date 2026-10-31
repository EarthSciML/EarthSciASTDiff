//! `sysview_flat` (the reference's `sysview(flat::FlattenedSystem)`) and
//! `jacobian_bands_view`: a flattened system viewed directly gives the same
//! view, context, bands and evaluation document as the route a caller had
//! to take before, which wraps the flattened system back up as a one-model
//! document and views that model with `sysview`.

use earthsci_ast::{EsmFile, FlattenedSystem};
use earthsci_astdiff::{
    ctx_of, evaluation_document, jacobian_bands, jacobian_bands_view, serialize_band, stable_json,
    sysview, sysview_flat, JacEntry, SysView, Wrt,
};
use indexmap::IndexMap;
use serde_json::{json, Value};

mod common;
use common::load;

/// Every shared fixture: the single-model ones and the coupled document
/// (`coupled_chem_advection`, which both routes refuse alike today: the Rust
/// `flatten` leaves its undiscretized `grad` lift as a `faq` the band
/// calculus has no rule for).
const NAMES: [&str; 9] = [
    "bd_chem",
    "adv_interior",
    "contracted_ops",
    "flux_form_adv",
    "minmod_adv",
    "contracting_observed",
    "tgrad_photolysis",
    "forcing_bracket",
    "coupled_chem_advection",
];
const AXES: [Wrt; 4] = [Wrt::States, Wrt::Parameters, Wrt::Forcing, Wrt::Time];
const MODEL: &str = "flattened";

/// The union of a flattened system's variable maps, merged as the reference
/// merges them (states, then parameters, then observed; a repeated name
/// keeps its first position and takes the last declaration).
fn merged(flat: &FlattenedSystem) -> IndexMap<String, earthsci_ast::ModelVariable> {
    let mut vars = IndexMap::new();
    for (n, v) in flat
        .state_variables
        .iter()
        .chain(&flat.parameters)
        .chain(&flat.observed_variables)
    {
        vars.insert(n.clone(), v.clone());
    }
    vars
}

/// The existing route: the flattened system as a one-model document.
fn wrapped(flat: &FlattenedSystem) -> EsmFile {
    let doc = json!({
        "esm": "1.1.0",
        "metadata": {"name": MODEL},
        "index_sets": serde_json::to_value(&flat.index_sets).unwrap(),
        "models": {MODEL: {
            "variables": serde_json::to_value(merged(flat)).unwrap(),
            "equations": serde_json::to_value(&flat.equations).unwrap(),
        }},
    });
    serde_json::from_value(doc).expect("wrapped document deserializes")
}

fn flat_of(name: &str) -> FlattenedSystem {
    earthsci_ast::flatten(&load(name)).unwrap_or_else(|e| panic!("{name} flattens: {e}"))
}

fn ser(entries: &[JacEntry]) -> String {
    stable_json(&Value::Array(entries.iter().map(serialize_band).collect()))
}

fn view_json(sv: &SysView) -> Value {
    json!({
        "variables": serde_json::to_value(&*sv.variables).unwrap(),
        "equations": serde_json::to_value(sv.equations).unwrap(),
        "index_sets": serde_json::to_value(&sv.index_sets).unwrap(),
    })
}

#[test]
fn view_matches_the_wrapped_document() {
    for name in NAMES {
        let flat = flat_of(name);
        let file = wrapped(&flat);
        let direct = sysview_flat(&flat);
        let via = sysview(&file, Some(MODEL)).unwrap();
        // Order-sensitive: the variable maps compare as serialized objects
        // in their own order, not as sets.
        assert_eq!(
            stable_json(&view_json(&direct)),
            stable_json(&view_json(&via)),
            "{name}: views differ"
        );
        let names_direct: Vec<&String> = direct.variables.keys().collect();
        let names_via: Vec<&String> = via.variables.keys().collect();
        assert_eq!(names_direct, names_via, "{name}: variable order differs");
        let (cd, cv) = (ctx_of(&direct).unwrap(), ctx_of(&via).unwrap());
        assert_eq!(
            cd.index_sets, cv.index_sets,
            "{name}: index-set sizes differ"
        );
        assert_eq!(cd.shapes, cv.shapes, "{name}: shapes differ");
        assert_eq!(cd.params, cv.params, "{name}: parameters differ");
    }
}

#[test]
fn bands_match_the_wrapped_document_on_every_axis() {
    let mut nonempty = 0;
    for name in NAMES {
        let flat = flat_of(name);
        let file = wrapped(&flat);
        let sv = sysview_flat(&flat);
        for wrt in AXES {
            let direct = jacobian_bands_view(&sv, wrt, None);
            let via = jacobian_bands(&file, Some(MODEL), wrt);
            match (direct, via) {
                (Ok(d), Ok(v)) => {
                    assert_eq!(ser(&d), ser(&v), "{name} {wrt:?}: bands differ");
                    nonempty += usize::from(!d.is_empty());
                }
                (Err(d), Err(v)) => assert_eq!(d.0, v.0, "{name} {wrt:?}: errors differ"),
                (d, v) => panic!(
                    "{name} {wrt:?}: one route failed: direct {:?}, wrapped {:?}",
                    d.err(),
                    v.err()
                ),
            }
        }
    }
    assert!(
        nonempty >= NAMES.len(),
        "the comparison must see real bands"
    );
}

#[test]
fn evaluation_document_matches_the_wrapped_document() {
    for name in NAMES {
        let flat = flat_of(name);
        let file = wrapped(&flat);
        let direct = sysview_flat(&flat);
        let via = sysview(&file, Some(MODEL)).unwrap();
        let Ok(entries) = jacobian_bands_view(&direct, Wrt::States, None) else {
            continue; // refused alike by both routes (checked above)
        };
        let (dd, fd) = evaluation_document(&direct, &entries, false, 0).unwrap();
        let (dv, fv) = evaluation_document(&via, &entries, false, 0).unwrap();
        assert_eq!(fd, fv, "{name}: band fields differ");
        assert_eq!(
            stable_json(&dd),
            stable_json(&dv),
            "{name}: documents differ"
        );
    }
}

/// Two models coupled by a `variable_map` (a source model's state feeds the
/// chemistry's rate parameter), as `flatten` joins them: the state Jacobian
/// couples the two models' states through the mapped name, and the direct
/// view gives the wrapped document's bands.
#[test]
fn two_coupled_models() {
    let file = earthsci_ast::load_document(&json!({
        "esm": "1.1.0",
        "metadata": {"name": "TwoModels"},
        "models": {
            "Chem": {
                "variables": {
                    "c": {"type": "unknown", "default": 1.0},
                    "k": {"type": "parameter", "default": 0.5}},
                "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [
                                   {"op": "-", "args": ["k"]}, "c", "c"]}}]},
            "Source": {
                "variables": {
                    "s": {"type": "unknown", "default": 2.0},
                    "r": {"type": "parameter", "default": 0.1}},
                "equations": [{"lhs": {"op": "D", "args": ["s"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [
                                   {"op": "-", "args": ["r"]}, "s"]}}]}},
        "coupling": [{"type": "variable_map", "from": "Source.s", "to": "Chem.k",
                      "transform": "param_to_var"}]
    }))
    .expect("document loads");
    let flat = earthsci_ast::flatten(&file).expect("document flattens");
    let entries = jacobian_bands_view(&sysview_flat(&flat), Wrt::States, None).unwrap();
    let pairs: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.u.as_str(), e.v.as_str()))
        .collect();
    assert!(pairs.contains(&("Chem.c", "Chem.c")), "{pairs:?}");
    assert!(pairs.contains(&("Chem.c", "Source.s")), "{pairs:?}");
    assert!(pairs.contains(&("Source.s", "Source.s")), "{pairs:?}");
    let via = jacobian_bands(&wrapped(&flat), Some(MODEL), Wrt::States).unwrap();
    assert_eq!(ser(&entries), ser(&via));
}

/// A name in more than one of the flattened system's maps keeps the
/// position of its first map and the declaration of its last, as the
/// reference (a `Dict` filled states, parameters, observed) and the Python
/// binding (`dict.update` in that order) merge them.
#[test]
fn repeated_names_take_the_last_declaration() {
    let mut flat = flat_of("bd_chem");
    let (name, mut var) = flat
        .state_variables
        .iter()
        .next()
        .map(|(n, v)| (n.clone(), v.clone()))
        .expect("bd_chem has a state");
    var.description = Some("from observed_variables".into());
    flat.observed_variables.insert(name.clone(), var);
    let sv = sysview_flat(&flat);
    assert_eq!(sv.variables.get_index_of(&name), Some(0));
    assert_eq!(
        sv.variables[&name].description.as_deref(),
        Some("from observed_variables")
    );
    assert_eq!(
        sv.variables.len(),
        flat.state_variables.len() + flat.parameters.len() + flat.observed_variables.len() - 1
    );
}
