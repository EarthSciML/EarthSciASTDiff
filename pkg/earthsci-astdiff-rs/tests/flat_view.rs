//! `sysview_flat` (the reference's `sysview(flat::FlattenedSystem)`) and
//! `jacobian_bands_view`: a flattened system viewed directly gives the same
//! view, context, bands and evaluation document as the route a caller had
//! to take before, which wraps the flattened system back up as a one-model
//! document and views that model with `sysview`.

use std::borrow::Cow;

use earthsci_ast::{EsmFile, FlattenedSystem, ModelVariable};
use earthsci_astdiff::{
    ctx_of, evaluation_document, jacobian_bands, jacobian_bands_view, serialize_band, stable_json,
    sysview, sysview_flat, JacEntry, SysView, Wrt,
};
use indexmap::IndexMap;
use serde_json::{json, Value};

mod common;
use common::load;

/// Every shared fixture: the single-model ones and the coupled document
/// (`coupled_chem_advection`, whose chemistry is pointwise-lifted onto the
/// advection grid).
const NAMES: [&str; 11] = [
    "bd_chem",
    "adv_interior",
    "contracted_ops",
    "flux_form_adv",
    "minmod_adv",
    "contracting_observed",
    "tgrad_photolysis",
    "forcing_bracket",
    "coupled_chem_advection",
    "recurrence_sweep",
    "pbl_rank1",
];
const AXES: [Wrt; 4] = [Wrt::States, Wrt::Parameters, Wrt::Forcing, Wrt::Time];
const MODEL: &str = "flattened";

/// The union of a flattened system's variable maps, stated independently
/// of `sysview_flat`'s merge loop: every name in the order of the first map
/// (states, parameters, observed) that holds it, declared as the last map
/// that holds it declares it.
fn merged(flat: &FlattenedSystem) -> IndexMap<String, ModelVariable> {
    let maps = [
        &flat.state_variables,
        &flat.parameters,
        &flat.observed_variables,
    ];
    let mut names: Vec<&String> = Vec::new();
    for m in maps {
        for n in m.keys() {
            if !names.contains(&n) {
                names.push(n);
            }
        }
    }
    names
        .into_iter()
        .map(|n| {
            let v = maps.iter().rev().find_map(|m| m.get(n)).unwrap();
            (n.clone(), v.clone())
        })
        .collect()
}

/// The existing route: the flattened system as a one-model document, each
/// pointwise-lifted state written with its grid shape (every fixture's grid
/// extents each name exactly one index set).
fn wrapped(flat: &FlattenedSystem) -> EsmFile {
    let shapes: Vec<(String, Vec<String>)> = flat
        .lifted_shapes
        .iter()
        .map(|(n, extents)| {
            let axes = extents
                .iter()
                .map(|e| {
                    let named: Vec<&String> = flat
                        .index_sets
                        .iter()
                        .filter(|(_, s)| s.size == Some(*e))
                        .map(|(k, _)| k)
                        .collect();
                    assert_eq!(named.len(), 1, "{n}: extent {e} names one index set");
                    named[0].clone()
                })
                .collect();
            (n.clone(), axes)
        })
        .collect();
    wrapped_with(flat, &shapes, json!({}))
}

/// The flattened system as a one-model document with the given variables'
/// shapes replaced and `extra` index sets added.
fn wrapped_with(flat: &FlattenedSystem, shapes: &[(String, Vec<String>)], extra: Value) -> EsmFile {
    let mut vars = merged(flat);
    for (n, sh) in shapes {
        vars[n].shape = Some(sh.clone());
    }
    let mut isets = serde_json::to_value(&flat.index_sets).unwrap();
    for (k, v) in extra.as_object().unwrap() {
        isets[k] = v.clone();
    }
    let mut doc = json!({
        "esm": "1.1.0",
        "metadata": {"name": MODEL},
        "index_sets": isets,
        "models": {MODEL: {
            "variables": serde_json::to_value(vars).unwrap(),
            "equations": serde_json::to_value(&flat.equations).unwrap(),
        }},
    });
    // a document with no domain has no `domain` key (the schema has no null)
    if let Some(d) = &flat.domain {
        doc["domain"] = serde_json::to_value(d).unwrap();
    }
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
        "variables": serde_json::to_value(&sv.variables).unwrap(),
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
    // The lifted coupled document differentiates: its chemistry rows are
    // read as the grid arrays they are.
    let flat = flat_of("coupled_chem_advection");
    let coupled = jacobian_bands_view(&sysview_flat(&flat), Wrt::States, None).unwrap();
    assert!(coupled.iter().any(|e| e.u == "SimpleOzone.O3"));
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

/// Chemistry lifted onto an advection grid (esm-spec §10.5): `Chem.O3` is
/// declared scalar. `index_sets` is the document's.
///
/// The view's fallback for a `flatten` that records only the grid extents
/// (`lifted_shapes`, EarthSciAST before 0.8): from 0.8, `flatten` records the
/// lifted state's index sets as its shape (esm-libraries-spec §4.7.5), and
/// the view takes that as it stands. The flattened system is returned as an
/// older `flatten` leaves it, the recorded shape and any synthetic index set
/// removed, so the fallback is tested on every EarthSciAST; the shape a newer
/// `flatten` recorded is returned beside it, to check the fallback against.
fn lifted(index_sets: Value) -> (FlattenedSystem, Option<Vec<String>>) {
    let file = earthsci_ast::load_document(&json!({
        "esm": "1.1.0",
        "metadata": {"name": "Lifted"},
        "index_sets": index_sets,
        "models": {
            "Chem": {
                "variables": {
                    "O3": {"type": "unknown", "default": 1.0},
                    "k1": {"type": "parameter", "default": 0.1}},
                "equations": [{"lhs": {"op": "D", "args": ["O3"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [
                                   {"op": "-", "args": ["k1"]}, "O3", "O3"]}}]},
            "Adv": {
                "variables": {
                    "u": {"type": "parameter", "default": 2.0},
                    "dx": {"type": "parameter", "default": 100.0}},
                "equations": [{"lhs": {"op": "D", "args": ["_var"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [
                                   {"op": "-", "args": ["u"]},
                                   {"op": "makearray", "args": [],
                                    "regions": [[[2, 3]], [[1, 1]]],
                                    "values": [
                                        {"op": "/", "args": [
                                            {"op": "-", "args": [
                                                {"op": "index", "args": ["_var", "i"]},
                                                {"op": "index", "args": [
                                                    "_var", {"op": "-", "args": ["i", 1]}]}]},
                                            "dx"]},
                                        {"op": "/", "args": [
                                            {"op": "index", "args": ["_var", "i"]}, "dx"]}]}]}}]}},
        "coupling": [{"type": "operator_compose", "systems": ["Chem", "Adv"],
                      "lifting": "pointwise"}]
    }))
    .expect("document loads");
    let mut flat = earthsci_ast::flatten(&file).expect("document flattens");
    assert_eq!(
        flat.lifted_shapes.get("Chem.O3"),
        Some(&vec![3]),
        "the lift ran"
    );
    let recorded = flat.state_variables["Chem.O3"]
        .shape
        .take()
        .filter(|s| !s.is_empty());
    flat.index_sets.retain(|k, _| !k.starts_with("_liftdim"));
    (flat, recorded)
}

/// With no recorded shape, a lifted state takes its grid shape from
/// `lifted_shapes`: the one index set of its extent names the axis (what a
/// newer `flatten` records too), and the bands are those of the same system
/// written with that shape.
#[test]
fn lifted_states_take_their_grid_shape() {
    let (flat, recorded) = lifted(json!({"x": {"kind": "interval", "size": 3}}));
    if let Some(r) = recorded {
        assert_eq!(r, ["x"]);
    }
    let sv = sysview_flat(&flat);
    assert_eq!(
        sv.variables["Chem.O3"].shape.as_deref(),
        Some(&["x".to_string()][..])
    );
    assert!(matches!(sv.variables["Chem.O3"], Cow::Owned(_)));
    assert!(matches!(sv.variables["Chem.k1"], Cow::Borrowed(_)));
    assert_eq!(sv.index_sets.len(), 1, "no synthetic axis");
    let ctx = ctx_of(&sv).unwrap();
    assert_eq!(ctx.shapes.get("Chem.O3"), Some(&vec![3]));
    let direct = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    assert!(!direct.is_empty());
    let file = wrapped_with(&flat, &[("Chem.O3".into(), vec!["x".into()])], json!({}));
    let via = jacobian_bands(&file, Some(MODEL), Wrt::States).unwrap();
    assert_eq!(ser(&direct), ser(&via));
    // The evaluation document declares the grid shape too.
    let wsv = sysview(&file, Some(MODEL)).unwrap();
    let (dd, fd) = evaluation_document(&sv, &direct, false, 0).unwrap();
    let (dv, fv) = evaluation_document(&wsv, &direct, false, 0).unwrap();
    assert_eq!(fd, fv);
    assert_eq!(stable_json(&dd), stable_json(&dv));
}

/// With no recorded shape and no single index set of the grid's extent, the
/// axis is the synthetic dense `_liftdim<d>_<extent>`, added to the view's
/// index sets.
#[test]
fn ambiguous_lift_extents_take_a_synthetic_axis() {
    let (flat, recorded) = lifted(json!({
        "x": {"kind": "interval", "size": 3},
        "y": {"kind": "interval", "size": 3}}));
    // no operand names the grid, so a newer `flatten` records the same
    if let Some(r) = recorded {
        assert_eq!(r, ["_liftdim1_3"]);
    }
    let sv = sysview_flat(&flat);
    assert_eq!(
        sv.variables["Chem.O3"].shape.as_deref(),
        Some(&["_liftdim1_3".to_string()][..])
    );
    assert_eq!(sv.index_sets["_liftdim1_3"].size, Some(3));
    assert_eq!(ctx_of(&sv).unwrap().shapes.get("Chem.O3"), Some(&vec![3]));
    let direct = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let file = wrapped_with(
        &flat,
        &[("Chem.O3".into(), vec!["_liftdim1_3".into()])],
        json!({"_liftdim1_3": {"kind": "interval", "size": 3}}),
    );
    let via = jacobian_bands(&file, Some(MODEL), Wrt::States).unwrap();
    assert_eq!(ser(&direct), ser(&via));
}
