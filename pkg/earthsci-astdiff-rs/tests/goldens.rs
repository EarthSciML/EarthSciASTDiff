//! Cross-language conformance gates: the band-list and block goldens must
//! match byte-for-byte; the value goldens to 1e-10 relative with the exact
//! (row, col) key set.

use std::fs;

use earthsci_astdiff::{
    evaluation_document, jacobian_bands, jacobian_block, parse_jacobian_block, serialize_band,
    stable_json, sysview, Wrt,
};
use serde_json::Value;

mod common;
use common::docvals::doc_jacobian;
use common::{load, root};

/// The first ten are the reference's; the last two are this binding's
/// (tests/write_goldens.rs: the reference cannot produce them yet).
const NAMES: [&str; 12] = [
    "bd_chem",
    "adv_interior",
    "contracted_ops",
    "flux_form_adv",
    "minmod_adv",
    "contracting_observed",
    "tgrad_photolysis",
    "forcing_bracket",
    "filtered_scan",
    "semiring_reduce",
    "recurrence_sweep",
    "pbl_rank1",
];
/// Per-(fixture, axis) goldens beyond the states axis: `<name>.<axis>.*`
/// (`wrt = time`, spec §4.2; `wrt = forcing` / `parameters`, spec §4.3).
const AXES: [(&str, Wrt); 4] = [
    ("tgrad_photolysis", Wrt::Time),
    ("forcing_bracket", Wrt::Forcing),
    ("forcing_bracket", Wrt::Parameters),
    ("recurrence_sweep", Wrt::Parameters),
];
/// Extra value-golden points per (fixture, axis): `<name>.<axis><sfx>.jvals.json`.
const VALUE_POINTS: [(&str, Wrt, &[&str]); 4] = [
    ("tgrad_photolysis", Wrt::Time, &[""]),
    ("forcing_bracket", Wrt::Forcing, &["", ".w0"]),
    ("forcing_bracket", Wrt::Parameters, &["", ".w0"]),
    ("recurrence_sweep", Wrt::Parameters, &[""]),
];

fn golden(file: &str) -> String {
    fs::read_to_string(root().join("tests/goldens").join(file)).unwrap()
}

/// An optional golden (`-nocse` exists only where hoisting changes the doc).
fn golden_or(file: &str, fallback: String) -> String {
    let p = root().join("tests/goldens").join(file);
    if p.exists() {
        fs::read_to_string(p).unwrap()
    } else {
        fallback
    }
}

#[test]
fn band_goldens() {
    for name in NAMES {
        let file = load(name);
        let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
        let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
        assert_eq!(
            text,
            golden(&format!("{name}.bands.json")),
            "band golden diverges: {name}"
        );
    }
}

#[test]
fn block_goldens() {
    for name in NAMES {
        let file = load(name);
        let block = jacobian_block(&file, None, Wrt::States, true, 12).unwrap();
        assert_eq!(
            stable_json(&block),
            golden(&format!("{name}.block.json")),
            "block golden diverges: {name}"
        );
        // the round trip is exact (a `where` guard included)
        let (w, back) = parse_jacobian_block(&block).unwrap();
        assert_eq!(w, Wrt::States);
        assert_eq!(
            stable_json(&Value::Array(back.iter().map(serialize_band).collect())),
            golden(&format!("{name}.bands.json")),
            "block round trip: {name}"
        );
    }
}

#[test]
fn evaldoc_goldens() {
    for name in NAMES {
        let file = load(name);
        let sv = sysview(&file, None).unwrap();
        let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
        let (doc, names) = evaluation_document(&sv, &entries, true, 12).unwrap();
        assert_eq!(names.len(), entries.len());
        let ev = golden(&format!("{name}.evaldoc.json"));
        assert_eq!(stable_json(&doc), ev, "evaldoc golden diverges: {name}");
        // without hoisting; no `-nocse` golden means nothing to hoist
        let (doc0, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
        let ev0 = golden_or(&format!("{name}.evaldoc-nocse.json"), ev);
        assert_eq!(
            stable_json(&doc0),
            ev0,
            "evaldoc (cse = false) diverges: {name}"
        );
    }
}

#[test]
fn value_goldens() {
    for name in NAMES {
        check_values(name, name, Wrt::States);
    }
}

fn check_values(name: &str, golden: &str, wrt: Wrt) {
    let file = load(name);
    let (env, g) = common::jvals_env(golden);
    let got = doc_jacobian(&file, None, wrt, None, &env).unwrap();
    let want: Vec<(String, String, f64)> = g["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let a = e.as_array().unwrap();
            (
                a[0].as_str().unwrap().to_string(),
                a[1].as_str().unwrap().to_string(),
                a[2].as_f64().unwrap(),
            )
        })
        .collect();
    assert_eq!(got.len(), want.len(), "{name}: pattern key count");
    for (r, c, v) in want {
        let gv = got
            .get(&(r.clone(), c.clone()))
            .unwrap_or_else(|| panic!("{name}: missing ({r}, {c})"));
        let tol = 1e-10 * v.abs().max(1e-3);
        assert!(
            (gv - v).abs() <= tol.max(1e-13),
            "{name}: ({r},{c}) {gv} vs {v}"
        );
    }
}

#[test]
fn axis_band_goldens() {
    for (name, wrt) in AXES {
        let axis = wrt.as_str();
        let file = load(name);
        let entries = jacobian_bands(&file, None, wrt).unwrap();
        assert!(!entries.is_empty(), "{name}: no {axis} bands");
        if wrt == Wrt::Time {
            // spec §4.2: one scalar column `t`, no contracted dimensions
            assert!(entries
                .iter()
                .all(|en| en.v == "t" && en.band.cidx.is_empty() && en.band.contracted.is_empty()));
        }
        let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
        assert_eq!(
            text,
            golden(&format!("{name}.{axis}.bands.json")),
            "{axis} band golden diverges: {name}"
        );
        let block = jacobian_block(&file, None, wrt, true, 12).unwrap();
        // spec §4: a parameters / forcing block carries `columns`, not
        // `structure` / `factorization`
        assert_eq!(block.get("columns").is_some(), wrt.has_columns(), "{name}");
        assert_eq!(
            block.get("structure").is_none(),
            wrt.has_columns(),
            "{name}"
        );
        assert_eq!(
            stable_json(&block),
            golden(&format!("{name}.{axis}.block.json")),
            "{axis} block golden diverges: {name}"
        );
        let (w, back) = parse_jacobian_block(&block).unwrap();
        assert_eq!(w, wrt);
        assert_eq!(
            stable_json(&Value::Array(back.iter().map(serialize_band).collect())),
            text,
            "{axis} block round trip: {name}"
        );
        let sv = sysview(&file, None).unwrap();
        let (doc, _) = evaluation_document(&sv, &entries, true, 12).unwrap();
        let ev = golden(&format!("{name}.{axis}.evaldoc.json"));
        assert_eq!(
            stable_json(&doc),
            ev,
            "{axis} evaldoc golden diverges: {name}"
        );
        let (doc0, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
        let ev0 = golden_or(&format!("{name}.{axis}.evaldoc-nocse.json"), ev);
        assert_eq!(
            stable_json(&doc0),
            ev0,
            "{axis} evaldoc (cse = false) diverges: {name}"
        );
    }
}

#[test]
fn axis_value_goldens() {
    for (name, wrt, sfxs) in VALUE_POINTS {
        for sfx in sfxs {
            check_values(name, &format!("{name}.{}{sfx}", wrt.as_str()), wrt);
        }
    }
}

#[test]
fn time_bands_empty_for_autonomous() {
    // spec §4.2: an empty time band list means f_t ≡ 0; every fixture
    // without `.time.*` goldens is autonomous — except forcing_bracket, whose
    // record blend weight reads `t` explicitly (its time axis is not pinned)
    for name in NAMES
        .into_iter()
        .filter(|n| !AXES.contains(&(*n, Wrt::Time)) && *n != "forcing_bracket")
    {
        assert!(
            jacobian_bands(&load(name), None, Wrt::Time)
                .unwrap()
                .is_empty(),
            "{name}"
        );
    }
}
