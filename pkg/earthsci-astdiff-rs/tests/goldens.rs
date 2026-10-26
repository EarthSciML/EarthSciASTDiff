//! Cross-language conformance gates: the band-list and block goldens must
//! match byte-for-byte; the value goldens to 1e-10 relative with the exact
//! (row, col) key set.

use std::fs;

use earthsci_astdiff::{
    assemble_jacobian, evaluation_document, jacobian_bands, jacobian_block,
    serialize_band, stable_json, sysview, Wrt,
};
use serde_json::Value;

mod common;
use common::{load, root};

const NAMES: [&str; 7] = ["bd_chem", "adv_interior", "contracted_ops",
                          "flux_form_adv", "minmod_adv", "contracting_observed",
                          "tgrad_photolysis"];
/// Fixtures with `<name>.time.*` goldens (`wrt = time`, spec §4.2).
const TIME_NAMES: [&str; 1] = ["tgrad_photolysis"];

fn golden(file: &str) -> String {
    fs::read_to_string(root().join("tests/goldens").join(file)).unwrap()
}

#[test]
fn band_goldens() {
    for name in NAMES {
        let file = load(name);
        let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
        let text = stable_json(&Value::Array(
            entries.iter().map(serialize_band).collect()));
        let golden = fs::read_to_string(
            root().join("tests/goldens").join(format!("{name}.bands.json")))
            .unwrap();
        assert_eq!(text, golden, "band golden diverges: {name}");
    }
}

#[test]
fn block_goldens() {
    for name in NAMES {
        let file = load(name);
        let block = jacobian_block(&file, None, Wrt::States, true, 12).unwrap();
        let golden = fs::read_to_string(
            root().join("tests/goldens").join(format!("{name}.block.json")))
            .unwrap();
        assert_eq!(stable_json(&block), golden, "block golden diverges: {name}");
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
        let golden = fs::read_to_string(
            root().join("tests/goldens").join(format!("{name}.evaldoc.json")))
            .unwrap();
        assert_eq!(stable_json(&doc), golden, "evaldoc golden diverges: {name}");
        // without hoisting; no `-nocse` golden means nothing to hoist
        let (doc0, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
        let p0 = root().join("tests/goldens").join(format!("{name}.evaldoc-nocse.json"));
        let golden0 = if p0.exists() { fs::read_to_string(p0).unwrap() } else { golden };
        assert_eq!(stable_json(&doc0), golden0, "evaldoc (cse = false) diverges: {name}");
    }
}

#[test]
fn value_goldens() {
    for name in NAMES {
        check_values(name, name, Wrt::States);
    }
}

fn check_values(name: &str, golden: &str, wrt: Wrt) {
    {
        let file = load(name);
        let (env, g) = common::jvals_env(golden);
        let got = assemble_jacobian(&file, None, wrt, &env).unwrap();
        let want: Vec<(String, String, f64)> = g["entries"].as_array().unwrap()
            .iter().map(|e| {
                let a = e.as_array().unwrap();
                (a[0].as_str().unwrap().to_string(),
                 a[1].as_str().unwrap().to_string(),
                 a[2].as_f64().unwrap())
            }).collect();
        assert_eq!(got.len(), want.len(), "{name}: pattern key count");
        for (r, c, v) in want {
            let gv = got.get(&(r.clone(), c.clone()))
                .unwrap_or_else(|| panic!("{name}: missing ({r}, {c})"));
            let tol = 1e-10 * v.abs().max(1e-3);
            assert!((gv - v).abs() <= tol.max(1e-13),
                    "{name}: ({r},{c}) {gv} vs {v}");
        }
    }
}

#[test]
fn time_band_goldens() {
    for name in TIME_NAMES {
        let file = load(name);
        let entries = jacobian_bands(&file, None, Wrt::Time).unwrap();
        assert!(!entries.is_empty(), "{name}: no time bands");
        // spec §4.2: one scalar column `t`, no contracted dimensions
        assert!(entries.iter().all(|en| en.v == "t" && en.band.cidx.is_empty()
                                    && en.band.contracted.is_empty()));
        let text = stable_json(&Value::Array(
            entries.iter().map(serialize_band).collect()));
        assert_eq!(text, golden(&format!("{name}.time.bands.json")),
                   "time band golden diverges: {name}");
        let block = jacobian_block(&file, None, Wrt::Time, true, 12).unwrap();
        assert_eq!(stable_json(&block), golden(&format!("{name}.time.block.json")),
                   "time block golden diverges: {name}");
        let sv = sysview(&file, None).unwrap();
        let (doc, _) = evaluation_document(&sv, &entries, true, 12).unwrap();
        let ev = golden(&format!("{name}.time.evaldoc.json"));
        assert_eq!(stable_json(&doc), ev, "time evaldoc golden diverges: {name}");
        let (doc0, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
        let p0 = root().join("tests/goldens").join(format!("{name}.time.evaldoc-nocse.json"));
        let ev0 = if p0.exists() { fs::read_to_string(p0).unwrap() } else { ev };
        assert_eq!(stable_json(&doc0), ev0, "time evaldoc (cse = false) diverges: {name}");
    }
}

#[test]
fn time_value_goldens() {
    for name in TIME_NAMES {
        check_values(name, &format!("{name}.time"), Wrt::Time);
    }
}

#[test]
fn time_bands_empty_for_autonomous() {
    // spec §4.2: an empty time band list means f_t ≡ 0
    for name in ["bd_chem", "adv_interior", "minmod_adv"] {
        assert!(jacobian_bands(&load(name), None, Wrt::Time).unwrap().is_empty(), "{name}");
    }
}
