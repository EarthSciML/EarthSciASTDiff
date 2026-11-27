//! A makearray whose region values are cell reads (scalars), read at a cell
//! through an observed: each value is read as itself, never wrapped in an
//! `index` with no subscripts. Byte-identical to the Julia reference's
//! golden.

use std::fs;

use earthsci_astdiff::{jacobian_bands, serialize_band, stable_json, Wrt};
use serde_json::Value;

mod common;
use common::{load, root};

#[test]
fn makearray_of_reads_bands() {
    let file = load("makearray_of_reads");
    let entries = jacobian_bands(&file, Some("M"), Wrt::States).unwrap();
    let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
    let want =
        fs::read_to_string(root().join("tests/goldens/makearray_of_reads.bands.json")).unwrap();
    assert_eq!(text, want);
    let pairs: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.u.as_str(), e.v.as_str()))
        .collect();
    assert_eq!(pairs, [("u", "u"), ("u", "v"), ("v", "v")]);
}
