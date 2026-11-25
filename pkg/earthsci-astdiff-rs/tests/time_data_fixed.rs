//! Time bands with data held fixed (spec §4.2): only observed that move with
//! `t` are followed, and an observed declared without a shape is read at a
//! cell. Byte-identical to the Julia reference's golden.

use std::fs;

use earthsci_astdiff::{jacobian_bands, serialize_band, stable_json, Wrt};
use serde_json::Value;

mod common;
use common::{load, root};

#[test]
fn tgrad_data_fixed_time_bands() {
    let file = load("tgrad_data_fixed");
    let entries = jacobian_bands(&file, Some("Box"), Wrt::Time).unwrap();
    assert_eq!(entries.len(), 1);
    let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
    let want =
        fs::read_to_string(root().join("tests/goldens/tgrad_data_fixed.time.bands.json")).unwrap();
    assert_eq!(text, want);
}
