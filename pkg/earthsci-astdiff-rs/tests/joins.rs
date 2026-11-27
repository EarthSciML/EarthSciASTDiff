//! Bands through a relational join (spec §3.5): a regrid whose only gate is
//! a spatial overlap join. Byte-identical to the Julia reference's goldens.

use std::fs;

use earthsci_astdiff::{
    evaluation_document, jacobian_bands_view, jacobian_block, jvp_document, serialize_band,
    stable_json, sysview, vjp_document, ProductOptions, Wrt,
};
use serde_json::Value;

mod common;
use common::{load, root};

fn golden(name: &str) -> String {
    fs::read_to_string(root().join("tests/goldens").join(name)).unwrap()
}

#[test]
fn join_regrid_goldens() {
    let file = load("join_regrid");
    let sv = sysview(&file, Some("Regrid")).unwrap();
    let opts = ProductOptions::default();
    for (wrt, tag) in [(Wrt::States, ""), (Wrt::Parameters, ".parameters")] {
        let entries = jacobian_bands_view(&sv, wrt, None).unwrap();
        let bands: Vec<Value> = entries.iter().map(serialize_band).collect();
        assert_eq!(
            stable_json(&Value::Array(bands)),
            golden(&format!("join_regrid{tag}.bands.json"))
        );
        let block = jacobian_block(&file, Some("Regrid"), wrt, true, 12).unwrap();
        assert_eq!(
            stable_json(&block),
            golden(&format!("join_regrid{tag}.block.json"))
        );
        let (ev, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
        assert_eq!(
            stable_json(&ev),
            golden(&format!("join_regrid{tag}.evaldoc.json"))
        );
        let (jv, _) = jvp_document(&sv, &entries, wrt, &opts).unwrap();
        assert_eq!(
            stable_json(&jv),
            golden(&format!("join_regrid{tag}.jvp.evaldoc.json"))
        );
        let (vj, _) = vjp_document(&sv, &entries, wrt, &opts).unwrap();
        assert_eq!(
            stable_json(&vj),
            golden(&format!("join_regrid{tag}.vjp.evaldoc.json"))
        );
        // the joined band carries the join and its ranges
        let col = if wrt == Wrt::States { "q" } else { "F" };
        let jb = entries.iter().find(|e| e.u == "c" && e.v == col).unwrap();
        let j = jb.band.join.as_ref().expect("joined band");
        assert_eq!(j.names(), ["i", "r"]);
    }
}
