//! A guard holds over a composed derivative (spec §3.6, named steps): the
//! derivative of `g = ifelse(s > 0, 10^lg, 1)` through the observed
//! `lg = x/s` is composed as `(∂g/∂lg)·_du_lg` with
//! `∂g/∂lg = ifelse(s > 0, …, 0)`, and where `s = 0` the inner derivative
//! `_du_lg` is NaN; the product is written `ifelse(s > 0, …·_du_lg, 0)`, so
//! the guard keeps the derivative finite, as MOSAIC's MTEM sulfate-poor log
//! gamma needs. The shared golden
//! (`tests/goldens/guarded_quotient.points.json`) holds the exact
//! derivatives with the guard holding and open; the Julia and Python
//! bindings check it too.

use std::collections::HashMap;
use std::fs;

use earthsci_astdiff::Wrt;
use serde_json::Value;

mod common;
use common::docvals::doc_jacobian;
use common::{cells_env, load, root};

#[test]
fn the_shared_points() {
    let file = load("guarded_quotient");
    let g: Value = serde_json::from_str(
        &fs::read_to_string(root().join("tests/goldens/guarded_quotient.points.json")).unwrap(),
    )
    .unwrap();
    for pt in g["points"].as_array().unwrap() {
        let params: HashMap<String, Value> = pt["params"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let env = cells_env(&pt["u"], params, g["t"].as_f64().unwrap());
        let got = doc_jacobian(&file, None, Wrt::States, None, &env).unwrap();
        assert!(got.values().all(|v| v.is_finite()), "{got:?}");
        for e in pt["entries"].as_array().unwrap() {
            let (r, c, v) = (
                e[0].as_str().unwrap().to_string(),
                e[1].as_str().unwrap().to_string(),
                e[2].as_f64().unwrap(),
            );
            let gv = got.get(&(r.clone(), c.clone())).copied().unwrap_or(0.0);
            if v == 0.0 {
                assert_eq!(gv, 0.0, "({r}, {c})");
            } else {
                assert!((gv - v).abs() <= 1e-12 * v.abs(), "({r}, {c}): {gv} vs {v}");
            }
        }
    }
}
