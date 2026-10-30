//! Shared fixture helpers for the integration tests.
#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use earthsci_astdiff::numeric::Env;
use serde_json::Value;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn load(name: &str) -> earthsci_ast::EsmFile {
    let path = root().join("tests/valid").join(format!("{name}.esm"));
    earthsci_ast::load_path(&path).expect("fixture loads")
}

/// The pinned point of `<name>.jvals.json` as an oracle `Env`, plus the
/// golden itself.
pub fn jvals_env(name: &str) -> (Env, Value) {
    let g: Value = serde_json::from_str(
        &fs::read_to_string(
            root()
                .join("tests/goldens")
                .join(format!("{name}.jvals.json")),
        )
        .unwrap(),
    )
    .unwrap();
    let params: HashMap<String, Value> = g["params"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let env = cells_env(&g["u"], params, g["t"].as_f64().unwrap());
    (env, g)
}

/// An oracle `Env` from cell-named state values (`{"c[2]": 0.5, "S": 1.0}`)
/// as the value goldens carry them.
pub fn cells_env(u: &Value, params: HashMap<String, Value>, t: f64) -> Env {
    // cell-named state values → nested arrays
    let mut states: HashMap<String, Value> = HashMap::new();
    let mut cells: HashMap<String, Vec<(Vec<i64>, f64)>> = HashMap::new();
    for (cellname, v) in u.as_object().unwrap() {
        let val = v.as_f64().unwrap();
        if let Some(br) = cellname.find('[') {
            let base = cellname[..br].to_string();
            let idx: Vec<i64> = cellname[br + 1..cellname.len() - 1]
                .split(',')
                .map(|x| x.parse().unwrap())
                .collect();
            cells.entry(base).or_default().push((idx, val));
        } else {
            states.insert(cellname.clone(), Value::from(val));
        }
    }
    for (base, pts) in cells {
        // nested arrays, 1-based cells, any rank
        let rank = pts[0].0.len();
        let shape: Vec<i64> = (0..rank)
            .map(|d| pts.iter().map(|(i, _)| i[d]).max().unwrap())
            .collect();
        fn full(shape: &[i64]) -> Value {
            match shape.split_first() {
                None => Value::Null,
                Some((n, rest)) => Value::Array(vec![full(rest); *n as usize]),
            }
        }
        let mut arr = full(&shape);
        for (i, v) in pts {
            let mut cur = &mut arr;
            for c in &i {
                cur = &mut cur.as_array_mut().unwrap()[(*c - 1) as usize];
            }
            *cur = Value::from(v);
        }
        states.insert(base, arr);
    }
    Env { states, params, t }
}
