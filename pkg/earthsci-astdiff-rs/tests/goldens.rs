//! Cross-language conformance gates: the band-list and block goldens must
//! match byte-for-byte; the value goldens to 1e-10 relative with the exact
//! (row, col) key set.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use earthsci_astdiff::{
    assemble_jacobian, jacobian_bands, jacobian_block, serialize_band,
    stable_json, Wrt,
};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load(name: &str) -> earthsci_ast::types::EsmFile {
    let path = root().join("tests/valid").join(format!("{name}.esm"));
    earthsci_ast::parse::load_path(&path).expect("fixture loads")
}

const NAMES: [&str; 5] = ["bd_chem", "adv_interior", "contracted_ops",
                          "flux_form_adv", "minmod_adv"];

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
fn value_goldens() {
    for name in NAMES {
        let file = load(name);
        let g: Value = serde_json::from_str(&fs::read_to_string(
            root().join("tests/goldens").join(format!("{name}.jvals.json")))
            .unwrap()).unwrap();
        // cell-named state values → nested arrays
        let mut states: HashMap<String, Value> = HashMap::new();
        let mut cells: HashMap<String, Vec<(Vec<i64>, f64)>> = HashMap::new();
        for (cellname, v) in g["u"].as_object().unwrap() {
            let val = v.as_f64().unwrap();
            if let Some(br) = cellname.find('[') {
                let base = cellname[..br].to_string();
                let idx: Vec<i64> = cellname[br + 1..cellname.len() - 1]
                    .split(',').map(|x| x.parse().unwrap()).collect();
                cells.entry(base).or_default().push((idx, val));
            } else {
                states.insert(cellname.clone(), Value::from(val));
            }
        }
        for (base, mut pts) in cells {
            pts.sort_by(|a, b| a.0.cmp(&b.0));
            let rank = pts[0].0.len();
            assert_eq!(rank, 1, "test helper handles rank-1 states");
            let n = pts.iter().map(|(i, _)| i[0]).max().unwrap();
            let mut arr = vec![Value::Null; n as usize];
            for (i, v) in pts {
                arr[(i[0] - 1) as usize] = Value::from(v);
            }
            states.insert(base, Value::Array(arr));
        }
        let params: HashMap<String, Value> = g["params"].as_object().unwrap()
            .iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let env = earthsci_astdiff::numeric::Env {
            states,
            params,
            t: g["t"].as_f64().unwrap(),
        };
        let got = assemble_jacobian(&file, None, Wrt::States, &env).unwrap();
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
