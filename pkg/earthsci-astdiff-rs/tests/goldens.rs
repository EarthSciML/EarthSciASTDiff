//! Cross-language conformance gates: the band-list and block goldens must
//! match byte-for-byte; the value goldens to 1e-10 relative with the exact
//! (row, col) key set.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use earthsci_astdiff::{
    assemble_jacobian, evaluation_document, jacobian_bands, jacobian_block,
    serialize_band, stable_json, sysview, Wrt,
};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load(name: &str) -> earthsci_ast::EsmFile {
    let path = root().join("tests/valid").join(format!("{name}.esm"));
    earthsci_ast::load_path(&path).expect("fixture loads")
}

const NAMES: [&str; 6] = ["bd_chem", "adv_interior", "contracted_ops",
                          "flux_form_adv", "minmod_adv", "contracting_observed"];

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
        for (base, pts) in cells {
            // nested arrays, 1-based cells, any rank
            let rank = pts[0].0.len();
            let shape: Vec<i64> = (0..rank)
                .map(|d| pts.iter().map(|(i, _)| i[d]).max().unwrap()).collect();
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
