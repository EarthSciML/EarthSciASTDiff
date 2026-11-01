//! The goldens this binding produces (spec §5): those of the fixtures the
//! Julia reference cannot produce yet. Causal recurrences (spec §4.5) need an
//! evaluator that executes them, which EarthSciAST.jl does not have
//! (EarthSciAST#546), so `recurrence_sweep`'s goldens — and `pbl_rank1`'s,
//! whose `diag_plus_rank1` plan (spec §6) the reference has not ported —
//! are written here and pinned in every binding like any other golden. The
//! values come from EarthSciAST evaluating this crate's product documents;
//! `tests/recurrence.rs` checks them against central differences of the
//! source's own right-hand side.
//!
//! Run after an INTENDED change:
//! `cargo test --test write_goldens -- --ignored` (with the
//! `patch.crates-io.earthsci-ast` path the conformance runner uses).

use std::collections::HashMap;
use std::fs;

use earthsci_astdiff::{
    evaluation_document, hvp_document, jacobian_bands, jacobian_bands2, jacobian_block,
    jacobian_block2, jacobian_block_with, jvp_document, serialize_band, serialize_band2,
    stable_json, sysview, vjp_document, BlockOptions, HvpForm, Layout, ProductOptions, ScatterPlan,
    Wrt,
};
use serde_json::{json, Map, Value};

mod common;
use common::docvals::{column_cells, doc_hvp, doc_jacobian, Env, HvpPair};
use common::{cells_env, load, root};

/// (fixture, the single axes it carries goldens for, its pairs, its column
/// axis): kept in step with the lists in goldens.rs, products.rs,
/// second_order.rs, scatter.rs and the Python tests.
#[allow(clippy::type_complexity)]
pub const RUST_OWNED: [(&str, &[Wrt], &[(Wrt, Wrt)], &str); 2] = [
    (
        "recurrence_sweep",
        &[Wrt::States, Wrt::Parameters],
        &[(Wrt::States, Wrt::States), (Wrt::States, Wrt::Parameters)],
        "lev",
    ),
    ("pbl_rank1", &[Wrt::States], &[], "lev"),
];

fn write(file: &str, text: String) {
    let p = root().join("tests/goldens").join(file);
    fs::write(&p, text).unwrap();
    println!("wrote {}", p.display());
}

fn tag(wrt: Wrt) -> String {
    if wrt == Wrt::States {
        String::new()
    } else {
        format!(".{}", wrt.as_str())
    }
}

/// The pinned point of the value goldens, as the reference pins it
/// (scripts/regenerate-value-goldens.jl): state cell `i` (1-based, in the
/// source's state layout) at `0.2 + 0.6·|sin(2.3 i)|`, `t = 0.7`, the scalar
/// parameters at their document defaults.
fn pinned(name: &str) -> (Value, Value, Env) {
    let file = load(name);
    let sv = sysview(&file, None).unwrap();
    let cells = column_cells(&sv, Wrt::States, None).unwrap();
    let mut u = Map::new();
    for (i, c) in cells.iter().enumerate() {
        u.insert(
            c.clone(),
            json!(0.2 + 0.6 * (2.3 * (i + 1) as f64).sin().abs()),
        );
    }
    let mut params = Map::new();
    for (n, v) in sv.variables.iter() {
        if v.var_type == earthsci_ast::VariableType::Parameter
            && v.shape.as_ref().is_none_or(|s| s.is_empty())
        {
            if let Some(x) = v
                .default
                .as_ref()
                .and_then(|d| serde_json::to_value(d).ok())
            {
                params.insert(n.clone(), x);
            }
        }
    }
    let u = Value::Object(u);
    let env = cells_env(
        &u,
        params.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        0.7,
    );
    (u, Value::Object(params), env)
}

#[test]
#[ignore = "writes goldens; run explicitly after an intended change"]
fn write_rust_goldens() {
    let opts = ProductOptions::default();
    for (name, axes, pairs, column_axis) in RUST_OWNED {
        let file = load(name);
        let sv = sysview(&file, None).unwrap();
        let (u, params, env) = pinned(name);
        for &wrt in axes {
            let t = tag(wrt);
            let entries = jacobian_bands(&file, None, wrt).unwrap();
            write(
                &format!("{name}{t}.bands.json"),
                stable_json(&Value::Array(entries.iter().map(serialize_band).collect())),
            );
            write(
                &format!("{name}{t}.block.json"),
                stable_json(&jacobian_block(&file, None, wrt, true, 12).unwrap()),
            );
            let (doc, _) = evaluation_document(&sv, &entries, true, 12).unwrap();
            let (doc0, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
            let nocse = format!("{name}{t}.evaldoc-nocse.json");
            if stable_json(&doc0) != stable_json(&doc) {
                write(&nocse, stable_json(&doc0));
            } else {
                let _ = fs::remove_file(root().join("tests/goldens").join(&nocse));
            }
            write(&format!("{name}{t}.evaldoc.json"), stable_json(&doc));
            let (jd, _) = jvp_document(&sv, &entries, wrt, &opts).unwrap();
            write(&format!("{name}{t}.jvp.evaldoc.json"), stable_json(&jd));
            let (vd, _) = vjp_document(&sv, &entries, wrt, &opts).unwrap();
            write(&format!("{name}{t}.vjp.evaldoc.json"), stable_json(&vd));
            let jac = doc_jacobian(&file, None, wrt, None, &env).unwrap();
            let rows: Vec<Value> = jac.iter().map(|((r, c), v)| json!([r, c, v])).collect();
            write(
                &format!("{name}{t}.jvals.json"),
                stable_json(&json!({"t": 0.7, "u": u, "params": params, "entries": rows})),
            );
            if wrt == Wrt::States {
                let plan = ScatterPlan::from_file(&file, None, &Layout::Csr).unwrap();
                write(
                    &format!("{name}.csr.plan.json"),
                    stable_json(&plan.to_json()),
                );
                let bopts = BlockOptions {
                    column_axis: Some(column_axis),
                    ..Default::default()
                };
                write(
                    &format!("{name}.column-{column_axis}.block.json"),
                    stable_json(&jacobian_block_with(&file, None, wrt, &bopts).unwrap()),
                );
            }
        }
        for &wrt in pairs {
            let pt = format!("{}-{}", wrt.0.as_str(), wrt.1.as_str());
            let entries = jacobian_bands2(&file, None, wrt).unwrap();
            write(
                &format!("{name}.{pt}.bands.json"),
                stable_json(&Value::Array(entries.iter().map(serialize_band2).collect())),
            );
            write(
                &format!("{name}.{pt}.block.json"),
                stable_json(&jacobian_block2(&file, None, wrt, None, true, 12).unwrap()),
            );
            let forms: &[HvpForm] = if wrt == (Wrt::States, Wrt::States) {
                &[HvpForm::Contract, HvpForm::Symmetric]
            } else {
                &[HvpForm::Contract]
            };
            for &form in forms {
                let (d, _) = hvp_document(&sv, &entries, wrt, 2, form, &opts).unwrap();
                let kind = if form == HvpForm::Contract {
                    "hvp"
                } else {
                    "hvp-sym"
                };
                write(&format!("{name}.{pt}.{kind}.evaldoc.json"), stable_json(&d));
            }
            // the reference's stage pairs: μ_k = sin(1.7k + 0.9s), w_k =
            // cos(0.6k + 1.1s) over the state cells (k 1-based), s = 1, 2;
            // parameters at their defaults
            let rows = column_cells(&sv, Wrt::States, None).unwrap();
            let cols = column_cells(&sv, wrt.0, None).unwrap();
            let henv = Env {
                params: HashMap::new(),
                ..env.clone()
            };
            let ps: Vec<HvpPair> = (1..=2)
                .map(|s| {
                    let mu = rows
                        .iter()
                        .enumerate()
                        .map(|(k, c)| (c.clone(), (1.7 * (k + 1) as f64 + 0.9 * s as f64).sin()))
                        .collect();
                    let w = cols
                        .iter()
                        .enumerate()
                        .map(|(k, c)| (c.clone(), (0.6 * (k + 1) as f64 + 1.1 * s as f64).cos()))
                        .collect();
                    (mu, w)
                })
                .collect();
            let g = doc_hvp(&file, None, wrt, None, &henv, &ps, HvpForm::Contract).unwrap();
            let jpairs: Vec<Value> = ps.iter().map(|(mu, w)| json!({"mu": mu, "w": w})).collect();
            let gl: Vec<Value> = g.iter().map(|(n, v)| json!([n, v])).collect();
            write(
                &format!("{name}.{pt}.hvals.json"),
                stable_json(&json!({
                    "wrt": [wrt.0.as_str(), wrt.1.as_str()],
                    "t": 0.7, "u": u, "pairs": jpairs, "g": gl,
                })),
            );
        }
    }
}
