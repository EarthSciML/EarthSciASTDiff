//! #4 W5: band-field timings — EarthSciAST evaluating the evaluation
//! document (this crate evaluates nothing itself) against one evaluation of
//! the operator's own RHS. Median of 50 warm single-thread calls.
//!
//!     cargo run --release --example bench_production -- [cells]
//!
//! B1: `bd_chem` on `cells` cells (default 6552, the CONUS 4×5° count).
//! Column: a 2-D chemistry + vertical-exchange operator on 72 levels ×
//! `cells/72` columns. Compare with
//! `scripts/bench-production.jl` (EarthSciAST.jl evaluating the same
//! documents' band fields).

use std::time::Instant;

use earthsci_ast::{esm_problem, rhs_program, Compiler, EsmFile, ProblemOptions, RhsProgram};
use earthsci_astdiff::{evaluation_document, jacobian_bands_view, sysview, Wrt};
use serde_json::{json, Value};

fn median<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut t: Vec<f64> = (0..50)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[25]
}

fn us(x: f64) -> String {
    format!("{:>10.1} µs", x * 1e6)
}

pub fn bd_chem(n: usize) -> Value {
    let mut d: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/valid/bd_chem.esm"
        ))
        .unwrap(),
    )
    .unwrap();
    d["index_sets"]["z"]["size"] = json!(n);
    d
}

pub fn column(nx: usize, nz: usize) -> Value {
    let ix = |v: &str, a: Value, b: Value| json!({"op": "index", "args": [v, a, b]});
    let kp1 = json!({"op": "+", "args": ["k", 1]});
    let chem = |k: Value| {
        json!({"op": "*", "args": [-1, "k1", ix("A", json!("i"), k.clone()),
                                                   ix("B", json!("i"), k)]})
    };
    json!({
        "esm": "1.1.0",
        "metadata": {"name": "Column"},
        "index_sets": {"x": {"kind": "interval", "size": nx}, "z": {"kind": "interval", "size": nz}},
        "models": {"Col": {
            "variables": {
                "A": {"type": "unknown", "shape": ["x", "z"], "default": 1.0},
                "B": {"type": "unknown", "shape": ["x", "z"], "default": 0.5},
                "k1": {"type": "parameter", "default": 0.3},
                "kz": {"type": "parameter", "default": 0.8}
            },
            "equations": [
                {"lhs": {"op": "D", "args": ["A"], "wrt": "t"},
                 "rhs": {"op": "makearray", "args": [],
                         "regions": [[[1, nx], [1, nz - 1]], [[1, nx], [nz, nz]]],
                         "values": [
                    {"op": "faq", "output_idx": ["i", "k"], "args": [],
                     "ranges": {"i": {"from": "x"}, "k": [1, nz - 1]},
                     "expr": {"op": "+", "args": [chem(json!("k")),
                        {"op": "*", "args": ["kz", {"op": "-", "args": [
                            ix("A", json!("i"), kp1.clone()), ix("A", json!("i"), json!("k"))]}]}]}},
                    {"op": "faq", "output_idx": ["i", "k"], "args": [],
                     "ranges": {"i": {"from": "x"}, "k": [nz, nz]}, "expr": chem(json!("k"))}]}},
                {"lhs": {"op": "D", "args": ["B"], "wrt": "t"},
                 "rhs": {"op": "faq", "output_idx": ["i", "k"], "args": [],
                         "ranges": {"i": {"from": "x"}, "k": {"from": "z"}},
                         "expr": {"op": "*", "args": ["k1", ix("A", json!("i"), json!("k")),
                                                      ix("B", json!("i"), json!("k"))]}}}
            ]
        }}
    })
}

fn source(file: &EsmFile) -> RhsProgram {
    rhs_program(&esm_problem(file, (0.0, 1.0), ProblemOptions::default()).unwrap()).unwrap()
}

fn state(src: &RhsProgram) -> (Vec<f64>, Vec<f64>) {
    let u = (0..src.n_states())
        .map(|i| 0.3 + 0.5 * ((i as f64) * 0.618).fract())
        .collect();
    let p = src
        .parameters()
        .iter()
        .map(|s| s.default.unwrap_or(0.5))
        .collect();
    (u, p)
}

fn rhs_time(src: &RhsProgram, u: &[f64], p: &[f64]) -> f64 {
    let mut ws = src.new_workspace();
    let f = src.default_forcing();
    let mut du = vec![0.0; u.len()];
    median(|| src.eval_into(u, &[], p, 0.5, &f, &mut du, &mut ws).unwrap())
}

/// The band fields: EarthSciAST compiling and evaluating the evaluation
/// document (`evaluation_document`, CSE on), next to the operator's own RHS.
/// Placing the values in a matrix is a scatter plan's work, the consumer's,
/// and is not timed.
fn band_fields(name: &str, file: &EsmFile) {
    let src = source(file);
    let (u, p) = state(&src);
    let t0 = Instant::now();
    let sv = sysview(file, None).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, fields) = evaluation_document(&sv, &entries, true, 12).unwrap();
    let opts = ProblemOptions {
        compiler: Some(Compiler::Native),
        ..Default::default()
    };
    let ev = rhs_program(&esm_problem(&doc, (0.0, 1.0), opts).unwrap()).unwrap();
    let build = t0.elapsed().as_secs_f64();
    let (eu, ep) = state(&ev);
    let tj = rhs_time(&ev, &eu, &ep);
    let rhs = rhs_time(&src, &u, &p);
    println!(
        "{name:<10} build {build:>7.3} s | band fields {} | {:.2}× RHS {} | {} band fields",
        us(tj),
        tj / rhs,
        us(rhs),
        fields.len()
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(6552);
    // `--write-docs DIR`: the benchmark documents, for scripts/bench-production.jl
    if let Some(dir) = std::env::args().skip_while(|a| a != "--write-docs").nth(1) {
        let w = |name: &str, d: Value| {
            std::fs::write(
                format!("{dir}/{name}.esm"),
                serde_json::to_string(&d).unwrap(),
            )
            .unwrap()
        };
        w("bd_chem", bd_chem(n));
        w("column", column((n / 72).max(1), 72));
    }
    let chem = earthsci_ast::load_document(&bd_chem(n)).unwrap();
    println!("B1: bd_chem, {n} cells × 3 species");
    band_fields("bd_chem", &chem);
    let nz = 72;
    let nx = (n / nz).max(1);
    let col = earthsci_ast::load_document(&column(nx, nz)).unwrap();
    println!("Column: {nx} columns × {nz} levels × 2 species, vertical exchange");
    band_fields("column", &col);
}
