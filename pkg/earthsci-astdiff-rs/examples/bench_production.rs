//! #4 W5: production assembly timings — `JacobianProgram::eval_into`, `apply`
//! and `apply_transpose` against one evaluation of the operator's own RHS.
//! Median of 50 warm single-thread calls.
//!
//!     cargo run --release --example bench_production -- [cells]
//!
//! B1: `bd_chem` on `cells` cells (default 6552, the CONUS 4×5° count),
//! `LaneBlocks`, domain and lanes (capacity 256). Column: a 2-D chemistry +
//! vertical-exchange operator on 72 levels × `cells/72` columns,
//! `ColumnBanded` and `Csr` (domain). Compare with
//! `scripts/bench-production.jl` (EarthSciAST.jl evaluating the same
//! documents' band fields).

use std::time::Instant;

use earthsci_ast::{esm_problem, rhs_program, EsmFile, ProblemOptions, RhsProgram};
use earthsci_astdiff::{BuildOptions, Exec, JacobianProgram, Layout};
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

fn domain(name: &str, file: &EsmFile, layout: Layout) {
    let src = source(file);
    let (u, p) = state(&src);
    let t0 = Instant::now();
    let prog = JacobianProgram::build(file, None, &BuildOptions::default()).unwrap();
    let plan = prog.plan(&layout, &src, Exec::Domain).unwrap();
    let build = t0.elapsed().as_secs_f64();
    let mut ws = prog.new_workspace(&plan);
    let f = src.default_forcing();
    let mut out = vec![0.0f64; plan.out_len()];
    let mut o32 = vec![0.0f32; plan.out_len()];
    let ev = median(|| {
        prog.eval_into(&plan, &u, &[], &p, 0.5, &f, &[], 0, &mut out, &mut ws)
            .unwrap()
    });
    let ev32 = median(|| {
        prog.eval_into(&plan, &u, &[], &p, 0.5, &f, &[], 0, &mut o32, &mut ws)
            .unwrap()
    });
    prog.eval_into(&plan, &u, &[], &p, 0.5, &f, &[], 0, &mut out, &mut ws)
        .unwrap();
    let mut y = vec![0.0; u.len()];
    let ap = median(|| prog.apply(&plan, &ws, &u, &mut y).unwrap());
    let mut g = vec![0.0; u.len()];
    let at = median(|| prog.apply_transpose(&plan, &ws, &u, &mut g).unwrap());
    let rhs = rhs_time(&src, &u, &p);
    println!(
        "{name:<28} build+plan {build:>7.3} s | eval_into {} (f32 {}) | {:.2}× RHS {} | \
              apply {} | apply_transpose {} ({:.2}× apply)",
        us(ev),
        us(ev32),
        ev / rhs,
        us(rhs),
        us(ap),
        us(at),
        at / ap
    );
}

fn lanes(name: &str, file: &EsmFile, layout: Layout, cap: usize) {
    let src = source(file);
    let (u, p) = state(&src);
    let prog = JacobianProgram::build(file, None, &BuildOptions::default()).unwrap();
    let plan = prog
        .plan(&layout, &src, Exec::Lanes { capacity: cap })
        .unwrap();
    let mut ws = prog.new_workspace(&plan);
    let f = src.default_forcing();
    let lay = src.layout();
    let ncell = lay.u[0].len();
    // every cell, batch by batch, its lane inputs gathered per batch
    let batches: Vec<(Vec<i32>, Vec<f64>)> = (0..ncell)
        .step_by(cap)
        .map(|c0| {
            let cells: Vec<i32> = (c0..(c0 + cap).min(ncell)).map(|c| c as i32 + 1).collect();
            let mut lu = vec![0.0; lay.u.len() * cap];
            for (s, b) in lay.u.iter().enumerate() {
                for (l, c) in cells.iter().enumerate() {
                    lu[s * cap + l] = u[b.base + *c as usize - 1];
                }
            }
            (cells, lu)
        })
        .collect();
    let mut out = vec![0.0f64; plan.out_len()];
    let ev = median(|| {
        for (cells, lu) in &batches {
            prog.eval_into(
                &plan,
                lu,
                &[],
                &p,
                0.5,
                &f,
                cells,
                cells.len(),
                &mut out,
                &mut ws,
            )
            .unwrap();
        }
    });
    let rhs = rhs_time(&src, &u, &p);
    println!(
        "{name:<28} lanes cap {cap:<4}            | eval_into {} over {} batches | {:.2}× RHS",
        us(ev),
        batches.len(),
        ev / rhs
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
    domain("  LaneBlocks (domain)", &chem, Layout::LaneBlocks);
    domain(
        "  LaneSparseBlocks (domain)",
        &chem,
        Layout::LaneSparseBlocks,
    );
    domain("  Csr (domain)", &chem, Layout::Csr);
    lanes("  LaneBlocks", &chem, Layout::LaneBlocks, 256);
    lanes("  LaneSparseBlocks", &chem, Layout::LaneSparseBlocks, 256);
    let nz = 72;
    let nx = (n / nz).max(1);
    let col = earthsci_ast::load_document(&column(nx, nz)).unwrap();
    println!("Column: {nx} columns × {nz} levels × 2 species, vertical exchange");
    domain(
        "  ColumnBanded (domain)",
        &col,
        Layout::ColumnBanded { axis: "z".into() },
    );
    domain("  Csr (domain)", &col, Layout::Csr);
}
