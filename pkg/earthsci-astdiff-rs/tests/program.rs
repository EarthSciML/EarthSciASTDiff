//! Production assembly (#4 W2, W4): the compiled band program against the
//! oracle on every plan fixture, lane programs bitwise against the domain
//! program, determinism across threads and lane orders, the matrix-free
//! products, refusals, and the thread-safety contract.

use std::collections::HashMap;
use std::sync::Arc;

use earthsci_ast::{esm_problem, rhs_program, EsmFile, ProblemOptions, RhsProgram};
use earthsci_astdiff::numeric::Env;
use earthsci_astdiff::{
    band_fields, stable_json, BuildOptions, Exec, JacPlan, JacWorkspace, JacobianProgram, Layout,
    PlanKind,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

// ---------------------------------------------------------------------------
// The static contract.
// ---------------------------------------------------------------------------

const _: () = {
    const fn shared<T: Send + Sync + Clone>() {}
    const fn sendable<T: Send>() {}
    shared::<JacobianProgram>();
    shared::<JacPlan>();
    sendable::<JacWorkspace>();
};

/// Resolves only when `T` is not `Sync`.
trait AmbiguousIfSync<A> {
    fn check() {}
}
impl<T: ?Sized> AmbiguousIfSync<()> for T {}
impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}
const _: fn() = || <JacWorkspace as AmbiguousIfSync<_>>::check();

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn cb(axis: &str) -> Layout {
    Layout::ColumnBanded { axis: axis.into() }
}

/// Kept in step with tests/scatter.rs.
fn plan_layouts() -> Vec<(&'static str, Layout)> {
    vec![
        ("bd_chem", Layout::LaneBlocks),
        ("bd_chem", Layout::LaneSparseBlocks),
        ("bd_chem", cb("z")),
        ("bd_chem", Layout::Csr),
        ("adv_interior", cb("x")),
        ("adv_interior", Layout::Csr),
        ("flux_form_adv", Layout::Csr),
        ("minmod_adv", Layout::Csr),
        ("contracted_ops", Layout::Csr),
        ("contracting_observed", Layout::Csr),
    ]
}

fn source(file: &EsmFile) -> RhsProgram {
    rhs_program(&esm_problem(file, (0.0, 1.0), ProblemOptions::default()).unwrap()).unwrap()
}

fn model_of(file: &EsmFile) -> String {
    file.models.as_ref().unwrap().keys().next().unwrap().clone()
}

/// A small deterministic generator in [lo, hi).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
}

/// Nested (first index outermost) array of a column-major block.
fn nested(vals: &[f64], shape: &[usize]) -> Value {
    fn rec(vals: &[f64], shape: &[usize], strides: &[usize], at: usize) -> Value {
        match shape.split_first() {
            None => json!(vals[at]),
            Some((n, rest)) => Value::Array(
                (0..*n)
                    .map(|i| rec(vals, rest, &strides[1..], at + i * strides[0]))
                    .collect(),
            ),
        }
    }
    let mut strides = Vec::new();
    let mut acc = 1;
    for n in shape {
        strides.push(acc);
        acc *= n;
    }
    rec(vals, shape, &strides, 0)
}

/// The oracle environment of a whole-domain point of the source program.
fn env_of(src: &RhsProgram, model: &str, u: &[f64], aux: &[f64], p: &[f64], t: f64) -> Env {
    let strip = |n: &str| {
        n.strip_prefix(&format!("{model}."))
            .unwrap_or(n)
            .to_string()
    };
    let lay = src.layout();
    let mut states = HashMap::new();
    for (blocks, buf) in [(&lay.u, u), (&lay.aux, aux)] {
        for b in blocks {
            let shape: Vec<usize> = b
                .storage
                .iter()
                .map(|(lo, hi)| (hi - lo + 1) as usize)
                .collect();
            let v = &buf[b.base..b.base + b.len()];
            states.insert(
                strip(&b.name),
                if shape.is_empty() {
                    json!(v[0])
                } else {
                    nested(v, &shape)
                },
            );
        }
    }
    let params = src
        .parameters()
        .iter()
        .zip(p)
        .map(|(s, x)| (strip(&s.name), json!(x)))
        .collect();
    Env { states, params, t }
}

/// A point of the source program: the jvals golden's parameters and `t`
/// (when there is a golden), random states.
fn point(src: &RhsProgram, name: &str, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let mut g = Lcg(seed);
    let golden: Option<Value> = std::fs::read_to_string(
        root()
            .join("tests/goldens")
            .join(format!("{name}.jvals.json")),
    )
    .ok()
    .map(|s| serde_json::from_str(&s).unwrap());
    let model_params: HashMap<String, f64> = golden
        .as_ref()
        .map(|g| {
            g["params"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                .collect()
        })
        .unwrap_or_default();
    let p: Vec<f64> = src
        .parameters()
        .iter()
        .map(|s| {
            let bare = s.name.rsplit('.').next().unwrap();
            model_params.get(bare).copied().or(s.default).unwrap_or(0.5)
        })
        .collect();
    let u = (0..src.n_states()).map(|_| g.next(0.2, 1.3)).collect();
    let aux = (0..src.layout().aux_len)
        .map(|_| g.next(0.2, 1.3))
        .collect();
    let t = golden.map(|g| g["t"].as_f64().unwrap()).unwrap_or(0.7) + g.next(0.0, 0.3);
    (u, aux, p, t)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-13 * b.abs().max(1.0)
}

// ---------------------------------------------------------------------------
// Values.
// ---------------------------------------------------------------------------

#[test]
fn evaluation_document_is_the_golden() {
    for name in [
        "bd_chem",
        "adv_interior",
        "minmod_adv",
        "contracting_observed",
        "forcing_bracket",
    ] {
        let prog = JacobianProgram::build(&load(name), None, &BuildOptions::default()).unwrap();
        let golden = std::fs::read_to_string(
            root()
                .join("tests/goldens")
                .join(format!("{name}.evaldoc.json")),
        )
        .unwrap();
        assert_eq!(stable_json(prog.evaluation_document()), golden, "{name}");
    }
}

/// Domain plans on every plan fixture, at four random points, against the
/// oracle's band fields scattered by the checked plan.
#[test]
fn domain_plans_match_the_oracle() {
    for (name, layout) in plan_layouts() {
        let file = load(name);
        let model = model_of(&file);
        let src = source(&file);
        for cse in [true, false] {
            let opts = BuildOptions {
                cse,
                ..Default::default()
            };
            let prog = JacobianProgram::build(&file, None, &opts).unwrap();
            let plan = prog.plan(&layout, &src, Exec::Domain).unwrap();
            let bound = plan.scatter().bind(&file, None, plan.capacity()).unwrap();
            let mut ws = prog.new_workspace(&plan);
            let forcing = src.default_forcing();
            for seed in 1..=4 {
                let (u, aux, p, t) = point(&src, name, seed);
                let mut out = vec![f64::NAN; plan.out_len()];
                prog.eval_into(&plan, &u, &aux, &p, t, &forcing, &[], 0, &mut out, &mut ws)
                    .unwrap();
                let env = env_of(&src, &model, &u, &aux, &p, t);
                let fields = band_fields(&file, None, &env).unwrap();
                let mut want = vec![0.0; bound.len];
                bound.scatter(&fields, &mut want).unwrap();
                for (i, (a, b)) in out.iter().zip(&want).enumerate() {
                    assert!(
                        close(*a, *b),
                        "{name} {layout:?} cse={cse} seed {seed} [{i}]: \
                                            {a} vs {b}"
                    );
                }
                // f32 storage: the same sums, rounded once
                let mut o32 = vec![f32::NAN; plan.out_len()];
                prog.eval_into(&plan, &u, &aux, &p, t, &forcing, &[], 0, &mut o32, &mut ws)
                    .unwrap();
                assert!(
                    o32.iter().zip(&out).all(|(a, b)| *a == *b as f32),
                    "{name} f32"
                );
            }
        }
    }
}

/// `apply` and `apply_transpose` against the assembled CSR matrix, and the
/// adjoint identity ⟨w, J v⟩ = ⟨Jᵀ w, v⟩.
#[test]
fn matrix_free_products_match_the_assembled_matrix() {
    for (name, layout) in plan_layouts() {
        let file = load(name);
        let model = model_of(&file);
        let src = source(&file);
        let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
        let plan = prog.plan(&layout, &src, Exec::Domain).unwrap();
        let csr = prog.plan(&Layout::Csr, &src, Exec::Domain).unwrap();
        let mut ws = prog.new_workspace(&plan);
        let mut wc = prog.new_workspace(&csr);
        let forcing = src.default_forcing();
        let (u, aux, p, t) = point(&src, name, 11);
        let mut out = vec![0.0; plan.out_len()];
        prog.eval_into(&plan, &u, &aux, &p, t, &forcing, &[], 0, &mut out, &mut ws)
            .unwrap();
        let mut vals = vec![0.0; csr.out_len()];
        prog.eval_into(&csr, &u, &aux, &p, t, &forcing, &[], 0, &mut vals, &mut wc)
            .unwrap();
        // canonical state cell → position in the source's u ++ aux
        let lay = src.layout();
        let mut at: HashMap<String, (usize, usize)> = HashMap::new();
        for (blocks, base) in [(&lay.u, 0), (&lay.aux, lay.u_len)] {
            for b in blocks {
                at.insert(
                    b.name
                        .strip_prefix(&format!("{model}."))
                        .unwrap()
                        .to_string(),
                    (base + b.base, b.len()),
                );
            }
        }
        let mut canon = Vec::new();
        for (s, shape) in &csr.scatter().states {
            let (b, n) = at[s];
            assert_eq!(n, shape.iter().product::<i64>().max(1) as usize);
            canon.extend(b..b + n);
        }
        let PlanKind::Csr { rowptr, colind, .. } = &csr.scatter().kind else {
            unreachable!()
        };
        let (nu, na) = plan.source_lengths();
        let mut g = Lcg(5);
        let v: Vec<f64> = (0..nu + na).map(|_| g.next(-1.0, 1.0)).collect();
        let w: Vec<f64> = (0..nu).map(|_| g.next(-1.0, 1.0)).collect();
        let (mut jv, mut jtw) = (vec![0.0; nu], vec![0.0; nu + na]);
        for r in 0..rowptr.len() - 1 {
            for q in rowptr[r]..rowptr[r + 1] {
                jv[canon[r]] += vals[q] * v[canon[colind[q]]];
                jtw[canon[colind[q]]] += vals[q] * w[canon[r]];
            }
        }
        let mut y = vec![0.0; nu];
        prog.apply(&plan, &ws, &v, &mut y).unwrap();
        let mut gb = vec![0.0; nu + na];
        prog.apply_transpose(&plan, &ws, &w, &mut gb).unwrap();
        for (a, b) in y.iter().zip(&jv).chain(gb.iter().zip(&jtw)) {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "{name} {layout:?}: {a} vs {b}"
            );
        }
        let lhs: f64 = w.iter().zip(&y).map(|(a, b)| a * b).sum();
        let rhs: f64 = gb.iter().zip(&v).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() <= 1e-12 * rhs.abs().max(1.0),
            "{name}: adjoint identity"
        );
        // `+=`: a second product doubles
        prog.apply(&plan, &ws, &v, &mut y).unwrap();
        assert!(y
            .iter()
            .zip(&jv)
            .all(|(a, b)| (a - 2.0 * b).abs() <= 1e-12 * b.abs().max(1.0)));
    }
}

// ---------------------------------------------------------------------------
// Lanes.
// ---------------------------------------------------------------------------

/// `bd_chem` on `nz` cells.
fn chem(nz: i64) -> EsmFile {
    let mut d: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("tests/valid/bd_chem.esm")).unwrap(),
    )
    .unwrap();
    d["index_sets"]["z"]["size"] = json!(nz);
    earthsci_ast::load_document(&d).unwrap()
}

/// Lane `ℓ` of a lane plan equals the domain plan's lane `cells[ℓ] - 1`,
/// bitwise, whatever the lane order, capacity and padding.
#[test]
fn lane_plans_are_bitwise_the_domain_plan() {
    let file = chem(9);
    let src = source(&file);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    let forcing = src.default_forcing();
    let (u, aux, p, t) = point(&src, "bd_chem", 3);
    for layout in [Layout::LaneBlocks, Layout::LaneSparseBlocks] {
        let dom = prog.plan(&layout, &src, Exec::Domain).unwrap();
        let mut wd = prog.new_workspace(&dom);
        let mut whole = vec![0.0f64; dom.out_len()];
        prog.eval_into(&dom, &u, &aux, &p, t, &forcing, &[], 0, &mut whole, &mut wd)
            .unwrap();
        let nslot = dom.out_len() / 9;
        for (cap, cells) in [
            (9, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (4, vec![7, 2, 9]),
            (16, vec![5, 5, 1, 9, 3]),
        ] {
            let plan = prog
                .plan(&layout, &src, Exec::Lanes { capacity: cap })
                .unwrap();
            assert_eq!(plan.out_len(), nslot * cap);
            // lane inputs: species-major, lanes fastest, gathered per cell
            let (nu, na) = plan.source_lengths();
            assert_eq!(na, 0);
            let mut lu = vec![0.0; nu];
            for (s, b) in src.layout().u.iter().enumerate() {
                for (l, c) in cells.iter().enumerate() {
                    lu[s * cap + l] = u[b.base + (*c as usize - 1)];
                }
            }
            let mut ws = prog.new_workspace(&plan);
            let mut out = vec![f64::NAN; plan.out_len()];
            prog.eval_into(
                &plan,
                &lu,
                &[],
                &p,
                t,
                &forcing,
                &cells,
                cells.len(),
                &mut out,
                &mut ws,
            )
            .unwrap();
            for q in 0..nslot {
                for (l, c) in cells.iter().enumerate() {
                    assert_eq!(
                        out[q * cap + l].to_bits(),
                        whole[q * 9 + (*c as usize - 1)].to_bits(),
                        "{layout:?} cap {cap} slot {q} lane {l}"
                    );
                }
            }
            // matrix-free over lanes: each lane its own block
            let v: Vec<f64> = (0..nu).map(|i| (i as f64 * 0.37).sin()).collect();
            let mut y = vec![0.0; nu];
            prog.apply(&plan, &ws, &v, &mut y).unwrap();
            let mut gb = vec![0.0; nu];
            prog.apply_transpose(&plan, &ws, &v, &mut gb).unwrap();
            let lhs: f64 = v.iter().zip(&y).map(|(a, b)| a * b).sum();
            let rhs: f64 = gb.iter().zip(&v).map(|(a, b)| a * b).sum();
            assert!((lhs - rhs).abs() <= 1e-12 * rhs.abs().max(1.0));
        }
    }
}

/// A two-dimensional column operator: per-cell chemistry plus (with
/// `exchange`) an upward exchange along `z`, so its Jacobian is banded per
/// column.
fn column_doc(nx: i64, nz: i64, exchange: bool) -> EsmFile {
    let ix = |v: &str, a: Value, b: Value| json!({"op": "index", "args": [v, a, b]});
    let kp1 = json!({"op": "+", "args": ["k", 1]});
    let d = json!({
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
                     "expr": {"op": "+", "args": [
                        {"op": "*", "args": [-1, "k1", ix("A", json!("i"), json!("k")),
                                             ix("B", json!("i"), json!("k"))]},
                        {"op": "*", "args": ["kz", {"op": "-", "args": [
                            ix("A", json!("i"), kp1.clone()), ix("A", json!("i"), json!("k"))]}]}]}},
                    {"op": "faq", "output_idx": ["i", "k"], "args": [],
                     "ranges": {"i": {"from": "x"}, "k": [nz, nz]},
                     "expr": {"op": "*", "args": [-1, "k1", ix("A", json!("i"), json!("k")),
                                                  ix("B", json!("i"), json!("k"))]}}]}},
                {"lhs": {"op": "D", "args": ["B"], "wrt": "t"},
                 "rhs": {"op": "faq", "output_idx": ["i", "k"], "args": [],
                         "ranges": {"i": {"from": "x"}, "k": {"from": "z"}},
                         "expr": {"op": "*", "args": ["k1", ix("A", json!("i"), json!("k")),
                                                      ix("B", json!("i"), json!("k"))]}}}
            ]
        }}
    });
    let mut d = d;
    if !exchange {
        d["models"]["Col"]["equations"][0]["rhs"] =
            d["models"]["Col"]["equations"][1]["rhs"].clone();
        d["models"]["Col"]["equations"][0]["rhs"]["expr"]["args"][0] = json!("kz");
    }
    earthsci_ast::load_document(&d).unwrap()
}

/// The exchange operator: the domain plan against the oracle. Its bands stop
/// short of the column ends, which a column lane program cannot restrict yet
/// (earthsci-ast refuses a `faq` region value there), so lanes are refused.
#[test]
fn column_domain_plan_matches_the_oracle() {
    let (nx, nz) = (6usize, 5usize);
    let file = column_doc(nx as i64, nz as i64, true);
    let src = source(&file);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    let (u, aux, p, t) = point(&src, "column", 8);
    let dom = prog.plan(&cb("z"), &src, Exec::Domain).unwrap();
    assert_eq!(dom.capacity(), nx);
    let PlanKind::ColumnBanded { kl, ku, .. } = &dom.scatter().kind else {
        unreachable!()
    };
    assert_eq!((*kl, *ku), (0, 1));
    let mut wd = prog.new_workspace(&dom);
    let mut whole = vec![0.0f64; dom.out_len()];
    prog.eval_into(
        &dom,
        &u,
        &aux,
        &p,
        t,
        &src.default_forcing(),
        &[],
        0,
        &mut whole,
        &mut wd,
    )
    .unwrap();
    let bound = dom.scatter().bind(&file, None, nx).unwrap();
    let fields = band_fields(&file, None, &env_of(&src, "Col", &u, &aux, &p, t)).unwrap();
    let mut want = vec![0.0; bound.len];
    bound.scatter(&fields, &mut want).unwrap();
    assert!(whole.iter().zip(&want).all(|(a, b)| close(*a, *b)));
    assert!(prog
        .plan(&cb("z"), &src, Exec::Lanes { capacity: 4 })
        .is_err());
    // per-cell coupling only: also column-banded along `x`, the FIRST
    // dimension (the domain plan's per-contribution path), and per cell
    let file = column_doc(nx as i64, nz as i64, false);
    let src = source(&file);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    for layout in [
        cb("x"),
        cb("z"),
        Layout::LaneBlocks,
        Layout::LaneSparseBlocks,
        Layout::Csr,
    ] {
        let dom = prog.plan(&layout, &src, Exec::Domain).unwrap();
        let mut wd = prog.new_workspace(&dom);
        let mut got = vec![f64::NAN; dom.out_len()];
        prog.eval_into(
            &dom,
            &u,
            &aux,
            &p,
            t,
            &src.default_forcing(),
            &[],
            0,
            &mut got,
            &mut wd,
        )
        .unwrap();
        let bound = dom.scatter().bind(&file, None, dom.capacity()).unwrap();
        let fields = band_fields(&file, None, &env_of(&src, "Col", &u, &aux, &p, t)).unwrap();
        let mut want = vec![0.0; bound.len];
        bound.scatter(&fields, &mut want).unwrap();
        assert!(
            got.iter().zip(&want).all(|(a, b)| close(*a, *b)),
            "{layout:?}"
        );
    }
}

#[test]
fn column_lanes_are_bitwise_the_domain_plan() {
    let (nx, nz) = (6usize, 5usize);
    let file = column_doc(nx as i64, nz as i64, false);
    let src = source(&file);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    let forcing = src.default_forcing();
    let (u, aux, p, t) = point(&src, "column", 8);
    let layout = cb("z");
    let dom = prog.plan(&layout, &src, Exec::Domain).unwrap();
    assert_eq!(dom.capacity(), nx);
    let mut wd = prog.new_workspace(&dom);
    let mut whole = vec![0.0f64; dom.out_len()];
    prog.eval_into(&dom, &u, &aux, &p, t, &forcing, &[], 0, &mut whole, &mut wd)
        .unwrap();
    let cap = 4;
    let cols = [5, 2, 6];
    let plan = prog
        .plan(&layout, &src, Exec::Lanes { capacity: cap })
        .unwrap();
    let nslot = dom.out_len() / nx;
    let (nu, _) = plan.source_lengths();
    // source lane layout: per state a [lane, z] block
    let mut lu = vec![0.0; nu];
    for (s, b) in src.layout().u.iter().enumerate() {
        for (l, c) in cols.iter().enumerate() {
            for k in 0..nz {
                lu[s * cap * nz + k * cap + l] = u[b.base + (*c as usize - 1) + nx * k];
            }
        }
    }
    let mut ws = prog.new_workspace(&plan);
    let mut out = vec![f64::NAN; plan.out_len()];
    prog.eval_into(
        &plan,
        &lu,
        &[],
        &p,
        t,
        &forcing,
        &cols,
        cols.len(),
        &mut out,
        &mut ws,
    )
    .unwrap();
    for q in 0..nslot {
        for (l, c) in cols.iter().enumerate() {
            assert_eq!(
                out[q * cap + l].to_bits(),
                whole[q * nx + (*c as usize - 1)].to_bits(),
                "slot {q} lane {l}"
            );
        }
    }
}

/// Assembly is bitwise invariant to thread count and lane order: every
/// thread owns whole batches of lanes, with its own workspace.
#[test]
fn deterministic_across_threads_and_lane_orders() {
    let nz = 500usize;
    let file = chem(nz as i64);
    let src = source(&file);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    let forcing = src.default_forcing();
    let (u, _, p, t) = point(&src, "bd_chem", 21);
    let cap = 32;
    let plan = prog
        .plan(&Layout::LaneBlocks, &src, Exec::Lanes { capacity: cap })
        .unwrap();
    let nslot = plan.out_len() / cap;
    let nstate = src.layout().u.len();
    let bases: Vec<usize> = src.layout().u.iter().map(|b| b.base).collect();
    // per-cell blocks, by cell, for a lane order split over `threads`
    let run = |order: &[usize], threads: usize| -> Vec<Vec<u64>> {
        let batches: Vec<&[usize]> = order.chunks(cap).collect();
        let per = batches.len().div_ceil(threads);
        let mut blocks = vec![Vec::new(); nz];
        std::thread::scope(|s| {
            let hs: Vec<_> = batches
                .chunks(per)
                .map(|mine| {
                    let (prog, plan, forcing, u, p, bases) =
                        (&prog, &plan, &forcing, &u, &p, &bases);
                    s.spawn(move || {
                        let mut ws = prog.new_workspace(plan);
                        let mut res = Vec::new();
                        for batch in mine {
                            let cells: Vec<i32> = batch.iter().map(|c| *c as i32 + 1).collect();
                            let mut lu = vec![0.0; nstate * cap];
                            for (s, b) in bases.iter().enumerate() {
                                for (l, c) in batch.iter().enumerate() {
                                    lu[s * cap + l] = u[b + c];
                                }
                            }
                            let mut out = vec![0.0f64; plan.out_len()];
                            prog.eval_into(
                                plan,
                                &lu,
                                &[],
                                p,
                                t,
                                forcing,
                                &cells,
                                cells.len(),
                                &mut out,
                                &mut ws,
                            )
                            .unwrap();
                            for (l, c) in batch.iter().enumerate() {
                                res.push((
                                    *c,
                                    (0..nslot)
                                        .map(|q| out[q * cap + l].to_bits())
                                        .collect::<Vec<u64>>(),
                                ));
                            }
                        }
                        res
                    })
                })
                .collect();
            for h in hs {
                for (c, b) in h.join().unwrap() {
                    blocks[c] = b;
                }
            }
        });
        blocks
    };
    let natural: Vec<usize> = (0..nz).collect();
    let reference = run(&natural, 1);
    let mut shuffled = natural.clone();
    let mut g = Lcg(99);
    for i in (1..nz).rev() {
        shuffled.swap(i, (g.next(0.0, 1.0) * (i + 1) as f64) as usize);
    }
    for threads in [2, 7, 16] {
        assert_eq!(run(&natural, threads), reference, "{threads} threads");
        assert_eq!(
            run(&shuffled, threads),
            reference,
            "{threads} threads, shuffled"
        );
    }
}

// ---------------------------------------------------------------------------
// Refusals.
// ---------------------------------------------------------------------------

#[test]
fn refusals() {
    let file = load("bd_chem");
    let src = source(&file);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    // a source program from another document
    let adv = load("adv_interior");
    let other = source(&adv);
    assert!(prog.plan(&Layout::Csr, &other, Exec::Domain).is_err());
    assert!(prog
        .plan(&Layout::LaneBlocks, &other, Exec::Lanes { capacity: 4 })
        .is_err());
    // the same states at another extent (another build of the document)
    let big = source(&chem(8));
    assert!(prog.plan(&Layout::Csr, &big, Exec::Domain).is_err());
    // Csr has no lane form; a lane capacity of 0
    assert!(prog
        .plan(&Layout::Csr, &src, Exec::Lanes { capacity: 4 })
        .is_err());
    assert!(prog
        .plan(&Layout::LaneBlocks, &src, Exec::Lanes { capacity: 0 })
        .is_err());
    // a stencil's per-cell layouts (not block-diagonal)
    let aprog = JacobianProgram::build(&adv, None, &BuildOptions::default()).unwrap();
    assert!(aprog
        .plan(&Layout::LaneBlocks, &other, Exec::Domain)
        .is_err());
    // a stencil whose band regions depend on the lane's cell does not lane
    assert!(aprog
        .plan(&cb("x"), &other, Exec::Lanes { capacity: 4 })
        .is_err());
    // an empty band list: a document with array states and no coupling
    let mut d: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("tests/valid/bd_chem.esm")).unwrap(),
    )
    .unwrap();
    for eq in d["models"]["Chem"]["equations"].as_array_mut().unwrap() {
        eq["rhs"] = json!({"op": "faq", "output_idx": ["i"], "args": [],
                           "ranges": {"i": {"from": "z"}}, "expr": "k1"});
    }
    assert!(JacobianProgram::build(
        &earthsci_ast::load_document(&d).unwrap(),
        None,
        &BuildOptions::default()
    )
    .is_err());
    // call-time checks
    let plan = prog.plan(&Layout::Csr, &src, Exec::Domain).unwrap();
    let forcing = src.default_forcing();
    let (u, aux, p, t) = point(&src, "bd_chem", 1);
    let mut ws = prog.new_workspace(&plan);
    let mut out = vec![0.0; plan.out_len()];
    let y = &mut vec![0.0; u.len()];
    // no slab before a successful evaluation
    assert!(prog.apply(&plan, &ws, &u, y).is_err());
    assert!(prog
        .eval_into(
            &plan,
            &u[1..],
            &aux,
            &p,
            t,
            &forcing,
            &[],
            0,
            &mut out,
            &mut ws
        )
        .is_err());
    assert!(prog
        .eval_into(
            &plan,
            &u,
            &aux,
            &p[1..],
            t,
            &forcing,
            &[],
            0,
            &mut out,
            &mut ws
        )
        .is_err());
    assert!(prog
        .eval_into(&plan, &u, &aux, &p, t, &forcing, &[1], 1, &mut out, &mut ws)
        .is_err());
    let mut short = vec![0.0; plan.out_len() - 1];
    assert!(prog
        .eval_into(
            &plan,
            &u,
            &aux,
            &p,
            t,
            &forcing,
            &[],
            0,
            &mut short,
            &mut ws
        )
        .is_err());
    // a workspace of another plan, a plan of another program
    let plan2 = prog.plan(&Layout::LaneBlocks, &src, Exec::Domain).unwrap();
    let mut ws2 = prog.new_workspace(&plan2);
    assert!(prog
        .eval_into(&plan, &u, &aux, &p, t, &forcing, &[], 0, &mut out, &mut ws2)
        .is_err());
    let prog2 = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    assert!(prog2
        .eval_into(&plan, &u, &aux, &p, t, &forcing, &[], 0, &mut out, &mut ws)
        .is_err());
    prog.eval_into(&plan, &u, &aux, &p, t, &forcing, &[], 0, &mut out, &mut ws)
        .unwrap();
    assert!(prog.apply(&plan, &ws, &u, y).is_ok());
    assert!(prog.apply(&plan, &ws, &u[1..], y).is_err());
}

/// A forcing parameter stays a forcing slot of the band program, read from
/// the integrator's own set: a refreshed buffer moves the Jacobian.
#[test]
fn forcing_reaches_the_band_fields() {
    let refresh =
        json!({"kind": "schedule", "interval": 60.0, "handler": {"handler_id": "reload"}});
    let d = json!({
        "esm": "1.2.0",
        "metadata": {"name": "forced", "authors": ["test"]},
        "index_sets": {"x": {"kind": "interval", "size": 4}},
        "models": {"M": {
            "variables": {
                "u": {"type": "unknown", "units": "1", "shape": ["x"], "default": 1.0},
                "F": {"type": "parameter", "units": "1", "shape": ["x"], "default": 0.25,
                      "update": refresh}
            },
            "equations": [{
                "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                "rhs": {"op": "faq", "args": [], "output_idx": ["i"], "ranges": {"i": {"from": "x"}},
                        "expr": {"op": "*", "args": [-1, {"op": "index", "args": ["F", "i"]},
                                 {"op": "index", "args": ["u", "i"]},
                                 {"op": "index", "args": ["u", "i"]}]}}
            }]
        }}
    });
    let file = earthsci_ast::load_document(&d).unwrap();
    let src = source(&file);
    assert_eq!(src.forcing_slots().len(), 1);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    let plan = prog.plan(&Layout::LaneBlocks, &src, Exec::Domain).unwrap();
    let mut ws = prog.new_workspace(&plan);
    let u = vec![1.0, 2.0, 3.0, 4.0];
    let p = src.parameter_vector(&HashMap::new()).unwrap();
    let f0 = src.default_forcing();
    let mut out = vec![0.0; plan.out_len()];
    prog.eval_into(&plan, &u, &[], &p, 0.0, &f0, &[], 0, &mut out, &mut ws)
        .unwrap();
    assert_eq!(out, vec![-0.5, -1.0, -1.5, -2.0]);
    let f1 = f0.replace(0, Arc::from(vec![1.0, 2.0, 3.0, 4.0])).unwrap();
    prog.eval_into(&plan, &u, &[], &p, 0.0, &f1, &[], 0, &mut out, &mut ws)
        .unwrap();
    assert_eq!(out, vec![-2.0, -8.0, -18.0, -32.0]);
}

/// A band row fixed along an extent-1 dimension spans the field, but its
/// `faq` leaves the fixed dimension out: it stays a `makearray` (§ fused
/// band form) and evaluates like any other.
#[test]
fn fixed_row_over_an_extent_one_dimension() {
    let d = json!({
        "esm": "1.2.0",
        "metadata": {"name": "pinned", "authors": ["test"]},
        "index_sets": {"x": {"kind": "interval", "size": 1},
                       "y": {"kind": "interval", "size": 3}},
        "models": {"M": {
            "variables": {
                "u": {"type": "unknown", "units": "1", "shape": ["x", "y"], "default": 1.0},
                "k": {"type": "parameter", "units": "1", "default": 0.5}
            },
            "equations": [{
                "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                "rhs": {"op": "makearray", "args": [], "regions": [[[1, 1], [1, 3]]],
                        "values": [{"op": "faq", "args": [], "output_idx": ["i", "j"],
                                    "ranges": {"i": [1, 1], "j": [1, 3]},
                                    "expr": {"op": "*", "args": [-1, "k",
                                             {"op": "index", "args": ["u", 1, "j"]},
                                             {"op": "index", "args": ["u", 1, "j"]}]}}]}
            }]
        }}
    });
    let file = earthsci_ast::load_document(&d).unwrap();
    let src = source(&file);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    let plan = prog.plan(&Layout::Csr, &src, Exec::Domain).unwrap();
    let mut ws = prog.new_workspace(&plan);
    let p = src.parameter_vector(&HashMap::new()).unwrap();
    let mut out = vec![0.0; plan.out_len()];
    prog.eval_into(
        &plan,
        &[1.0, 2.0, 3.0],
        &[],
        &p,
        0.0,
        &src.default_forcing(),
        &[],
        0,
        &mut out,
        &mut ws,
    )
    .unwrap();
    assert_eq!(out, vec![-1.0, -2.0, -3.0]);
}

/// A forcing set for another program's slots is refused (an error, never an
/// out-of-bounds slot), here for a source with a slot the Jacobian does not
/// read.
#[test]
fn a_foreign_forcing_set_is_refused() {
    let refresh =
        json!({"kind": "schedule", "interval": 60.0, "handler": {"handler_id": "reload"}});
    let forced = |extra: bool| {
        let mut vars = json!({
            "u": {"type": "unknown", "units": "1", "shape": ["x"], "default": 1.0},
            "F": {"type": "parameter", "units": "1", "shape": ["x"], "default": 0.25,
                  "update": refresh}
        });
        let mut rhs = json!({"op": "*", "args": [-1, {"op": "index", "args": ["F", "i"]},
                             {"op": "index", "args": ["u", "i"]},
                             {"op": "index", "args": ["u", "i"]}]});
        if extra {
            // `A` is a forcing slot ahead of `F` that only `f`, not `J`, reads.
            vars["A"] = json!({"type": "parameter", "units": "1", "shape": ["x"],
                               "default": 0.1, "update": refresh});
            rhs = json!({"op": "+", "args": [rhs, {"op": "index", "args": ["A", "i"]}]});
        }
        earthsci_ast::load_document(&json!({
            "esm": "1.2.0",
            "metadata": {"name": "forced", "authors": ["test"]},
            "index_sets": {"x": {"kind": "interval", "size": 4}},
            "models": {"M": {"variables": vars, "equations": [{
                "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                "rhs": {"op": "faq", "args": [], "output_idx": ["i"],
                        "ranges": {"i": {"from": "x"}}, "expr": rhs}}]}}
        }))
        .unwrap()
    };
    let file = forced(true);
    let src = source(&file);
    assert_eq!(src.forcing_slots().len(), 2);
    let prog = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap();
    let plan = prog.plan(&Layout::LaneBlocks, &src, Exec::Domain).unwrap();
    let mut ws = prog.new_workspace(&plan);
    let u = vec![1.0, 2.0, 3.0, 4.0];
    let p = src.parameter_vector(&HashMap::new()).unwrap();
    let mut out = vec![0.0; plan.out_len()];
    let foreign = source(&forced(false)).default_forcing();
    assert!(prog
        .eval_into(&plan, &u, &[], &p, 0.0, &foreign, &[], 0, &mut out, &mut ws)
        .is_err());
    prog.eval_into(
        &plan,
        &u,
        &[],
        &p,
        0.0,
        &src.default_forcing(),
        &[],
        0,
        &mut out,
        &mut ws,
    )
    .unwrap();
    assert_eq!(out, vec![-0.5, -1.0, -1.5, -2.0]);
}

/// The evaluation document names its band fields `J_k`: a source variable
/// of that name is refused, not shadowed.
#[test]
fn a_source_variable_named_like_a_band_field_is_refused() {
    let mut d: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("tests/valid/bd_chem.esm")).unwrap(),
    )
    .unwrap();
    d["models"]["Chem"]["variables"]["J_1"] =
        json!({"type": "parameter", "units": "1", "default": 0.0});
    let file = earthsci_ast::load_document(&d).unwrap();
    let err = JacobianProgram::build(&file, None, &BuildOptions::default()).unwrap_err();
    assert!(err.to_string().contains("J_1"), "{err}");
}
