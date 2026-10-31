//! Causal recurrences (spec §4.5, §9.7). The goldens this crate writes for
//! `recurrence_sweep` (tests/write_goldens.rs) are checked here against
//! numbers no band of this crate produced: central differences of the
//! SOURCE right-hand side, evaluated by EarthSciAST, which executes the
//! recurrences itself. The three derivations of `J` — the materialized bands
//! through the transfer recurrence (the evaluation document), the tangent
//! sweeps (`JvpEval`) and the adjoint sweeps (`VjpEval`) — agree with each
//! other to rounding, and the second-order contraction (`HvpEval`, both
//! forms) with central differences of `Σ μᵢᵀ J wᵢ`.

use std::collections::{BTreeMap, HashMap};
use std::fs;

use earthsci_astdiff::points::{cellname, foreach_point, outside};
use earthsci_astdiff::{
    ctx_of, evaluation_document, jacobian_bands, jacobian_bands_view, jvp_document, recurrences,
    stable_json, sysview, ProductOptions, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::docvals::{column_cells, doc_cell, doc_jacobian, doc_vjp, split_cell, DocEval, Env};
use common::{jvals_env, load, root};

const NAME: &str = "recurrence_sweep";

fn source(name: &str) -> Value {
    serde_json::from_str(
        &fs::read_to_string(root().join("tests/valid").join(format!("{name}.esm"))).unwrap(),
    )
    .unwrap()
}

fn golden_entries(golden: &Value) -> BTreeMap<(String, String), f64> {
    golden["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                (
                    e[0].as_str().unwrap().to_string(),
                    e[1].as_str().unwrap().to_string(),
                ),
                e[2].as_f64().unwrap(),
            )
        })
        .collect()
}

/// The value of state cell `cell` at `env`.
fn state_value(env: &Env, cell: &str) -> f64 {
    let (v, idx) = split_cell(cell);
    let mut x = &env.states[v];
    for i in idx {
        x = &x[i - 1];
    }
    x.as_f64().unwrap()
}

#[test]
fn jacobian_matches_central_differences_of_the_source() {
    let (env, g) = jvals_env(NAME);
    let want = golden_entries(&g);
    let ev = DocEval::new(&source(NAME), "RecurrenceSweep", &env);
    let file = load(NAME);
    let sv = sysview(&file, None).unwrap();
    let cells = column_cells(&sv, Wrt::States, None).unwrap();
    for col in &cells {
        let h = 1e-6;
        let x0 = state_value(&env, col);
        let up = ev.call(&HashMap::from([(col.clone(), x0 + h)]));
        let dn = ev.call(&HashMap::from([(col.clone(), x0 - h)]));
        for row in &cells {
            let fd = (up.at(row) - dn.at(row)) / (2.0 * h);
            match want.get(&(row.clone(), col.clone())) {
                Some(j) => assert!(
                    (j - fd).abs() <= 1e-7 * (1.0 + fd.abs()),
                    "J[{row}, {col}] = {j}, central difference {fd}"
                ),
                // off the structural pattern the derivative is exactly zero
                None => assert!(fd.abs() < 1e-9, "J[{row}, {col}]: {fd} off the pattern"),
            }
        }
    }
}

#[test]
fn parameter_jacobian_matches_central_differences_of_the_source() {
    let (env, g) = jvals_env(&format!("{NAME}.parameters"));
    let want = golden_entries(&g);
    let src = source(NAME);
    let file = load(NAME);
    let sv = sysview(&file, None).unwrap();
    let cells = column_cells(&sv, Wrt::States, None).unwrap();
    let eps = env.params["eps"].as_f64().unwrap();
    let h = 1e-6;
    let at = |e: f64| {
        let mut env2 = env.clone();
        env2.params.insert("eps".into(), json!(e));
        DocEval::new(&src, "RecurrenceSweep", &env2)
    };
    let (up, dn) = (at(eps + h), at(eps - h));
    let (du, dd) = (up.call(&HashMap::new()), dn.call(&HashMap::new()));
    for row in &cells {
        let fd = (du.at(row) - dd.at(row)) / (2.0 * h);
        match want.get(&(row.clone(), "eps".to_string())) {
            Some(j) => assert!(
                (j - fd).abs() <= 1e-7 * (1.0 + fd.abs()),
                "{row}: {j} vs {fd}"
            ),
            None => assert!(fd.abs() < 1e-9, "{row}: {fd} off the pattern"),
        }
    }
}

/// `J` assembled from the evaluation document's band fields (the
/// materialized bands through the transfer recurrence, spec §4.5).
fn materialized(name: &str, wrt: Wrt, env: &Env) -> BTreeMap<(String, String), f64> {
    materialized_of(&load(name), wrt, env)
}

fn materialized_of(
    file: &earthsci_ast::EsmFile,
    wrt: Wrt,
    env: &Env,
) -> BTreeMap<(String, String), f64> {
    let sv = sysview(file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands_view(&sv, wrt, None).unwrap();
    let (doc, names) = evaluation_document(&sv, &entries, true, 12).unwrap();
    let ev = DocEval::new(&doc, "JacobianEval", env);
    let du = ev.call(&HashMap::new());
    let mut out = BTreeMap::new();
    for (k, en) in entries.iter().enumerate() {
        let field = du.block(&names[k]);
        let mut fshape = ctx.shapes.get(&en.u).cloned().unwrap_or_default();
        fshape.extend(en.band.contracted.iter().map(|c| c.2));
        foreach_point(&en.band, &mut |cell, cc, _, cidx| {
            if outside(ctx.shapes.get(&en.v), cidx) {
                return Ok(());
            }
            let (mut f, mut st) = (0usize, 1usize);
            for (i, n) in cell.iter().chain(cc).zip(&fshape) {
                f += (*i as usize - 1) * st;
                st *= *n as usize;
            }
            *out.entry((cellname(&en.u, cell), cellname(&en.v, cidx)))
                .or_default() += field[f];
            Ok(())
        })
        .unwrap();
    }
    out
}

#[test]
fn the_three_derivations_agree() {
    for (wrt, golden) in [
        (Wrt::States, NAME.to_string()),
        (Wrt::Parameters, format!("{NAME}.parameters")),
    ] {
        let (env, _) = jvals_env(&golden);
        let file = load(NAME);
        let sweep = doc_jacobian(&file, None, wrt, None, &env).unwrap();
        let mat = materialized(NAME, wrt, &env);
        assert_eq!(sweep.len(), mat.len(), "{golden}: key sets");
        for (k, v) in &sweep {
            let m = mat[k];
            assert!(
                (m - v).abs() <= 1e-12 * (1.0 + v.abs()),
                "{golden} {k:?}: {m} vs {v}"
            );
        }
        // ⟨J v, λ⟩ = ⟨v, Jᵀ λ⟩, the tangent sweeps against the adjoint ones
        let sv = sysview(&file, None).unwrap();
        let rows = column_cells(&sv, Wrt::States, None).unwrap();
        let cols = column_cells(&sv, wrt, None).unwrap();
        let lam: HashMap<String, f64> = rows
            .iter()
            .enumerate()
            .map(|(n, c)| (c.clone(), (0.7 * n as f64 + 0.3).sin()))
            .collect();
        let vjp = doc_vjp(&file, None, wrt, None, &env, &lam).unwrap();
        let v: HashMap<String, f64> = cols
            .iter()
            .enumerate()
            .map(|(n, c)| (c.clone(), (0.4 * n as f64 + 1.1).cos()))
            .collect();
        let entries = jacobian_bands_view(&sv, wrt, None).unwrap();
        let (doc, io) = jvp_document(&sv, &entries, wrt, &ProductOptions::default()).unwrap();
        let ev = DocEval::new(&doc, &io.model, &env);
        let inputs = v.iter().map(|(c, x)| (doc_cell(&io.v, c), *x)).collect();
        let du = ev.call(&inputs);
        let jv_lam: f64 = rows
            .iter()
            .map(|r| du.at(&doc_cell(&io.jvp, r)) * lam[r])
            .sum();
        let v_jtl: f64 = cols.iter().map(|c| v[c] * vjp[c]).sum();
        assert!(
            (jv_lam - v_jtl).abs() <= 1e-12 * jv_lam.abs().max(1.0),
            "{golden}: <Jv, λ> = {jv_lam}, <v, Jᵀλ> = {v_jtl}"
        );
    }
}

#[test]
fn contraction_matches_central_differences() {
    use common::docvals::doc_hvp;
    use earthsci_astdiff::HvpForm;
    for wrt in [(Wrt::States, Wrt::States), (Wrt::States, Wrt::Parameters)] {
        let tag = format!("{}-{}", wrt.0.as_str(), wrt.1.as_str());
        let g: Value = serde_json::from_str(
            &fs::read_to_string(
                root()
                    .join("tests/goldens")
                    .join(format!("{NAME}.{tag}.hvals.json")),
            )
            .unwrap(),
        )
        .unwrap();
        let env = common::cells_env(&g["u"], HashMap::new(), g["t"].as_f64().unwrap());
        let num = |m: &Value| -> HashMap<String, f64> {
            m.as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                .collect()
        };
        let pairs: Vec<_> = g["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (num(&p["mu"]), num(&p["w"])))
            .collect();
        let file = load(NAME);
        let sv = sysview(&file, None).unwrap();
        let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
        let (doc, io) =
            jvp_document(&sv, &entries, Wrt::States, &ProductOptions::default()).unwrap();
        // Σᵢ μᵢᵀ J(x, p) wᵢ from the tangent sweeps, at a perturbed point
        let phi = |env: &Env, x: &HashMap<String, f64>| -> f64 {
            let ev = DocEval::new(&doc, &io.model, env);
            let mut tot = 0.0;
            for (mu, w) in &pairs {
                let mut inp = x.clone();
                for (c, v) in w {
                    inp.insert(doc_cell(&io.v, c), *v);
                }
                let du = ev.call(&inp);
                tot += mu
                    .iter()
                    .map(|(r, m)| m * du.at(&doc_cell(&io.jvp, r)))
                    .sum::<f64>();
            }
            tot
        };
        let h = 1e-5;
        let want: Vec<(String, f64)> = g["g"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| (e[0].as_str().unwrap().to_string(), e[1].as_f64().unwrap()))
            .collect();
        for (c, gv) in &want {
            let fd = if wrt.1 == Wrt::States {
                let x0 = state_value(&env, c);
                (phi(&env, &HashMap::from([(c.clone(), x0 + h)]))
                    - phi(&env, &HashMap::from([(c.clone(), x0 - h)])))
                    / (2.0 * h)
            } else {
                let p0 = 0.8; // eps at its document default
                let at = |p: f64| {
                    let mut e = env.clone();
                    e.params.insert(c.clone(), json!(p));
                    phi(&e, &HashMap::new())
                };
                (at(p0 + h) - at(p0 - h)) / (2.0 * h)
            };
            assert!(
                (gv - fd).abs() <= 1e-7 * (1.0 + fd.abs()),
                "{tag} g[{c}] = {gv}, fd {fd}"
            );
        }
        // both forms agree on a states pair
        if wrt == (Wrt::States, Wrt::States) {
            let a = doc_hvp(&file, None, wrt, None, &env, &pairs, HvpForm::Contract).unwrap();
            let b = doc_hvp(&file, None, wrt, None, &env, &pairs, HvpForm::Symmetric).unwrap();
            for (k, v) in &a {
                assert!((b[k] - v).abs() <= 1e-12 * (1.0 + v.abs()), "{k}");
            }
        }
    }
}

// ── small documents ─────────────────────────────────────────────────────────

fn o(op: &str, a: Vec<Value>) -> Value {
    json!({"op": op, "args": a})
}

fn ix(v: &str, a: Vec<Value>) -> Value {
    let mut args = vec![json!(v)];
    args.extend(a);
    o("index", args)
}

/// One state `x[lev]`, one recurrence `V` with body `body` (over `k`),
/// `D(x)[k] = V[k]`.
fn recdoc(body: Value) -> Value {
    json!({"esm": "1.1.0", "metadata": {"name": "R"},
        "index_sets": {"lev": {"kind": "interval", "size": 4}},
        "models": {"R": {"variables": {
            "x": {"type": "unknown", "shape": ["lev"], "default": 0.5},
            "V": {"type": "unknown", "shape": ["lev"]}},
          "equations": [
            {"lhs": "V", "rhs": {"op": "faq", "args": [], "output_idx": ["k"],
                                 "ranges": {"k": {"from": "lev"}}, "expr": body}},
            {"lhs": {"op": "D", "args": ["x"], "wrt": "t"},
             "rhs": {"op": "faq", "args": [], "output_idx": ["i"],
                     "ranges": {"i": {"from": "lev"}}, "expr": ix("V", vec![json!("i")])}}]}}})
}

#[test]
fn a_running_max_tie_takes_the_earlier_argument() {
    // V[k] = max(V[k-1], x[k]): at an exact tie V[k-1] == x[k] the
    // derivative follows the earlier argument, the carried maximum (spec
    // §3.2's left fold), so J[k, k] = 0 and the sweep carries J[k-1, ·]
    let body = o(
        "ifelse",
        vec![
            o("<=", vec![json!("k"), json!(1)]),
            ix("x", vec![json!(1)]),
            o(
                "max",
                vec![
                    ix("V", vec![o("-", vec![json!("k"), json!(1)])]),
                    ix("x", vec![json!("k")]),
                ],
            ),
        ],
    );
    let file = earthsci_ast::load_document(&recdoc(body)).unwrap();
    let env = Env {
        states: HashMap::from([("x".to_string(), json!([0.7, 0.4, 0.7, 0.9]))]),
        params: HashMap::new(),
        t: 0.0,
    };
    let j = doc_jacobian(&file, None, Wrt::States, None, &env).unwrap();
    let at = |r: &str, c: &str| {
        j.get(&(r.to_string(), c.to_string()))
            .copied()
            .unwrap_or(0.0)
    };
    // V = [0.7, 0.7, 0.7 (tie), 0.9]
    assert_eq!(at("x[3]", "x[1]"), 1.0);
    assert_eq!(at("x[3]", "x[3]"), 0.0);
    assert_eq!(at("x[4]", "x[4]"), 1.0);
    assert_eq!(at("x[4]", "x[1]"), 0.0);
}

#[test]
fn ill_founded_recurrences_are_refused() {
    let k = json!("k");
    for (why, body) in [
        (
            "same cell",
            o(
                "+",
                vec![ix("x", vec![k.clone()]), ix("V", vec![k.clone()])],
            ),
        ),
        (
            "bare self-reference",
            o("+", vec![ix("x", vec![k.clone()]), json!("V")]),
        ),
        (
            "not along a frame axis",
            o(
                "+",
                vec![ix("x", vec![k.clone()]), ix("V", vec![json!(1), k.clone()])],
            ),
        ),
        // a read of a later cell: its lag is -1, never "≥ 1", so its sweep
        // term would be gated to a silent zero
        (
            "a later cell",
            o(
                "ifelse",
                vec![
                    o(">=", vec![k.clone(), json!(4)]),
                    ix("x", vec![k.clone()]),
                    ix("V", vec![o("+", vec![k.clone(), json!(1)])]),
                ],
            ),
        ),
    ] {
        // the loader takes the document; the refusal is this crate's
        let file = earthsci_ast::load_document(&recdoc(body)).unwrap();
        let err = jacobian_bands(&file, None, Wrt::States).unwrap_err();
        assert!(err.0.contains("recurrence `V`"), "{why}: {}", err.0);
    }
}

#[test]
fn recognition() {
    let file = load(NAME);
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let obs = earthsci_astdiff::system::observed_defs(&sv).unwrap();
    let recs = recurrences(&sv, &obs, &ctx).unwrap();
    let got: Vec<(&str, usize)> = recs.iter().map(|r| (r.name.as_str(), r.dim)).collect();
    assert_eq!(
        got,
        [
            ("bet", 1),
            ("uf", 1),
            ("yrev", 1),
            ("p", 1),
            ("phi", 1),
            ("r", 2)
        ]
    );
    // the state block is column_block along lev: Thomas makes the c line
    // dense, so the whole plan is
    let block: Value = serde_json::from_str(
        &fs::read_to_string(root().join(format!("tests/goldens/{NAME}.column-lev.block.json")))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(block["structure"], json!("column_block"));
    assert_eq!(block["factorization"]["pattern"], json!("dense"));
    // the iteration axis `it` is not a state axis: r's derivative stays on
    // its cell
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    let s = stable_json(&Value::Array(
        entries
            .iter()
            .map(earthsci_astdiff::serialize_band)
            .collect(),
    ));
    assert!(s.contains("_tr_r"));
}

#[test]
fn rank_one_plan() {
    let file = load("pbl_rank1");
    let opts = earthsci_astdiff::BlockOptions {
        column_axis: Some("lev"),
        ..Default::default()
    };
    let b = earthsci_astdiff::jacobian_block_with(&file, None, Wrt::States, &opts).unwrap();
    let f = &b["factorization"];
    assert_eq!(f["pattern"], json!("diag_plus_rank1"));
    // one rank-one term per species: the O3 and NO mean entries
    let terms: Vec<usize> = f["rank1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["entry"].as_u64().unwrap() as usize)
        .collect();
    let entries = b["entries"].as_array().unwrap();
    for t in &terms {
        let en = &entries[t - 1];
        assert_eq!(en["row"], en["col"]);
        assert_eq!(en["contracted"].as_array().unwrap().len(), 1);
    }
    assert_eq!(terms.len(), 2);
    // without the column axis it is an ordinary general block
    let plain = earthsci_astdiff::jacobian_block(&file, None, Wrt::States, true, 12).unwrap();
    assert_eq!(plain["structure"], json!("general"));
}

/// `v[k] = src[k] + a·v[k − 1]` (`src[1]` at `k = 1`).
fn sweep(v: &str, src: &str, a: f64) -> Value {
    let k = json!("k");
    json!({"lhs": v, "rhs": {"op": "faq", "args": [], "output_idx": ["k"],
    "ranges": {"k": {"from": "lev"}},
    "expr": o("ifelse", vec![
        o("<=", vec![k.clone(), json!(1)]),
        ix(src, vec![json!(1)]),
        o("+", vec![
            ix(src, vec![k.clone()]),
            o("*", vec![json!(a), ix(v, vec![o("-", vec![k.clone(), json!(1)])])]),
        ]),
    ])}})
}

/// `faq{D(u[i])} ~ rhs` over `i` in `[lo, hi]`: one equation of an array
/// state.
fn lifted(u: &str, lo: i64, hi: i64, rhs: Value) -> Value {
    json!({"lhs": {"op": "faq", "args": [], "output_idx": ["i"], "ranges": {"i": [lo, hi]},
                   "expr": {"op": "D", "args": [ix(u, vec![json!("i")])], "wrt": "t"}},
           "rhs": rhs})
}

fn model(variables: Value, equations: Vec<Value>) -> Value {
    json!({"esm": "1.1.0", "metadata": {"name": "R"},
        "index_sets": {"lev": {"kind": "interval", "size": 4}},
        "models": {"R": {"variables": variables, "equations": equations}}})
}

#[test]
fn transfer_names_do_not_depend_on_activity() {
    // `A` (inactive on the state axis: it reads only `p`) sorts before `A_`,
    // and the source already holds `_tr_A`: each recurrence's transfer name
    // is the same in the coefficients and in the `auxiliary` field, so the
    // evaluation document's J is A_'s chain 0.5^(k − j), not A's 0.25^(k − j)
    let doc = model(
        json!({
            "x": {"type": "unknown", "shape": ["lev"], "default": 0.5},
            "p": {"type": "parameter", "shape": ["lev"], "default": 1.0},
            "_tr_A": {"type": "parameter", "default": 0.0},
            "A": {"type": "unknown", "shape": ["lev"]},
            "A_": {"type": "unknown", "shape": ["lev"]}}),
        vec![
            sweep("A", "p", 0.25),
            sweep("A_", "x", 0.5),
            json!({"lhs": {"op": "D", "args": ["x"], "wrt": "t"},
                   "rhs": {"op": "faq", "args": [], "output_idx": ["i"],
                           "ranges": {"i": {"from": "lev"}},
                           "expr": o("+", vec![ix("A_", vec![json!("i")]),
                                               ix("A", vec![json!("i")])])}}),
        ],
    );
    let file = earthsci_ast::load_document(&doc).unwrap();
    let block = earthsci_astdiff::jacobian_block(&file, None, Wrt::States, true, 12).unwrap();
    let read = stable_json(&block["entries"]);
    let aux = block["auxiliary"].as_array().unwrap();
    assert!(!aux.is_empty());
    for a in aux {
        let nm = a["name"].as_str().unwrap();
        assert!(read.contains(&format!("\"{nm}\"")), "{nm} is not read");
    }
    let env = Env {
        states: HashMap::from([("x".to_string(), json!([0.5, 0.5, 0.5, 0.5]))]),
        params: HashMap::new(),
        t: 0.0,
    };
    let j = materialized_of(&file, Wrt::States, &env);
    for k in 1..=4 {
        for c in 1..=4 {
            let want = if c <= k { 0.5f64.powi(k - c) } else { 0.0 };
            let got = j
                .get(&(format!("x[{k}]"), format!("x[{c}]")))
                .copied()
                .unwrap_or(0.0);
            assert!(
                (got - want).abs() <= 1e-12,
                "J[x[{k}], x[{c}]] = {got}, want {want}"
            );
        }
    }
}

#[test]
fn entries_keep_each_equation_of_a_state_apart() {
    // two equations of `x` (its first level, then the rest): the second's
    // direct band on `x` must not move ahead of the first's band on `y`
    let lev = json!({"type": "unknown", "shape": ["lev"], "default": 0.5});
    let doc = model(
        json!({"x": lev, "y": lev, "V": {"type": "unknown", "shape": ["lev"]}}),
        vec![
            sweep("V", "y", 0.5),
            lifted("x", 1, 1, ix("y", vec![json!("i")])),
            lifted(
                "x",
                2,
                4,
                o(
                    "+",
                    vec![
                        o("-", vec![ix("x", vec![json!("i")])]),
                        ix("V", vec![json!("i")]),
                    ],
                ),
            ),
        ],
    );
    let file = earthsci_ast::load_document(&doc).unwrap();
    let cols: Vec<String> = jacobian_bands(&file, None, Wrt::States)
        .unwrap()
        .into_iter()
        .filter(|en| en.u == "x")
        .map(|en| en.v)
        .collect();
    // the first equation's `y`, then the second's `x` and its composites on
    // `y` (merging the two would sort the second's `x` first)
    assert!(cols.len() > 2, "{cols:?}");
    assert_eq!(&cols[..2], ["y", "x"], "{cols:?}");
    assert!(cols[2..].iter().all(|c| c == "y"), "{cols:?}");
}
