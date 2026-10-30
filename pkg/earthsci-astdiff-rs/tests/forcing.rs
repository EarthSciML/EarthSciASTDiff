//! Forcing-buffer and array-parameter columns (spec §4, §4.3): the
//! `wrt = forcing` axis over the cells of live data buffers (record axis
//! included) and array parameters as real columns of `wrt = parameters`.
//! Mirrors the reference's forcing_test.jl; the byte and value goldens are
//! gated in tests/goldens.rs.

use std::collections::HashMap;

use earthsci_ast::EsmFile;
use earthsci_astdiff::{
    assemble_jacobian, assemble_jacobian_with_forcing, column_names, forcing_names, jacobian_bands,
    jacobian_bands_with_forcing, jacobian_block, jacobian_blocks, jacobian_columns,
    parse_jacobian_block, parse_jacobian_blocks, prepare_jacobian, stable_json, sysview, vjp,
    vjp_forcing, BlockWrt, ParsedBlock, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{jvals_env, load, root};

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// A one-model document around `F` (shape [2, 5]) and a state `c` (shape
/// [4]) with the given `D(c)` body over `i`; `extra` adds variables.
fn bracket_doc(body: Value, extra: Value) -> EsmFile {
    let mut vars = json!({
        "c": {"type": "unknown", "shape": ["x"], "default": 0.5},
        "F": {"type": "parameter", "shape": ["rec", "xf"], "default": 1.0},
    });
    for (k, v) in extra.as_object().unwrap() {
        vars[k] = v.clone();
    }
    let doc = json!({
        "esm": "1.1.0", "metadata": {"name": "B"},
        "index_sets": {
            "x": {"kind": "interval", "size": 4},
            "xf": {"kind": "interval", "size": 5},
            "rec": {"kind": "interval", "size": 2},
        },
        "models": {"B": {"variables": vars, "equations": [{
            "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
            "rhs": {"op": "faq", "args": [], "output_idx": ["i"],
                    "ranges": {"i": {"from": "x"}}, "expr": body},
        }]}},
    });
    earthsci_ast::load_document(&doc).expect("document loads")
}

fn ix(v: &str, i: &[Value]) -> Value {
    let mut args = vec![json!(v)];
    args.extend(i.iter().cloned());
    json!({"op": "index", "args": args})
}

fn op(o: &str, a: &[Value]) -> Value {
    json!({"op": o, "args": a})
}

#[test]
fn the_forcing_set() {
    let file = load("forcing_bracket");
    let sv = sysview(&file, None).unwrap();
    // inferred: the array parameters with a `data` update; E (no update) and
    // the scalars stay on the parameters axis
    assert_eq!(forcing_names(&sv, None).unwrap(), s(&["F", "G"]));
    assert_eq!(
        jacobian_columns(&sv, Wrt::Forcing, None).unwrap(),
        vec![("F".to_string(), vec![2, 5]), ("G".to_string(), vec![2, 3])]
    );
    assert_eq!(
        jacobian_columns(&sv, Wrt::Parameters, None).unwrap(),
        vec![
            ("E".to_string(), vec![4]),
            ("dt".to_string(), vec![]),
            ("k".to_string(), vec![]),
            ("t_ref".to_string(), vec![]),
        ]
    );
    // an explicit set (a planner that builds the providers itself)
    let g = s(&["G"]);
    assert_eq!(forcing_names(&sv, Some(&g)).unwrap(), g);
    let cols: Vec<String> = jacobian_columns(&sv, Wrt::Parameters, Some(&g))
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(cols, s(&["E", "F", "dt", "k", "t_ref"]));
    // a name that resolves to no buffer is an error, never an empty axis
    assert!(forcing_names(&sv, Some(&s(&["F", "nope"]))).is_err());
    assert!(jacobian_bands_with_forcing(&file, None, Wrt::Forcing, Some(&s(&["c"]))).is_err());
    // columns exist only on the parameters and forcing axes
    assert!(jacobian_columns(&sv, Wrt::States, None).is_err());
    // canonical ordinal: names sorted, each one's cells column-major
    let names = column_names(&jacobian_columns(&sv, Wrt::Forcing, None).unwrap());
    assert_eq!(names.len(), 16);
    assert_eq!(&names[..3], &s(&["F[1,1]", "F[2,1]", "F[1,2]"]));
    assert_eq!(names[10], "G[1,1]");
}

#[test]
fn the_bracket_shape() {
    // each record is its own column, the blend weight a factor of the
    // coefficient, and the regrid a contracted band over one record
    let fb = jacobian_bands(&load("forcing_bracket"), None, Wrt::Forcing).unwrap();
    let first_col = |u: &str, c: &str| {
        fb.iter().any(|en| {
            en.u == u
                && en.v == "F"
                && stable_json(&earthsci_astdiff::helpers::ser_expr(&en.band.cidx[0])) == c
        })
    };
    assert!(first_col("c", "1") && first_col("c", "2") && first_col("S", "1"));
    assert!(fb
        .iter()
        .any(|en| en.v == "G" && !en.band.contracted.is_empty()));
}

#[test]
fn array_parameter_columns_are_real() {
    let file = load("forcing_bracket");
    let (env, _) = jvals_env("forcing_bracket.parameters");
    let j = assemble_jacobian(&file, None, Wrt::Parameters, &env).unwrap();
    for i in 1..=4 {
        let v = j
            .get(&(format!("c[{i}]"), format!("E[{i}]")))
            .unwrap_or_else(|| panic!("E[{i}] column dropped"));
        assert_eq!(*v, 1.0);
    }
    // the forcing buffers are not parameter columns
    assert!(j
        .keys()
        .all(|(_, c)| !c.starts_with("F[") && !c.starts_with("G[")));
}

#[test]
fn several_axes_in_one_jacobians_entry() {
    let file = load("forcing_bracket");
    let blocks =
        jacobian_blocks(&file, None, &[Wrt::States, Wrt::Forcing], None, true, 12).unwrap();
    let arr = blocks.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["wrt"], "states");
    assert_eq!(arr[1]["wrt"], "forcing");
    let golden = |f: &str| std::fs::read_to_string(root().join("tests/goldens").join(f)).unwrap();
    assert_eq!(stable_json(&arr[0]), golden("forcing_bracket.block.json"));
    assert_eq!(
        stable_json(&arr[1]),
        golden("forcing_bracket.forcing.block.json")
    );
    let parsed = parse_jacobian_blocks(&blocks).unwrap();
    assert_eq!(
        parsed.iter().map(ParsedBlock::wrt).collect::<Vec<_>>(),
        vec![BlockWrt::Axis(Wrt::States), BlockWrt::Axis(Wrt::Forcing)]
    );
    assert_eq!(
        parse_jacobian_blocks(&arr[1]).unwrap()[0].wrt(),
        BlockWrt::Axis(Wrt::Forcing)
    );
    assert!(jacobian_blocks(&file, None, &[Wrt::States, Wrt::States], None, true, 12).is_err());
    assert!(parse_jacobian_blocks(&json!([arr[1], arr[1]])).is_err());
    // the parsed entries are the differentiated ones (templates expanded)
    let direct = jacobian_bands(&file, None, Wrt::Forcing).unwrap();
    let ser = |es: &[earthsci_astdiff::JacEntry]| {
        stable_json(&Value::Array(
            es.iter().map(earthsci_astdiff::serialize_band).collect(),
        ))
    };
    let ParsedBlock::Axis(_, forcing_entries) = &parsed[1] else {
        panic!("a single-axis block parses to first-order entries")
    };
    assert_eq!(ser(forcing_entries), ser(&direct));
    // a states block keeps structure, a forcing block carries columns
    let st = jacobian_block(&file, None, Wrt::States, true, 12).unwrap();
    assert!(st.get("structure").is_some() && st.get("columns").is_none());
}

#[test]
fn vjp_is_the_transposed_scatter() {
    let file = load("forcing_bracket");
    let (env, _) = jvals_env("forcing_bracket.forcing");
    let sv = sysview(&file, None).unwrap();
    // λ over every row cell, deterministic and non-trivial
    let rows = s(&["S", "c[1]", "c[2]", "c[3]", "c[4]"]);
    let lambda: HashMap<String, f64> = rows
        .iter()
        .enumerate()
        .map(|(k, r)| (r.clone(), (1.7 * k as f64 + 0.3).sin()))
        .collect();
    for wrt in [Wrt::Forcing, Wrt::Parameters] {
        let names = column_names(&jacobian_columns(&sv, wrt, None).unwrap());
        let j = assemble_jacobian(&file, None, wrt, &env).unwrap();
        let mut want = vec![0.0; names.len()];
        for ((r, c), v) in &j {
            let k = names.iter().position(|n| n == c).expect("column in space");
            want[k] += v * lambda[r];
        }
        let mut g = vec![0.0; names.len()];
        vjp(&file, None, wrt, None, &env, &lambda, &mut g).unwrap();
        for (a, b) in g.iter().zip(&want) {
            assert!(
                (a - b).abs() <= 1e-13 * b.abs().max(1.0),
                "{wrt:?}: {a} vs {b}"
            );
        }
        // accumulates
        vjp(&file, None, wrt, None, &env, &lambda, &mut g).unwrap();
        for (a, b) in g.iter().zip(&want) {
            assert!((a - 2.0 * b).abs() <= 1e-13 * b.abs().max(1.0));
        }
        // a wrong-length ḡ is refused
        assert!(vjp(&file, None, wrt, None, &env, &lambda, &mut [0.0; 3]).is_err());
        if wrt == Wrt::Forcing {
            // buffer-shaped accumulators, column-major like the ordinal
            let mut acc: HashMap<String, Vec<f64>> = [
                ("F".to_string(), vec![0.0; 10]),
                ("G".to_string(), vec![0.0; 6]),
            ]
            .into();
            vjp_forcing(&file, None, None, &env, &lambda, &mut acc).unwrap();
            let flat: Vec<f64> = acc["F"].iter().chain(&acc["G"]).copied().collect();
            for (a, b) in flat.iter().zip(&want) {
                assert!((a - b).abs() <= 1e-13 * b.abs().max(1.0));
            }
            let mut missing: HashMap<String, Vec<f64>> = [("F".to_string(), vec![0.0; 10])].into();
            assert!(vjp_forcing(&file, None, None, &env, &lambda, &mut missing).is_err());
        }
    }
}

#[test]
fn refused_rather_than_silently_wrong() {
    let f = s(&["F"]);
    // a column index that reads a parameter cannot be placed at build time
    let bad = bracket_doc(
        op(
            "*",
            &[
                ix("c", &[json!("i")]),
                ix("F", &[json!(1), op("+", &[json!("i"), json!("off")])]),
            ],
        ),
        json!({"off": {"type": "parameter", "default": 1}}),
    );
    let err = jacobian_bands_with_forcing(&bad, None, Wrt::Forcing, Some(&f)).unwrap_err();
    assert!(err.0.contains("not closed at build time"), "{}", err.0);
    // a literal offset is closed
    let ok = bracket_doc(
        op(
            "*",
            &[
                ix("c", &[json!("i")]),
                ix("F", &[json!(2), op("+", &[json!("i"), json!(1)])]),
            ],
        ),
        json!({}),
    );
    assert_eq!(
        jacobian_bands_with_forcing(&ok, None, Wrt::Forcing, Some(&f))
            .unwrap()
            .len(),
        1
    );
    // a column outside the buffer's extent is an error, not a ghost skip
    let past = bracket_doc(
        op(
            "*",
            &[
                ix("c", &[json!("i")]),
                ix("F", &[json!(1), op("+", &[json!("i"), json!(2)])]),
            ],
        ),
        json!({}),
    );
    assert_eq!(
        jacobian_bands_with_forcing(&past, None, Wrt::Forcing, Some(&f))
            .unwrap()
            .len(),
        1
    );
    let env = earthsci_astdiff::numeric::Env {
        states: [("c".to_string(), json!([0.1, 0.2, 0.3, 0.4]))].into(),
        params: HashMap::new(),
        t: 0.0,
    };
    let err =
        assemble_jacobian_with_forcing(&past, None, Wrt::Forcing, Some(&f), &env).unwrap_err();
    assert!(err.0.contains("no ghost skip"), "{}", err.0);
    // the same read on the parameters axis (F is not data-fed here) is a
    // ghost read, as on the state axis: i = 4 reads F[1,6]. F's scalar
    // default is broadcast over its shape.
    let p = assemble_jacobian(&past, None, Wrt::Parameters, &env).unwrap();
    let keys: Vec<(String, String)> = p.keys().cloned().collect();
    assert_eq!(
        keys,
        (1..=3)
            .map(|i| (format!("c[{i}]"), format!("F[1,{}]", i + 2)))
            .collect::<Vec<_>>()
    );
    for (i, c) in [0.1, 0.2, 0.3].iter().enumerate() {
        assert_eq!(p[&keys[i]], *c);
    }
    assert_eq!(
        assemble_jacobian_with_forcing(&ok, None, Wrt::Forcing, Some(&f), &env)
            .unwrap()
            .len(),
        4
    );
    // while a state stencil reading past the domain stays a ghost skip
    let ghost = bracket_doc(
        op(
            "*",
            &[json!(-1), ix("c", &[op("+", &[json!("i"), json!(1)])])],
        ),
        json!({}),
    );
    assert_eq!(
        assemble_jacobian(&ghost, None, Wrt::States, &env)
            .unwrap()
            .len(),
        3
    );
    // reads forcing only where the derivative is 0: an empty block, no error
    let flr = bracket_doc(
        op(
            "*",
            &[
                ix("c", &[json!("i")]),
                op("floor", &[ix("F", &[json!(1), json!("i")])]),
            ],
        ),
        json!({}),
    );
    assert!(
        jacobian_bands_with_forcing(&flr, None, Wrt::Forcing, Some(&f))
            .unwrap()
            .is_empty()
    );
    let cond = bracket_doc(
        op(
            "ifelse",
            &[
                op(">", &[ix("F", &[json!(1), json!("i")]), json!(0)]),
                op("*", &[json!(-1), ix("c", &[json!("i")])]),
                json!(0),
            ],
        ),
        json!({}),
    );
    assert!(
        jacobian_bands_with_forcing(&cond, None, Wrt::Forcing, Some(&f))
            .unwrap()
            .is_empty()
    );
    // a model that does not read the forcing set at all has an empty block
    let none = bracket_doc(op("*", &[json!(-1), ix("c", &[json!("i")])]), json!({}));
    assert!(
        jacobian_bands_with_forcing(&none, None, Wrt::Forcing, Some(&f))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn array_parameter_defaults_broadcast_or_zero_fill() {
    // `K` carries a scalar default over shape [4], `Z` none at all: both
    // evaluate (0.5 and 0.0 per cell), as in the Python binding.
    let body = op(
        "+",
        &[
            op("*", &[ix("K", &[json!("i")]), ix("c", &[json!("i")])]),
            op("*", &[ix("Z", &[json!("i")]), ix("c", &[json!("i")])]),
        ],
    );
    let file = bracket_doc(
        body,
        json!({
            "K": {"type": "parameter", "shape": ["x"], "default": 0.5},
            "Z": {"type": "parameter", "shape": ["x"]},
        }),
    );
    let env = earthsci_astdiff::numeric::Env {
        states: [("c".to_string(), json!([0.1, 0.2, 0.3, 0.4]))].into(),
        params: HashMap::new(),
        t: 0.0,
    };
    let j = assemble_jacobian(&file, None, Wrt::States, &env).unwrap();
    for i in 1..=4 {
        assert_eq!(j[&(format!("c[{i}]"), format!("c[{i}]"))], 0.5);
    }
    let p = assemble_jacobian(&file, None, Wrt::Parameters, &env).unwrap();
    assert_eq!(p[&("c[2]".to_string(), "K[2]".to_string())], 0.2);
    assert_eq!(p[&("c[3]".to_string(), "Z[3]".to_string())], 0.3);
}

#[test]
fn prepared_evaluator_is_reused() {
    let file = load("forcing_bracket");
    let (env, _) = jvals_env("forcing_bracket.forcing");
    let jac = prepare_jacobian(&file, None, Wrt::Forcing, None).unwrap();
    assert_eq!(
        jac.colnames,
        column_names(
            &jacobian_columns(&sysview(&file, None).unwrap(), Wrt::Forcing, None).unwrap()
        )
    );
    assert_eq!(
        jac.values(&env).unwrap(),
        assemble_jacobian(&file, None, Wrt::Forcing, &env).unwrap()
    );
    let lambda: HashMap<String, f64> = ["S", "c[1]", "c[2]", "c[3]", "c[4]"]
        .iter()
        .enumerate()
        .map(|(k, r)| (r.to_string(), 0.5 + k as f64))
        .collect();
    for t in [0.0, 0.37] {
        let env = earthsci_astdiff::numeric::Env { t, ..env.clone() };
        let mut want = vec![0.0; jac.colnames.len()];
        vjp(&file, None, Wrt::Forcing, None, &env, &lambda, &mut want).unwrap();
        let mut got = vec![0.0; jac.colnames.len()];
        jac.vjp(&env, &lambda, &mut got).unwrap();
        assert_eq!(got, want);
        let mut acc: HashMap<String, Vec<f64>> = [
            ("F".to_string(), vec![0.0; 10]),
            ("G".to_string(), vec![0.0; 6]),
        ]
        .into();
        jac.vjp_forcing(&env, &lambda, &mut acc).unwrap();
        let flat: Vec<f64> = acc["F"].iter().chain(&acc["G"]).copied().collect();
        assert_eq!(flat, want);
    }
    // only on the forcing axis
    let jp = prepare_jacobian(&file, None, Wrt::Parameters, None).unwrap();
    assert!(jp.vjp_forcing(&env, &lambda, &mut HashMap::new()).is_err());
}

#[test]
fn a_block_without_wrt_is_refused() {
    let file = load("forcing_bracket");
    let mut block = jacobian_block(&file, None, Wrt::Forcing, true, 12).unwrap();
    assert!(parse_jacobian_block(&block).is_ok());
    block.as_object_mut().unwrap().remove("wrt");
    let err = parse_jacobian_block(&block).unwrap_err();
    assert!(err.0.contains("without `wrt`"), "{}", err.0);
}
