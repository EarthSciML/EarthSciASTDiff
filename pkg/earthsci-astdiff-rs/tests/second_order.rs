//! Second-order bands (spec §4.4): the band calculus applied to first-order
//! coefficients, for the axis pairs a Rosenbrock adjoint contracts. Mirrors
//! the reference's second_order_test.jl: byte-gated by
//! `tests/goldens/<fixture>.<a1>-<a2>.{bands,block}.json` and by value by
//! the matching `*.hvals.json` (1e-10 relative, exact key set).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;

use earthsci_ast::free_variables;
use earthsci_astdiff::helpers::ser_expr;
use earthsci_astdiff::{
    hvp, jacobian_bands2, jacobian_bands2_with_forcing, jacobian_block2, jacobian_blocks,
    parse_jacobian_block, parse_jacobian_block2, parse_jacobian_blocks, prepare_hvp,
    serialize_band2, stable_json, BlockWrt, HvpForm, HvpPair, JacEntry2, ParsedBlock, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{cells_env, load, root};

const SS: (Wrt, Wrt) = (Wrt::States, Wrt::States);

const SECOND_ORDER: [(&str, (Wrt, Wrt)); 6] = [
    ("bd_chem", SS),
    ("arrhenius_chem", SS),
    ("arrhenius_chem", (Wrt::States, Wrt::Parameters)),
    ("forcing_bracket", (Wrt::States, Wrt::Forcing)),
    ("forcing_bracket", (Wrt::Time, Wrt::States)),
    ("minmod_adv", SS),
];

fn tag((a1, a2): (Wrt, Wrt)) -> String {
    format!("{}-{}", a1.as_str(), a2.as_str())
}

fn golden(file: &str) -> String {
    fs::read_to_string(root().join("tests/goldens").join(file)).unwrap()
}

fn ser(es: &[JacEntry2]) -> String {
    stable_json(&Value::Array(es.iter().map(serialize_band2).collect()))
}

fn triples(es: &[JacEntry2]) -> HashSet<(String, String, String)> {
    es.iter()
        .map(|en| (en.u.clone(), en.v.clone(), en.w.clone()))
        .collect()
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

#[test]
fn second_order_goldens() {
    for (name, wrt) in SECOND_ORDER {
        let file = load(name);
        let entries = jacobian_bands2(&file, None, wrt).unwrap();
        let text = ser(&entries);
        assert_eq!(
            text,
            golden(&format!("{name}.{}.bands.json", tag(wrt))),
            "second-order band golden diverges: {name} {}",
            tag(wrt)
        );
        let block = jacobian_block2(&file, None, wrt, None, true, 12).unwrap();
        assert_eq!(block["wrt"], json!([wrt.0.as_str(), wrt.1.as_str()]));
        assert!(block.get("factorization").is_none(), "{name}");
        assert_eq!(
            block.get("columns").is_some(),
            wrt.1.has_columns(),
            "{name}"
        );
        assert_eq!(
            stable_json(&block),
            golden(&format!("{name}.{}.block.json", tag(wrt))),
            "second-order block golden diverges: {name} {}",
            tag(wrt)
        );
        let (w, back) = parse_jacobian_block2(&block).unwrap();
        assert_eq!(w, wrt);
        assert_eq!(ser(&back), text, "pair block round trip: {name}");
    }
}

#[test]
fn what_the_entries_are() {
    // bd_chem's bimolecular terms: k3·B·C gives both orientations of (B, C)
    // in A's row, and B² a single (B, B) entry; the linear k1 term none
    let e2 = jacobian_bands2(&load("bd_chem"), None, SS).unwrap();
    let want: HashSet<(String, String, String)> = [
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "B", "B"),
        ("C", "B", "B"),
    ]
    .iter()
    .map(|(a, b, c)| (a.to_string(), b.to_string(), c.to_string()))
    .collect();
    assert_eq!(triples(&e2), want);
    // every orientation comes out on its own: (r, c, s) ⇒ (r, s, c)
    for name in ["bd_chem", "arrhenius_chem"] {
        let t = triples(&jacobian_bands2(&load(name), None, SS).unwrap());
        assert!(t
            .iter()
            .all(|(r, c, s)| t.contains(&(r.clone(), s.clone(), c.clone()))));
    }
    // the Arrhenius rate is an observed of parameters only: opaque on the
    // state pair, inlined on the parameter pair so T's sites are seen
    let file = load("arrhenius_chem");
    let ss = jacobian_bands2(&file, None, SS).unwrap();
    assert!(ss
        .iter()
        .any(|en| free_variables(&en.band.coef).contains("k1")));
    let sp = jacobian_bands2(&file, None, (Wrt::States, Wrt::Parameters)).unwrap();
    assert!(sp.iter().any(|en| en.w == "T"
        && en.col2_idx().len() == 1
        && stable_json(&ser_expr(&en.col2_idx()[0])) == "\"i\""));
    assert!(sp
        .iter()
        .all(|en| !free_variables(&en.band.coef).contains("k1")));
    // minmod's limiter is piecewise linear: its first-order coefficients are
    // branch-wise constant, so the second-order pattern is empty (§3.2)
    assert!(jacobian_bands2(&load("minmod_adv"), None, SS)
        .unwrap()
        .is_empty());
    // the bracket's records stay distinct second columns
    let fb = jacobian_bands2(&load("forcing_bracket"), None, (Wrt::States, Wrt::Forcing)).unwrap();
    assert!(fb
        .iter()
        .any(|en| en.w == "G" && !en.band.contracted.is_empty()));
    let recs: HashSet<String> = fb
        .iter()
        .filter(|en| en.u == "c" && en.w == "F")
        .map(|en| stable_json(&ser_expr(&en.col2_idx()[0])))
        .collect();
    assert_eq!(recs, ["1", "2"].iter().map(|x| x.to_string()).collect());
}

#[test]
fn structure() {
    let blk = |name: &str, wrt| jacobian_block2(&load(name), None, wrt, None, true, 12).unwrap();
    assert_eq!(blk("bd_chem", SS)["structure"], "block_diagonal");
    assert_eq!(blk("arrhenius_chem", SS)["structure"], "block_diagonal");
    // a scalar parameter column is shared by every lane: not lane-local
    assert!(blk("arrhenius_chem", (Wrt::States, Wrt::Parameters))
        .get("structure")
        .is_none());
    assert!(blk("forcing_bracket", (Wrt::Time, Wrt::States))
        .get("structure")
        .is_none());
    let hv = prepare_hvp(&load("bd_chem"), None, SS, None).unwrap();
    assert_eq!(hv.structure, Some("block_diagonal"));
}

/// The pinned point and stage pairs of `<name>.<a1>-<a2>.hvals.json`.
fn hvals(name: &str, wrt: (Wrt, Wrt)) -> (earthsci_astdiff::numeric::Env, Vec<HvpPair>, Value) {
    let g: Value =
        serde_json::from_str(&golden(&format!("{name}.{}.hvals.json", tag(wrt)))).unwrap();
    assert_eq!(g["wrt"], json!([wrt.0.as_str(), wrt.1.as_str()]));
    // parameters take the document defaults
    let env = cells_env(&g["u"], HashMap::new(), g["t"].as_f64().unwrap());
    let num = |m: &Value| -> HashMap<String, f64> {
        m.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
            .collect()
    };
    let pairs = g["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pr| {
            let w = match &pr["w"] {
                Value::Number(x) => [("t".to_string(), x.as_f64().unwrap())].into(),
                m => num(m),
            };
            (num(&pr["mu"]), w)
        })
        .collect();
    (env, pairs, g)
}

fn check_g(what: &str, got: &BTreeMap<String, f64>, golden: &Value) {
    let want: Vec<(String, f64)> = golden["g"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e[0].as_str().unwrap().to_string(), e[1].as_f64().unwrap()))
        .collect();
    // exact key set: every second column a band point reaches
    assert_eq!(
        got.keys().cloned().collect::<Vec<_>>(),
        want.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "{what}: key set"
    );
    for (n, v) in want {
        let gv = got[&n];
        assert!(
            (gv - v).abs() <= (1e-10 * v.abs()).max(1e-13),
            "{what}: g[{n}] {gv} vs {v}"
        );
    }
}

#[test]
fn value_goldens() {
    for (name, wrt) in SECOND_ORDER {
        let file = load(name);
        let (env, pairs, gd) = hvals(name, wrt);
        let hv = prepare_hvp(&file, None, wrt, None).unwrap();
        let mut g = BTreeMap::new();
        hv.hvp(&env, &pairs, HvpForm::Contract, &mut g).unwrap();
        check_g(&format!("{name} {}", tag(wrt)), &g, &gd);
        if wrt == SS {
            let mut gs = BTreeMap::new();
            hv.hvp(&env, &pairs, HvpForm::Symmetric, &mut gs).unwrap();
            check_g(&format!("{name} {} (symmetric)", tag(wrt)), &gs, &gd);
        }
        // one pair at a time accumulates to the same
        let mut g1 = BTreeMap::new();
        for pr in &pairs {
            hv.hvp(&env, std::slice::from_ref(pr), HvpForm::Contract, &mut g1)
                .unwrap();
        }
        assert_eq!(g1.len(), g.len());
        for (n, v) in &g {
            assert!(
                (g1[n] - v).abs() <= 1e-12 * v.abs().max(1e-2),
                "{name}: {n}"
            );
        }
        // the one-call form is the prepared one
        let mut g2 = BTreeMap::new();
        hvp(
            &file,
            None,
            wrt,
            None,
            &env,
            &pairs,
            HvpForm::Contract,
            &mut g2,
        )
        .unwrap();
        assert_eq!(g2, g);
    }
}

#[test]
fn document_forms() {
    let file = load("forcing_bracket");
    let wrt = [
        BlockWrt::Axis(Wrt::States),
        BlockWrt::Pair(Wrt::States, Wrt::Forcing),
        BlockWrt::Pair(Wrt::Time, Wrt::States),
    ];
    let jb = jacobian_blocks(&file, None, &wrt, None, true, 12).unwrap();
    let arr = jb.as_array().unwrap();
    assert_eq!(
        arr.iter().map(|b| b["wrt"].clone()).collect::<Vec<_>>(),
        vec![
            json!("states"),
            json!(["states", "forcing"]),
            json!(["time", "states"])
        ]
    );
    assert_eq!(
        stable_json(&arr[1]),
        golden("forcing_bracket.states-forcing.block.json")
    );
    let parsed = parse_jacobian_blocks(&jb).unwrap();
    assert_eq!(
        parsed.iter().map(ParsedBlock::wrt).collect::<Vec<_>>(),
        wrt.to_vec()
    );
    let ParsedBlock::Pair(_, es) = &parsed[1] else {
        panic!("a pair block parses to second-order entries")
    };
    assert_eq!(
        ser(es),
        ser(&jacobian_bands2(&file, None, (Wrt::States, Wrt::Forcing)).unwrap())
    );
    // a pair is not a single-axis block, and vice versa
    assert!(parse_jacobian_block(&arr[1]).is_err());
    assert!(parse_jacobian_block2(&arr[0]).is_err());
    assert!(jacobian_blocks(&file, None, &[SS, SS], None, true, 12).is_err());
    assert!(parse_jacobian_blocks(&json!([arr[2], arr[2]])).is_err());
}

#[test]
fn refusals() {
    let file = load("forcing_bracket");
    for bad in [
        (Wrt::Forcing, Wrt::States),
        (Wrt::Parameters, Wrt::States),
        (Wrt::States, Wrt::Time),
    ] {
        assert!(jacobian_bands2(&file, None, bad).is_err(), "{bad:?}");
        assert!(jacobian_block2(&file, None, bad, None, true, 12).is_err());
        assert!(prepare_hvp(&file, None, bad, None).is_err());
    }
    let mut block = jacobian_block2(&file, None, (Wrt::Time, Wrt::States), None, true, 12).unwrap();
    block["wrt"] = json!(["states"]);
    assert!(parse_jacobian_block2(&block).is_err());
    block["wrt"] = json!(["forcing", "states"]);
    assert!(parse_jacobian_block2(&block).is_err());
    // the symmetric form needs a (states, states) pair
    let ht = prepare_hvp(&file, None, (Wrt::Time, Wrt::States), None).unwrap();
    let (env, pairs, _) = hvals("forcing_bracket", (Wrt::Time, Wrt::States));
    assert!(ht
        .hvp(&env, &pairs, HvpForm::Symmetric, &mut BTreeMap::new())
        .is_err());
    // a stage that does not bind a row the bands read
    let mut short = pairs.clone();
    short[0].0.remove("c[2]");
    assert!(ht
        .hvp(&env, &short, HvpForm::Contract, &mut BTreeMap::new())
        .is_err());
    // a second column read through a parameter offset cannot be placed
    let ix = |v: &str, i: &[Value]| {
        let mut args = vec![json!(v)];
        args.extend(i.iter().cloned());
        json!({"op": "index", "args": args})
    };
    let body = json!({"op": "*", "args": [
        ix("c", &[json!("i")]),
        ix("c", &[json!("i")]),
        ix("F", &[json!(1), json!({"op": "+", "args": ["i", "s"]})]),
    ]});
    let doc = json!({
        "esm": "1.1.0", "metadata": {"name": "B"},
        "index_sets": {
            "x": {"kind": "interval", "size": 4},
            "xf": {"kind": "interval", "size": 5},
            "rec": {"kind": "interval", "size": 2},
        },
        "models": {"B": {
            "variables": {
                "c": {"type": "unknown", "shape": ["x"], "default": 0.5},
                "F": {"type": "parameter", "shape": ["rec", "xf"], "default": 1.0},
                "s": {"type": "parameter", "default": 1},
            },
            "equations": [{
                "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                "rhs": {"op": "faq", "args": [], "output_idx": ["i"],
                        "ranges": {"i": {"from": "x"}}, "expr": body},
            }],
        }},
    });
    let doc = earthsci_ast::load_document(&doc).expect("document loads");
    let err =
        jacobian_bands2_with_forcing(&doc, None, (Wrt::States, Wrt::Forcing), Some(&s(&["F"])))
            .unwrap_err();
    assert!(err.0.contains("not closed at build time"), "{}", err.0);
}
