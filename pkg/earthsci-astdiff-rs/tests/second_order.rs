//! Second-order bands (spec §4.4): the band calculus applied to first-order
//! coefficients, for the axis pairs a Rosenbrock adjoint contracts. Mirrors
//! the reference's second_order_test.jl: byte-gated by
//! `tests/goldens/<fixture>.<a1>-<a2>.{bands,block}.json` and by value by
//! the matching `*.hvals.json` (1e-10 relative, exact key set), on the
//! numeric oracle and on the compiled [`HvpProgram`] alike.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;

use earthsci_ast::{esm_problem, free_variables, rhs_program, EsmFile, ProblemOptions, RhsProgram};
use earthsci_astdiff::helpers::ser_expr;
use earthsci_astdiff::{
    hvp, jacobian_bands2, jacobian_bands2_with_forcing, jacobian_block, jacobian_block2,
    jacobian_blocks, parse_jacobian_block, parse_jacobian_block2, parse_jacobian_blocks,
    prepare_hvp, serialize_band2, stable_json, BlockWrt, BuildOptions, Exec, HvpForm, HvpPair,
    HvpPlan, HvpProgram, HvpWorkspace, JacEntry2, ParsedBlock, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{cells_env, load, root};

const SS: (Wrt, Wrt) = (Wrt::States, Wrt::States);

// The compiled contraction's thread-safety contract, as tests/program.rs
// pins `JacobianProgram`'s.
const _: () = {
    const fn shared<T: Send + Sync + Clone>() {}
    const fn sendable<T: Send>() {}
    shared::<HvpProgram>();
    shared::<HvpPlan>();
    sendable::<HvpWorkspace>();
};

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

fn golden_bands(name: &str, wrt: (Wrt, Wrt)) -> String {
    golden(&format!("{name}.{}.bands.json", tag(wrt)))
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
    // one emission shares its first-order passes: the blocks equal those
    // emitted alone
    for name in ["arrhenius_chem", "bd_chem"] {
        let f = load(name);
        let wrt = [
            BlockWrt::Axis(Wrt::States),
            BlockWrt::Pair(Wrt::States, Wrt::States),
            BlockWrt::Pair(Wrt::States, Wrt::Parameters),
            BlockWrt::Axis(Wrt::Time),
        ];
        let both = jacobian_blocks(&f, None, &wrt, None, true, 12).unwrap();
        let alone: Vec<Value> = wrt
            .iter()
            .map(|w| match *w {
                BlockWrt::Axis(a) => jacobian_block(&f, None, a, true, 12).unwrap(),
                BlockWrt::Pair(a1, a2) => {
                    jacobian_block2(&f, None, (a1, a2), None, true, 12).unwrap()
                }
            })
            .collect();
        assert_eq!(both, Value::Array(alone), "{name}");
    }
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

// ---------------------------------------------------------------------------
// The compiled contraction (program.rs) against the oracle and the goldens.
// ---------------------------------------------------------------------------

/// The integrator's program for `file`, as tests/program.rs builds it; with
/// `arrays`, shaped parameters without an `update` stay run-time slots.
fn source(file: &EsmFile, arrays: bool) -> RhsProgram {
    let opts = ProblemOptions {
        runtime_array_parameters: arrays,
        ..Default::default()
    };
    rhs_program(&esm_problem(file, (0.0, 1.0), opts).unwrap()).unwrap()
}

/// Column-major cell names of a block `name` over `shape`.
fn cell_names(name: &str, shape: &[usize]) -> Vec<String> {
    if shape.is_empty() {
        return vec![name.to_string()];
    }
    let n: usize = shape.iter().product();
    (0..n)
        .map(|mut f| {
            let idx: Vec<String> = shape
                .iter()
                .map(|s| {
                    let c = f % s + 1;
                    f /= s;
                    c.to_string()
                })
                .collect();
            format!("{name}[{}]", idx.join(","))
        })
        .collect()
}

/// The source's `u ++ aux` cell names, and its run-time inputs' (`p` then
/// every forcing buffer, column-major), model prefix stripped.
fn source_names(src: &RhsProgram) -> (Vec<String>, Vec<String>) {
    let bare = |n: &str| n.split_once('.').map_or(n, |x| x.1).to_string();
    let lay = src.layout();
    let mut states = Vec::new();
    for blocks in [&lay.u, &lay.aux] {
        for b in blocks {
            let shape: Vec<usize> = b
                .storage
                .iter()
                .map(|(lo, hi)| (hi - lo + 1) as usize)
                .collect();
            states.extend(cell_names(&bare(&b.name), &shape));
        }
    }
    let mut inputs: Vec<String> = src.parameters().iter().map(|q| bare(&q.name)).collect();
    for f in src.forcing_slots() {
        inputs.extend(cell_names(&bare(&f.name), &f.shape));
    }
    (states, inputs)
}

/// `forcing_bracket` with its forcing buffers refreshed by a registered
/// handler instead of a `data` source, and their inline defaults replaced by
/// a scalar: the same bands (given the forcing set explicitly), but the
/// source program routes them to forcing slots, where a `data` update
/// without a provider, or an inline-array default, leaves them build-time
/// data. Also returns each buffer's document value, column-major, to fill
/// those slots with the golden's point.
fn handler_fed(name: &str) -> (EsmFile, HashMap<String, Vec<f64>>) {
    let mut d: Value = serde_json::from_str(
        &fs::read_to_string(root().join("tests/valid").join(format!("{name}.esm"))).unwrap(),
    )
    .unwrap();
    let mut values = HashMap::new();
    for m in d["models"].as_object_mut().unwrap().values_mut() {
        for (n, v) in m["variables"].as_object_mut().unwrap().iter_mut() {
            if v.get("update").is_none() {
                continue;
            }
            // a rank-2 inline default, row-major nested → column-major
            let rows: Vec<Vec<f64>> = v["default"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect()
                })
                .collect();
            let cm = (0..rows[0].len())
                .flat_map(|j| rows.iter().map(move |r| r[j]))
                .collect();
            values.insert(n.clone(), cm);
            v["update"] = json!({"kind": "schedule", "interval": 60.0,
                                 "handler": {"handler_id": "reload"}});
            v["default"] = json!(1.0);
        }
    }
    (earthsci_ast::load_document(&d).unwrap(), values)
}

#[test]
fn compiled_contraction_matches_the_oracle_and_goldens() {
    for (name, wrt) in SECOND_ORDER {
        let forced = wrt.1 == Wrt::Forcing;
        let (file, buffers) = if forced {
            handler_fed(name)
        } else {
            (load(name), HashMap::new())
        };
        let fnames = s(&["F", "G"]);
        let forcing = forced.then_some(&fnames[..]);
        // a parameter pair needs the array parameter `T` as a run-time slot
        let src = source(&file, wrt.1 == Wrt::Parameters);
        let (states, inputs) = source_names(&src);
        let (nu, na) = (src.n_states(), src.layout().aux_len);
        assert_eq!(states.len(), nu + na);
        let (env, pairs, gd) = hvals(name, wrt);
        // the oracle reads the unmodified document (its defaults are the
        // golden's point)
        let oracle = prepare_hvp(&load(name), None, wrt, None).unwrap();
        for cse in [true, false] {
            let opts = BuildOptions {
                cse,
                ..Default::default()
            };
            let prog = HvpProgram::build(&file, None, wrt, forcing, &opts).unwrap();
            assert_eq!(
                ser(prog.bands()),
                golden_bands(name, wrt),
                "{name}: the compiled bands are the golden's"
            );
            let plan = prog.plan(&src, Exec::Domain).unwrap();
            assert_eq!(plan.source_lengths(), (nu, na));
            let gnames = if wrt.1 == Wrt::States {
                &states
            } else {
                &inputs
            };
            assert_eq!(plan.g_len(), gnames.len());
            // the golden's point and stages in the source's layout
            let (du, daux) = src.default_state();
            let at = |n: &String, d: f64| gd["u"].get(n).and_then(Value::as_f64).unwrap_or(d);
            let u: Vec<f64> = states[..nu]
                .iter()
                .zip(&du)
                .map(|(n, d)| at(n, *d))
                .collect();
            let aux: Vec<f64> = states[nu..]
                .iter()
                .zip(&daux)
                .map(|(n, d)| at(n, *d))
                .collect();
            assert!(
                states[..nu].iter().all(|n| gd["u"].get(n).is_some()),
                "{name}"
            );
            let p = src.parameter_vector(&HashMap::new()).unwrap();
            let mut fset = src.default_forcing();
            for (k, slot) in src.forcing_slots().iter().enumerate() {
                if let Some(b) = buffers.get(slot.name.split_once('.').unwrap().1) {
                    fset = fset.replace(k, b.clone().into()).unwrap();
                }
            }
            let t = gd["t"].as_f64().unwrap();
            let stage: Vec<(Vec<f64>, Vec<f64>)> = pairs
                .iter()
                .map(|(mu, w)| {
                    let m = states[..nu].iter().map(|n| mu[n]).collect();
                    let w = if wrt.0 == Wrt::Time {
                        vec![w["t"]]
                    } else {
                        states
                            .iter()
                            .map(|n| w.get(n).copied().unwrap_or(0.0))
                            .collect()
                    };
                    (m, w)
                })
                .collect();
            let refs: Vec<(&[f64], &[f64])> = stage.iter().map(|(m, w)| (&m[..], &w[..])).collect();
            let mut ws = prog.new_workspace(&plan);
            prog.eval(&plan, &u, &aux, &p, t, &fset, &mut ws).unwrap();
            let forms: &[HvpForm] = if wrt == SS {
                &[HvpForm::Contract, HvpForm::Symmetric]
            } else {
                &[HvpForm::Contract]
            };
            for &form in forms {
                let what = format!("{name} {} {form:?} cse={cse}", tag(wrt));
                let mut g = vec![0.0; plan.g_len()];
                prog.contract(&plan, &mut ws, &refs, form, &mut g).unwrap();
                let touched = plan.touched(form);
                // by name, as the oracle and the golden key it
                let mut got: BTreeMap<String, f64> = BTreeMap::new();
                for &i in &touched {
                    assert!(got.insert(gnames[i].clone(), g[i]).is_none(), "{what}");
                }
                assert!(
                    g.iter()
                        .enumerate()
                        .all(|(i, x)| *x == 0.0 || touched.binary_search(&i).is_ok()),
                    "{what}: g written outside the touched set"
                );
                check_g(&what, &got, &gd);
                let mut want = BTreeMap::new();
                oracle.hvp(&env, &pairs, form, &mut want).unwrap();
                assert_eq!(
                    got.keys().collect::<Vec<_>>(),
                    want.keys().collect::<Vec<_>>(),
                    "{what}: key set against the oracle"
                );
                for (n, v) in &want {
                    assert!(
                        (got[n] - v).abs() <= (1e-10 * v.abs()).max(1e-13),
                        "{what}: g[{n}] {} vs oracle {v}",
                        got[n]
                    );
                }
                // a fixed plan order: the same call is bitwise the same
                let mut g2 = vec![0.0; plan.g_len()];
                prog.contract(&plan, &mut ws, &refs, form, &mut g2).unwrap();
                assert_eq!(g2, g, "{what}");
                // `+=`: one stage at a time accumulates to the same
                let mut g1 = vec![0.0; plan.g_len()];
                for pr in &refs {
                    prog.contract(&plan, &mut ws, std::slice::from_ref(pr), form, &mut g1)
                        .unwrap();
                }
                assert!(
                    g1.iter()
                        .zip(&g)
                        .all(|(a, b)| (a - b).abs() <= 1e-12 * b.abs().max(1e-2)),
                    "{what}: per-stage accumulation"
                );
            }
        }
    }
}

#[test]
fn compiled_contraction_refusals() {
    let opts = BuildOptions::default();
    let file = load("bd_chem");
    let src = source(&file, false);
    let prog = HvpProgram::build(&file, None, SS, None, &opts).unwrap();
    // a source program from another document; no lane form yet
    let other = source(&load("adv_interior"), false);
    assert!(prog.plan(&other, Exec::Domain).is_err());
    assert!(prog.plan(&src, Exec::Lanes { capacity: 4 }).is_err());
    // bad axis pairs
    assert!(HvpProgram::build(&file, None, (Wrt::Forcing, Wrt::States), None, &opts).is_err());
    // a forcing buffer the source holds as build-time data has no `g` slot
    let fb = load("forcing_bracket");
    let fprog = HvpProgram::build(&fb, None, (Wrt::States, Wrt::Forcing), None, &opts).unwrap();
    assert!(fprog.plan(&source(&fb, false), Exec::Domain).is_err());
    // the array parameter `T` is build-time data of a default source
    let arr = load("arrhenius_chem");
    let sp = (Wrt::States, Wrt::Parameters);
    let aprog = HvpProgram::build(&arr, None, sp, None, &opts).unwrap();
    let err = aprog.plan(&source(&arr, false), Exec::Domain).unwrap_err();
    assert!(
        err.to_string().contains("runtime_array_parameters"),
        "{err}"
    );
    // the symmetric form needs a (states, states) pair
    let asrc = source(&arr, true);
    let plan = aprog.plan(&asrc, Exec::Domain).unwrap();
    let mut ws = aprog.new_workspace(&plan);
    let (u, aux) = asrc.default_state();
    let p = asrc.parameter_vector(&HashMap::new()).unwrap();
    let f = asrc.default_forcing();
    let mu = vec![1.0; u.len()];
    let w = vec![1.0; plan.w_len()];
    let mut g = vec![0.0; plan.g_len()];
    // no slab before an evaluation
    assert!(aprog
        .contract(&plan, &mut ws, &[(&mu, &w)], HvpForm::Contract, &mut g)
        .is_err());
    aprog.eval(&plan, &u, &aux, &p, 0.7, &f, &mut ws).unwrap();
    aprog
        .contract(&plan, &mut ws, &[(&mu, &w)], HvpForm::Contract, &mut g)
        .unwrap();
    assert!(aprog
        .contract(&plan, &mut ws, &[(&mu, &w)], HvpForm::Symmetric, &mut g)
        .is_err());
    // call-time lengths, and a workspace or plan of another program
    assert!(aprog
        .contract(&plan, &mut ws, &[(&mu[1..], &w)], HvpForm::Contract, &mut g)
        .is_err());
    assert!(aprog
        .contract(&plan, &mut ws, &[(&mu, &w)], HvpForm::Contract, &mut g[1..])
        .is_err());
    assert!(aprog
        .eval(&plan, &u[1..], &aux, &p, 0.7, &f, &mut ws)
        .is_err());
    let plan2 = prog.plan(&src, Exec::Domain).unwrap();
    let mut ws2 = prog.new_workspace(&plan2);
    assert!(aprog.eval(&plan, &u, &aux, &p, 0.7, &f, &mut ws2).is_err());
    assert!(prog.eval(&plan, &u, &aux, &p, 0.7, &f, &mut ws).is_err());
}

#[test]
fn an_empty_second_order_list_compiles_to_a_no_op() {
    // minmod's limiter is piecewise linear (§3.2): nothing to compile
    let file = load("minmod_adv");
    let src = source(&file, false);
    let prog = HvpProgram::build(&file, None, SS, None, &BuildOptions::default()).unwrap();
    assert!(prog.bands().is_empty() && prog.rhs().is_none());
    let plan = prog.plan(&src, Exec::Domain).unwrap();
    assert!(plan.touched(HvpForm::Contract).is_empty());
    let mut ws = prog.new_workspace(&plan);
    let (u, aux) = src.default_state();
    let p = src.parameter_vector(&HashMap::new()).unwrap();
    prog.eval(&plan, &u, &aux, &p, 0.7, &src.default_forcing(), &mut ws)
        .unwrap();
    let mu = vec![1.0; u.len()];
    let w = vec![1.0; plan.w_len()];
    let mut g = vec![0.0; plan.g_len()];
    for form in [HvpForm::Contract, HvpForm::Symmetric] {
        prog.contract(&plan, &mut ws, &[(&mu, &w)], form, &mut g)
            .unwrap();
    }
    assert!(g.iter().all(|x| *x == 0.0));
    // the source is still checked at the call
    assert!(prog
        .eval(
            &plan,
            &u[1..],
            &aux,
            &p,
            0.7,
            &src.default_forcing(),
            &mut ws
        )
        .is_err());
}
