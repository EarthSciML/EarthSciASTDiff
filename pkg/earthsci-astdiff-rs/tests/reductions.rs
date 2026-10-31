//! Filtered faq nodes, non-`+` reductions and value-indexed reads
//! (esm-jacobian-spec.md §3.5; port of the reference's
//! test/reductions_test.jl): every case is checked against central finite
//! differences of earthsci-ast's own right-hand side, and the guarded cases
//! for an EXACT pattern — the window's points, no padding. Also the
//! `column_block` structure class and the unresolvable-column refusal.

use std::collections::{BTreeMap, HashMap};

use earthsci_ast::{esm_problem, rhs_program, EsmFile, ProblemOptions};
use earthsci_astdiff::{
    column_lines, ctx_of, detect_structure, detect_structure_along, jacobian_bands, jacobian_block,
    jacobian_block_with, sysview, BlockOptions, JacEntry, Wrt,
};
use serde_json::{json, Map, Value};

mod common;
use common::docvals::{doc_jacobian, doc_jacobian_entries, Env};

fn idx(v: &str, i: &[Value]) -> Value {
    let mut args = vec![json!(v)];
    args.extend(i.iter().cloned());
    json!({"op": "index", "args": args})
}

fn op(o: &str, a: &[Value]) -> Value {
    json!({"op": o, "args": a})
}

fn faq(oidx: &[&str], ranges: &[(&str, i64, i64)], body: Value, kw: Value) -> Value {
    let mut r = Map::new();
    for (n, lo, hi) in ranges {
        r.insert(n.to_string(), json!([lo, hi]));
    }
    let mut d = json!({"op": "faq", "args": [], "output_idx": oidx, "ranges": r, "expr": body});
    for (k, v) in kw.as_object().unwrap() {
        d[k] = v.clone();
    }
    d
}

fn neg(x: Value) -> Value {
    op("-", &[x])
}

fn cmp(o: &str, a: impl Into<Value>, b: impl Into<Value>) -> Value {
    op(o, &[a.into(), b.into()])
}

/// One model over `x` (size `n`) and, optionally, `z` (size `nz`); `u` is
/// the state.
struct Doc {
    n: i64,
    nz: i64,
    ushape: Vec<&'static str>,
    vars: Value,
    eqs: Vec<Value>,
}

impl Default for Doc {
    fn default() -> Self {
        Doc {
            n: 5,
            nz: 0,
            ushape: vec!["x"],
            vars: json!({}),
            eqs: vec![],
        }
    }
}

fn doc(rhs: Value, d: Doc) -> EsmFile {
    let mut sets = json!({"x": {"kind": "interval", "size": d.n}});
    if d.nz > 0 {
        sets["z"] = json!({"kind": "interval", "size": d.nz});
    }
    let mut vars = json!({
        "u": {"type": "unknown", "shape": d.ushape, "default": 0.5},
        "w": {"type": "parameter", "default": 0.7},
    });
    for (k, v) in d.vars.as_object().unwrap() {
        vars[k] = v.clone();
    }
    let mut eqs = d.eqs;
    eqs.push(json!({"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": rhs}));
    let doc = json!({
        "esm": "1.1.0",
        "metadata": {"name": "U9"},
        "index_sets": sets,
        "models": {"U9": {"variables": vars, "equations": eqs}},
    });
    earthsci_ast::load_document(&doc).expect("document loads")
}

/// `u` (column-major over its shape) as nested arrays.
fn nested(vals: &[f64], shape: &[usize]) -> Value {
    fn rec(vals: &[f64], shape: &[usize], strides: &[usize], off: usize) -> Value {
        match shape.split_first() {
            None => json!(vals[off]),
            Some((n, rest)) => Value::Array(
                (0..*n)
                    .map(|i| rec(vals, rest, &strides[1..], off + i * strides[0]))
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

/// The cell name of column-major flat index `f` over `shape`.
fn cell(f: usize, shape: &[usize]) -> String {
    let mut rest = f;
    let mut c = Vec::new();
    for n in shape {
        c.push((rest % n + 1).to_string());
        rest /= n;
    }
    format!("u[{}]", c.join(","))
}

/// The analytic Jacobian at `u` against central differences of the model's
/// own right-hand side, the pattern a superset of the local one; returns the
/// band entries and the pattern size.
fn check(file: &EsmFile, u: &[f64], label: &str, npat: Option<usize>) -> Vec<JacEntry> {
    let src =
        rhs_program(&esm_problem(file, (0.0, 1.0), ProblemOptions::default()).unwrap()).unwrap();
    let lay = src.layout();
    let ub = lay
        .u
        .iter()
        .find(|b| b.name.ends_with(".u") || b.name == "u")
        .unwrap();
    let shape: Vec<usize> = ub
        .storage
        .iter()
        .map(|(lo, hi)| (hi - lo + 1) as usize)
        .collect();
    assert_eq!(u.len(), shape.iter().product::<usize>(), "{label}");
    let (mut y, aux) = src.default_state();
    y[ub.base..ub.base + u.len()].copy_from_slice(u);
    let p: Vec<f64> = src
        .parameters()
        .iter()
        .map(|s| s.default.unwrap_or(0.0))
        .collect();
    let forcing = src.default_forcing();
    let mut ws = src.new_workspace();
    let mut f = |y: &[f64]| {
        let mut du = vec![0.0; y.len()];
        src.eval_into(y, &aux, &p, 0.0, &forcing, &mut du, &mut ws)
            .unwrap();
        du[ub.base..ub.base + u.len()].to_vec()
    };
    let h = 1e-6;
    let mut fd: BTreeMap<(String, String), f64> = BTreeMap::new();
    for j in 0..u.len() {
        let (mut yp, mut ym) = (y.clone(), y.clone());
        yp[ub.base + j] += h;
        ym[ub.base + j] -= h;
        let (fp, fm) = (f(&yp), f(&ym));
        for i in 0..u.len() {
            let d = (fp[i] - fm[i]) / (2.0 * h);
            if d != 0.0 {
                fd.insert((cell(i, &shape), cell(j, &shape)), d);
            }
        }
    }
    let env = Env {
        states: HashMap::from([("u".to_string(), nested(u, &shape))]),
        params: HashMap::new(),
        t: 0.0,
    };
    let jac = doc_jacobian(file, None, Wrt::States, None, &env)
        .unwrap_or_else(|e| panic!("{label}: {}", e.0));
    for (k, want) in &fd {
        let got = jac
            .get(k)
            .unwrap_or_else(|| panic!("{label}: {k:?} = {want} outside the pattern"));
        assert!((got - want).abs() <= 1e-6, "{label} {k:?}: {got} vs {want}");
    }
    for (k, got) in &jac {
        let want = fd.get(k).copied().unwrap_or(0.0);
        assert!((got - want).abs() <= 1e-6, "{label} {k:?}: {got} vs {want}");
    }
    if let Some(n) = npat {
        assert_eq!(jac.len(), n, "{label}: pattern size");
    }
    jacobian_bands(file, None, Wrt::States).unwrap()
}

const U5: [f64; 5] = [0.3, 0.9, 0.2, 0.6, 0.45];

fn oi(body: Value, kw: Value) -> Value {
    faq(&["i"], &[("i", 1, 5), ("j", 1, 5)], body, kw)
}

fn scan(f: Value) -> Value {
    neg(oi(
        op("*", &[json!("w"), idx("u", &[json!("j")])]),
        json!({"filter": f}),
    ))
}

#[test]
fn filtered_sums_at_array_level() {
    let d = Doc::default;
    for (f, label, n) in [
        (cmp("<=", "j", "i"), "inclusive forward scan", 15),
        (cmp("<", "j", "i"), "exclusive forward scan", 10),
        (cmp(">=", "j", "i"), "inclusive reverse scan", 15),
        (cmp(">", "i", "j"), "mirrored spelling", 10),
        (cmp("<=", "j", 2), "constant window (box only)", 10),
        (
            op("and", &[cmp("<=", "j", "i"), cmp(">=", "j", 2)]),
            "window clipped by a constant",
            10,
        ),
    ] {
        check(&doc(scan(f), d()), &U5, label, Some(n));
    }
    // the window read squared, through the chain rule, and a value gate
    let vf = op(
        "and",
        &[cmp("<=", "j", "i"), cmp(">", idx("u", &[json!("j")]), 0.4)],
    );
    let body = op("^", &[idx("u", &[json!("j")]), json!(2)]);
    check(
        &doc(neg(oi(body, json!({"filter": vf}))), d()),
        &U5,
        "value-dependent filter conjunct",
        Some(15),
    );
    // a column that reads no contracted index: the filtered scalar sum
    let body = op("*", &[idx("u", &[json!("i")]), idx("u", &[json!("j")])]);
    check(
        &doc(neg(oi(body, json!({"filter": cmp("<", "j", "i")}))), d()),
        &U5,
        "column with and without the contracted index",
        None,
    );
}

#[test]
fn level_scan_takes_a_where_guard() {
    // the Mz shape: (gi, gke) rows over a level contraction gk < gke
    let mz = faq(
        &["gi", "gke"],
        &[("gi", 1, 3), ("gke", 1, 4), ("gk", 1, 4)],
        op("*", &[json!("w"), idx("u", &[json!("gi"), json!("gk")])]),
        json!({"filter": cmp("<", "gk", "gke")}),
    );
    let u34: Vec<f64> = (0..12).map(|k| 0.1 + 0.1 * k as f64).collect();
    let d = Doc {
        n: 3,
        nz: 4,
        ushape: vec!["x", "z"],
        ..Default::default()
    };
    let es = check(
        &doc(neg(mz), d),
        &u34,
        "Mz: exclusive level scan",
        Some(3 * 6),
    );
    assert!(es.iter().any(|en| en.band.guard.is_some()));
    // a row-only filter on a non-contracting node: rows clipped, no guard
    let ro = faq(
        &["i"],
        &[("i", 1, 5)],
        op("*", &[json!("w"), idx("u", &[json!("i")])]),
        json!({"filter": cmp(">=", "i", 3)}),
    );
    let es = check(&doc(ro, Doc::default()), &U5, "row-only filter", Some(3));
    assert!(es.iter().all(|en| en.band.guard.is_none()));
}

#[test]
fn nested_filtered_sum_guards_its_lifted_index() {
    let seq = json!({"lhs": "s", "rhs": oi(op("^", &[idx("u", &[json!("j")]), json!(2)]),
                                           json!({"filter": cmp("<=", "j", "i")}))});
    let rhs = faq(
        &["i"],
        &[("i", 1, 5)],
        neg(op("*", &[idx("s", &[json!("i")]), idx("u", &[json!("i")])])),
        json!({}),
    );
    let d = Doc {
        vars: json!({"s": {"type": "unknown", "shape": ["x"]}}),
        eqs: vec![seq],
        ..Default::default()
    };
    let es = check(&doc(rhs, d), &U5, "nested filtered sum", Some(15));
    assert!(es.iter().any(|en| en.band.guard.is_some()));
}

#[test]
fn nested_filter_over_a_contracted_index_filters_its_sum() {
    // Σ_k Σ_{j≤k}: the nested filter tied to the node's own contraction
    // filters the band's sum over k, never guards the band's point
    let seq = json!({"lhs": "s", "rhs": oi(op("^", &[idx("u", &[json!("j")]), json!(2)]),
                                           json!({"filter": cmp("<=", "j", "i")}))});
    let rhs = faq(
        &["i"],
        &[("i", 1, 5), ("k", 1, 5)],
        neg(op("*", &[idx("s", &[json!("k")]), idx("u", &[json!("i")])])),
        json!({}),
    );
    let d = Doc {
        vars: json!({"s": {"type": "unknown", "shape": ["x"]}}),
        eqs: vec![seq],
        ..Default::default()
    };
    let es = check(
        &doc(rhs, d),
        &U5,
        "nested filter over a contracted index",
        Some(25),
    );
    assert!(es.iter().all(|en| en
        .band
        .guard
        .as_ref()
        .is_none_or(|g| !earthsci_astdiff::helpers::free_vars(g).contains("k"))));
}

fn mx(f: Value, red: &str) -> Value {
    oi(
        op("*", &[idx("u", &[json!("j")]), idx("u", &[json!("j")])]),
        json!({"filter": f, "reduce": red}),
    )
}

#[test]
fn max_min_reductions() {
    let d = Doc::default;
    check(
        &doc(neg(mx(cmp("<=", "j", "i"), "max")), d()),
        &U5,
        "running max",
        Some(15),
    );
    check(
        &doc(neg(mx(cmp(">=", "j", "i"), "min")), d()),
        &U5,
        "suffix min",
        Some(15),
    );
    check(
        &doc(
            neg(oi(
                idx("u", &[json!("j")]),
                json!({"semiring": "max_product"}),
            )),
            d(),
        ),
        &U5,
        "max_product semiring",
        Some(25),
    );
    // nested: D(u)[i] = -u[i]·max_j u[j]
    let mxs = faq(
        &[],
        &[("j", 1, 5)],
        idx("u", &[json!("j")]),
        json!({"reduce": "max"}),
    );
    check(
        &doc(
            faq(
                &["i"],
                &[("i", 1, 5)],
                neg(op("*", &[idx("u", &[json!("i")]), mxs])),
                json!({}),
            ),
            d(),
        ),
        &U5,
        "nested max",
        None,
    );
    // Exact ties: the first extremum in ascending order takes the derivative.
    let utie = [0.3, 0.9, 0.2, 0.9, 0.45];
    let file = doc(neg(mx(cmp("<=", "j", "i"), "max")), d());
    let env = Env {
        states: HashMap::from([("u".to_string(), json!(utie))]),
        params: HashMap::new(),
        t: 0.0,
    };
    let j = doc_jacobian(&file, None, Wrt::States, None, &env).unwrap();
    let at = |r: usize, c: usize| j[&(format!("u[{r}]"), format!("u[{c}]"))];
    assert!((at(4, 2) + 2.0 * 0.9).abs() < 1e-14 && at(4, 4) == 0.0);
    assert!((at(5, 2) + 2.0 * 0.9).abs() < 1e-14 && at(5, 4) == 0.0);
    // a max over two contracted indices has no ascending order: refused
    let mx2 = faq(
        &["i"],
        &[("i", 1, 3), ("j", 1, 3), ("k", 1, 4)],
        idx("u", &[json!("j"), json!("k")]),
        json!({"reduce": "max"}),
    );
    let d2 = Doc {
        n: 3,
        nz: 4,
        ushape: vec!["x", "z"],
        ..Default::default()
    };
    assert!(jacobian_bands(&doc(mx2, d2), None, Wrt::States).is_err());
}

#[test]
fn product_reductions() {
    let pr = oi(
        op("+", &[json!(1), idx("u", &[json!("j")])]),
        json!({"filter": cmp("<=", "j", "i"), "reduce": "*"}),
    );
    check(
        &doc(neg(pr.clone()), Doc::default()),
        &U5,
        "prefix product",
        Some(15),
    );
    check(
        &doc(neg(pr), Doc::default()),
        &[0.3, -1.0, 0.2, 0.6, 0.45],
        "prefix product, zero factor",
        Some(15),
    );
}

#[test]
fn level_search_as_an_index_is_one_hot() {
    // κ = min{k : u[k] > 0.5}; D(u)[i] = -u[κ]·u[i]
    let keq = json!({"lhs": "kap", "rhs": faq(&[], &[("k", 1, 5)], json!("k"),
        json!({"reduce": "min", "filter": cmp(">", idx("u", &[json!("k")]), 0.5)}))});
    let rhs = faq(
        &["i"],
        &[("i", 1, 5)],
        neg(op(
            "*",
            &[idx("u", &[json!("kap")]), idx("u", &[json!("i")])],
        )),
        json!({}),
    );
    let d = Doc {
        vars: json!({"kap": {"type": "unknown"}}),
        eqs: vec![keq],
        ..Default::default()
    };
    let es = check(&doc(rhs, d), &U5, "level search as an index", Some(25));
    assert!(es
        .iter()
        .any(|en| en.band.contracted.iter().any(|c| c.0.starts_with("_oh"))));
}

#[test]
fn unary_minus_around_an_array_keeps_its_sign() {
    // D(u) = -(w·u): the array-level unary `-` scales by -1
    let file = doc(neg(op("*", &[json!("w"), json!("u")])), Doc::default());
    check(&file, &U5, "unary minus", Some(5));
}

#[test]
fn column_block_structure() {
    let r = [("i", 1, 3), ("k", 1, 4)];
    let d = || Doc {
        n: 3,
        nz: 4,
        ushape: vec!["x", "z"],
        ..Default::default()
    };
    // an exclusive level scan along z plus a local term: dense lower lines
    let scan = faq(
        &["i", "k"],
        &[r[0], r[1], ("m", 1, 4)],
        op("*", &[json!("w"), idx("u", &[json!("i"), json!("m")])]),
        json!({"filter": cmp("<", "m", "k")}),
    );
    let loc = faq(
        &["i", "k"],
        &r,
        neg(idx("u", &[json!("i"), json!("k")])),
        json!({}),
    );
    let file = doc(op("+", &[scan, loc.clone()]), d());
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    assert_eq!(detect_structure(&entries, &ctx), "general");
    assert_eq!(
        detect_structure_along(&entries, &sv, &ctx, Some("z")).unwrap(),
        "column_block"
    );
    assert_eq!(
        detect_structure_along(&entries, &sv, &ctx, Some("x")).unwrap(),
        "general"
    );
    let cl = column_lines(&entries, &sv, &ctx, "z").unwrap().unwrap();
    assert_eq!(
        (
            cl.pattern,
            cl.kl,
            cl.ku,
            cl.line_length,
            cl.block_vars.clone()
        ),
        ("lower", 3, 0, 4, vec!["u".to_string()])
    );
    let opts = BlockOptions {
        column_axis: Some("z"),
        ..Default::default()
    };
    let block = jacobian_block_with(&file, None, Wrt::States, &opts).unwrap();
    assert_eq!(block["structure"], json!("column_block"));
    assert_eq!(
        block["factorization"],
        json!({"type": "column_block", "axis": "z", "line_length": 4,
               "block_vars": ["u"], "pattern": "lower", "bandwidth": [3, 0]})
    );
    // without the axis the block is unchanged
    let plain = jacobian_block(&file, None, Wrt::States, true, 12).unwrap();
    assert_eq!(plain["structure"], json!("general"));
    assert!(plain.get("factorization").is_none());
    // a one-step stencil along z: banded lines
    let sten = faq(
        &["i", "k"],
        &[("i", 1, 3), ("k", 2, 4)],
        op(
            "*",
            &[
                json!("w"),
                idx("u", &[json!("i"), op("-", &[json!("k"), json!(1)])]),
            ],
        ),
        json!({}),
    );
    let file2 = doc(op("+", &[sten, loc.clone()]), d());
    let sv2 = sysview(&file2, None).unwrap();
    let ctx2 = ctx_of(&sv2).unwrap();
    let e2 = jacobian_bands(&file2, None, Wrt::States).unwrap();
    let cl2 = column_lines(&e2, &sv2, &ctx2, "z").unwrap().unwrap();
    assert_eq!((cl2.pattern, cl2.kl, cl2.ku), ("banded", 1, 0));
    assert_eq!(
        detect_structure_along(&e2, &sv2, &ctx2, Some("z")).unwrap(),
        "column_block"
    );
    // a chemistry-shaped (per-cell) Jacobian stays block-diagonal
    let file3 = doc(loc, d());
    let sv3 = sysview(&file3, None).unwrap();
    let ctx3 = ctx_of(&sv3).unwrap();
    let e3 = jacobian_bands(&file3, None, Wrt::States).unwrap();
    assert_eq!(
        detect_structure_along(&e3, &sv3, &ctx3, Some("z")).unwrap(),
        "block_diagonal"
    );
}

#[test]
fn unresolvable_column_is_refused() {
    // A column whose variable the layout does not hold at all is refused at
    // assembly; a cell past a state's extent is a ghost read.
    let file = doc(scan(cmp("<=", "j", "i")), Doc::default());
    let env = Env {
        states: HashMap::from([("u".to_string(), json!(U5))]),
        params: HashMap::new(),
        t: 0.0,
    };
    let sv = sysview(&file, None).unwrap();
    let mut entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    let values = |es: &[JacEntry]| doc_jacobian_entries(&sv, es, Wrt::States, None, &env);
    let n0 = values(&entries).unwrap().len();
    let mut ghost = entries.clone();
    ghost[0].band.cidx[0] = op("+", &[json!("j"), json!(10)]).to_expr();
    assert!(
        values(&ghost).unwrap().len() < n0,
        "past the extent: ghost reads"
    );
    entries[0].v = "V".into();
    let err = values(&entries).unwrap_err();
    assert!(err.0.contains("names no state"), "{}", err.0);
}

trait ToExpr {
    fn to_expr(&self) -> earthsci_ast::Expr;
}

impl ToExpr for Value {
    fn to_expr(&self) -> earthsci_ast::Expr {
        serde_json::from_value(self.clone()).unwrap()
    }
}
