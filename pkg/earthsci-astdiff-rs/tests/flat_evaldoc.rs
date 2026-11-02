//! The evaluation document of a flattened system (spec §8.1; the
//! reference's flat_evaldoc_test.jl): its scoped names declared in the
//! component tree, byte-identical to the reference's goldens
//! (`coupled_chem_advection.evaldoc[-nocse].json`), and loaded and built by
//! EarthSciAST (`rhs_program(esm_problem(…))`, as a runtime builds it), whose
//! right-hand side gives the band values of `coupled_chem_advection.jvals.json`
//! once each band point places its field cell (the consumer's work, done here
//! to check the document by value).

use std::collections::{BTreeMap, HashMap};
use std::fs;

use earthsci_ast::{esm_problem, rhs_program, Compiler, FlattenedSystem, ProblemOptions};
use earthsci_astdiff::points::{cellname, foreach_point, outside};
use earthsci_astdiff::system::{observed_defs, view_states};
use earthsci_astdiff::{
    ctx_of, evaluation_document, jacobian_bands_view, jvp_document, serialize_band, stable_json,
    sysview_flat, vjp_document, IoMap, ProductOptions, SysView, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

const NAME: &str = "coupled_chem_advection";

fn golden(file: &str) -> String {
    fs::read_to_string(root().join("tests/goldens").join(file)).unwrap()
}

fn flat_of(doc: &Value) -> FlattenedSystem {
    let file = earthsci_ast::load_document(doc).expect("document loads");
    earthsci_ast::flatten(&file).expect("document flattens")
}

#[test]
fn goldens_and_layout() {
    let flat = earthsci_ast::flatten(&load(NAME)).unwrap();
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    for (cse, tag) in [(true, "evaldoc"), (false, "evaldoc-nocse")] {
        let (doc, fields) = evaluation_document(&sv, &entries, cse, 12).unwrap();
        assert_eq!(
            stable_json(&doc),
            golden(&format!("{NAME}.{tag}.json")),
            "{tag} diverges"
        );
        let want: Vec<String> = (1..=entries.len()).map(|k| format!("J_{k}")).collect();
        assert_eq!(fields, want);
        let models = doc["models"].as_object().unwrap();
        let mut keys: Vec<&String> = models.keys().collect();
        keys.sort();
        assert_eq!(keys, ["Advection", "JacobianEval", "SimpleOzone"]);
        for m in models.values() {
            assert!(m["variables"]
                .as_object()
                .unwrap()
                .keys()
                .all(|n| !n.contains('.')));
        }
        assert_eq!(
            doc["models"]["SimpleOzone"]["variables"]["O3"]["shape"],
            json!(["lon", "lat"])
        );
        let back = flat_of(&doc);
        for n in flat.state_variables.keys().chain(flat.parameters.keys()) {
            assert!(
                back.state_variables.contains_key(n) || back.parameters.contains_key(n),
                "{n} lost"
            );
        }
        for f in &fields {
            assert!(back
                .state_variables
                .contains_key(&format!("JacobianEval.{f}")));
        }
    }
}

/// The flattened coupled document's bands on the axes it has bands on (its
/// pointwise-lifted chemistry read as the grid arrays it is), byte-identical
/// to the reference's `coupled_chem_advection[.parameters].bands.json`.
#[test]
fn band_goldens() {
    let flat = earthsci_ast::flatten(&load(NAME)).unwrap();
    let sv = sysview_flat(&flat).unwrap();
    for (wrt, tag) in [(Wrt::States, ""), (Wrt::Parameters, ".parameters")] {
        let entries = jacobian_bands_view(&sv, wrt, None).unwrap();
        assert!(!entries.is_empty(), "{wrt:?}: no bands");
        let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
        assert_eq!(
            text,
            golden(&format!("{NAME}{tag}.bands.json")),
            "{wrt:?} diverges"
        );
    }
}

/// `J` at the state cells `u` from one right-hand-side call of the
/// evaluation document of `flat`, each band point adding its field cell into
/// its (row cell, column cell); a ghost read has no cell.
fn band_jacobian(
    flat: &FlattenedSystem,
    u: &HashMap<String, f64>,
    t: f64,
    cse: bool,
    compiler: Compiler,
) -> BTreeMap<(String, String), f64> {
    let sv = sysview_flat(flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, fields) = evaluation_document(&sv, &entries, cse, 12).unwrap();
    let file = earthsci_ast::load_document(&doc).expect("the evaluation document loads");
    let opts = ProblemOptions {
        compiler: Some(compiler),
        ..Default::default()
    };
    let prog = rhs_program(&esm_problem(&file, (0.0, 1.0), opts).unwrap()).unwrap();
    // every cell of every state, by name: (offset, column-major shape)
    let blocks: HashMap<String, (usize, Vec<usize>)> = prog
        .state_layout()
        .iter()
        .map(|b| (b.name.clone(), (b.flat_offset, b.shape.clone())))
        .collect();
    let slot = |v: &str, idx: &[i64]| -> usize {
        let (off, shape) = blocks
            .get(v)
            .unwrap_or_else(|| panic!("no state `{v}` in the document"));
        let (mut k, mut stride) = (0, 1);
        for (i, n) in idx.iter().zip(shape) {
            k += (*i as usize - 1) * stride;
            stride *= n;
        }
        off + k
    };
    let (mut x, aux) = prog.default_state();
    for (c, v) in u {
        let (name, idx) = match c.find('[') {
            None => (c.as_str(), Vec::new()),
            Some(k) => (
                &c[..k],
                c[k + 1..c.len() - 1]
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect(),
            ),
        };
        x[slot(name, &idx)] = *v;
    }
    let p = prog.parameter_vector(&HashMap::new()).unwrap();
    let mut du = vec![0.0; x.len()];
    let mut ws = prog.new_workspace();
    prog.eval_into(&x, &aux, &p, t, &prog.default_forcing(), &mut du, &mut ws)
        .unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let mut j = BTreeMap::new();
    for (en, nm) in entries.iter().zip(&fields) {
        let shape = ctx.shapes.get(&en.v);
        foreach_point(&en.band, &mut |cell, cc, _, cidx| {
            if !outside(shape, cidx) {
                let mut fc = cell.to_vec();
                fc.extend_from_slice(cc);
                let val = du[slot(&format!("JacobianEval.{nm}"), &fc)];
                *j.entry((cellname(&en.u, cell), cellname(&en.v, cidx)))
                    .or_insert(0.0) += val;
            }
            Ok(())
        })
        .unwrap();
    }
    j
}

#[test]
fn values_match_the_golden() {
    let flat = earthsci_ast::flatten(&load(NAME)).unwrap();
    let g: Value = serde_json::from_str(&golden(&format!("{NAME}.jvals.json"))).unwrap();
    let u: HashMap<String, f64> = g["u"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
        .collect();
    let sv = sysview_flat(&flat).unwrap();
    let states = view_states(&sv, &observed_defs(&sv).unwrap());
    let ctx = ctx_of(&sv).unwrap();
    let ncells: usize = states
        .iter()
        .map(|s| {
            ctx.shapes
                .get(s)
                .map_or(1, |sh| sh.iter().product::<i64>() as usize)
        })
        .sum();
    assert_eq!(u.len(), ncells, "the golden pins every state cell");
    let want: BTreeMap<(String, String), f64> = g["entries"]
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
        .collect();
    let t = g["t"].as_f64().unwrap();
    for cse in [true, false] {
        for compiler in [Compiler::Native, Compiler::Interpreter] {
            let got = band_jacobian(&flat, &u, t, cse, compiler);
            assert_eq!(
                got.keys().collect::<Vec<_>>(),
                want.keys().collect::<Vec<_>>(),
                "cse={cse} {compiler:?}: key set"
            );
            for (k, w) in &want {
                let v = got[k];
                assert!(
                    (v - w).abs() <= 1e-10 * w.abs() + 1e-13,
                    "cse={cse} {compiler:?}: {k:?} {v} vs {w}"
                );
            }
        }
    }
}

/// Two coupled models, one with a nested subsystem: a name three deep
/// (`Chem.Inner.k`) goes into `subsystems`, and the evaluation document
/// gives the reference's J (checked there against ForwardDiff):
/// `D(Chem.c) = -k c² s`, `D(Inner.k) = -0.1 k`, `D(Source.s) = -q s`.
#[test]
fn nested_and_coupled_components() {
    let flat = flat_of(&json!({
        "esm": "1.1.0",
        "metadata": {"name": "Nested"},
        "models": {
            "Chem": {
                "variables": {
                    "c": {"type": "unknown", "default": 1.0},
                    "r": {"type": "parameter", "default": 0.5}},
                "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [
                                   {"op": "-", "args": ["Inner.k"]}, "c", "c", "r"]}}],
                "subsystems": {"Inner": {
                    "variables": {"k": {"type": "unknown", "default": 0.3}},
                    "equations": [{"lhs": {"op": "D", "args": ["k"], "wrt": "t"},
                                   "rhs": {"op": "*", "args": [-0.1, "k"]}}]}}},
            "Source": {
                "variables": {
                    "s": {"type": "unknown", "default": 2.0},
                    "q": {"type": "parameter", "default": 0.1}},
                "equations": [{"lhs": {"op": "D", "args": ["s"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [
                                   {"op": "-", "args": ["q"]}, "s"]}}]}},
        "coupling": [{"type": "variable_map", "from": "Source.s", "to": "Chem.r",
                      "transform": "param_to_var"}]
    }));
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, _) = evaluation_document(&sv, &entries, false, 12).unwrap();
    assert!(doc["models"]["Chem"]["subsystems"]["Inner"]["variables"]
        .get("k")
        .is_some());
    let (k, c, s) = (0.5, 0.6, 0.7);
    let u: HashMap<String, f64> = [("Chem.Inner.k", k), ("Chem.c", c), ("Source.s", s)]
        .into_iter()
        .map(|(n, v)| (n.to_string(), v))
        .collect();
    let got = band_jacobian(&flat, &u, 0.0, false, Compiler::Native);
    let want: BTreeMap<(String, String), f64> = [
        (("Chem.c", "Chem.c"), -2.0 * k * c * s),
        (("Chem.c", "Chem.Inner.k"), -c * c * s),
        (("Chem.c", "Source.s"), -k * c * c),
        (("Chem.Inner.k", "Chem.Inner.k"), -0.1),
        (("Source.s", "Source.s"), -0.1),
    ]
    .into_iter()
    .map(|((r, c), v)| ((r.to_string(), c.to_string()), v))
    .collect();
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>()
    );
    for (key, w) in &want {
        assert!((got[key] - w).abs() <= 1e-12 * w.abs(), "{key:?}");
    }
}

#[test]
fn refuses_a_component_named_like_the_derived_model() {
    let flat = flat_of(&json!({
        "esm": "1.1.0",
        "metadata": {"name": "Clash"},
        "models": {"JacobianEval": {
            "variables": {"c": {"type": "unknown", "default": 1.0}},
            "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                           "rhs": {"op": "*", "args": [-1.0, "c", "c"]}}]}}
    }));
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    assert!(evaluation_document(&sv, &entries, true, 12).is_err());
}

/// J_1 of an evaluation document at its default state and time `t`.
fn j1(doc: &Value, t: f64) -> f64 {
    let file = earthsci_ast::load_document(doc).expect("the evaluation document loads");
    let opts = ProblemOptions {
        compiler: Some(Compiler::Interpreter),
        ..Default::default()
    };
    let prog = rhs_program(&esm_problem(&file, (0.0, 1.0), opts).unwrap()).unwrap();
    let (x, aux) = prog.default_state();
    let p = prog.parameter_vector(&HashMap::new()).unwrap();
    let mut du = vec![0.0; x.len()];
    let mut ws = prog.new_workspace();
    prog.eval_into(&x, &aux, &p, t, &prog.default_forcing(), &mut du, &mut ws)
        .unwrap();
    let b = prog
        .state_layout()
        .iter()
        .find(|b| b.name == "JacobianEval.J_1")
        .expect("a band field");
    du[b.flat_offset]
}

/// The evaluation document carries the source's `domain` (spec §8.1), so it
/// has the source's time origin; the right-hand side passes `t` to the
/// calendar functions as UTC seconds whatever the origin, so a
/// `datetime.hour` coefficient reads the same with or without it (the
/// reference's flat_evaldoc_test.jl).
#[test]
fn carries_the_domain() {
    let src = json!({
        "esm": "1.1.0",
        "metadata": {"name": "Diurnal"},
        "domain": {"independent_variable": "t",
                   "temporal": {"start": "2024-05-01T06:00:00Z", "end": "2024-05-02T06:00:00Z"}},
        "models": {"Chem": {
            "variables": {
                "c": {"type": "unknown", "default": 1.0},
                "k": {"type": "parameter", "default": 0.5}},
            "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                           "rhs": {"op": "*", "args": [
                               {"op": "-", "args": ["k"]},
                               {"op": "fn", "name": "datetime.hour", "args": ["t"]}, "c"]}}]}}
    });
    let file = earthsci_ast::load_document(&src).unwrap();
    let flat = earthsci_ast::flatten(&file).unwrap();
    let views = [
        earthsci_astdiff::sysview(&file, None).unwrap(),
        sysview_flat(&flat).unwrap(),
    ];
    for sv in &views {
        let entries = jacobian_bands_view(sv, Wrt::States, None).unwrap();
        let (doc, fields) = evaluation_document(sv, &entries, true, 12).unwrap();
        assert_eq!(fields, ["J_1"]);
        assert_eq!(
            stable_json(&doc["domain"]),
            r#"{"temporal":{"end":"2024-05-02T06:00:00Z","start":"2024-05-01T06:00:00Z"}}"#
        );
        assert_eq!(doc["esm"], json!("1.1.0"));
        // 13:00 UTC on 2024-05-01, in UTC seconds: -k · 13
        assert_eq!(j1(&doc, 1714568400.0), -6.5);
        let mut bare = doc.clone();
        bare.as_object_mut().unwrap().remove("domain");
        assert_eq!(j1(&bare, 1714568400.0), -6.5);
    }
    let bd = load("bd_chem");
    let sv = earthsci_astdiff::sysview(&bd, None).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, _) = evaluation_document(&sv, &entries, true, 12).unwrap();
    assert!(doc.get("domain").is_none());
}

/// One right-hand-side call of a derived document of a flattened system,
/// built as a runtime builds it: `u` sets the state cells by their flattened
/// names (the source's, and the derived model's inputs as
/// `<model>.<input>[…]`). Returns a lookup of `du` by flattened cell name.
fn doc_rhs(
    doc: &Value,
    u: &HashMap<String, f64>,
    t: f64,
    compiler: Compiler,
) -> impl Fn(&str) -> f64 {
    let file = earthsci_ast::load_document(doc).expect("the document loads");
    let opts = ProblemOptions {
        compiler: Some(compiler),
        ..Default::default()
    };
    let prog = rhs_program(&esm_problem(&file, (0.0, 1.0), opts).unwrap()).unwrap();
    let blocks: HashMap<String, (usize, Vec<usize>)> = prog
        .state_layout()
        .iter()
        .map(|b| (b.name.clone(), (b.flat_offset, b.shape.clone())))
        .collect();
    let slot = move |c: &str| -> usize {
        let (name, idx): (&str, Vec<usize>) = match c.find('[') {
            None => (c, Vec::new()),
            Some(k) => (
                &c[..k],
                c[k + 1..c.len() - 1]
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect(),
            ),
        };
        let (off, shape) = blocks
            .get(name)
            .unwrap_or_else(|| panic!("no state `{name}` in the document"));
        let (mut k, mut stride) = (0, 1);
        for (i, n) in idx.iter().zip(shape) {
            k += (i - 1) * stride;
            stride *= n;
        }
        off + k
    };
    let (mut x, aux) = prog.default_state();
    for (c, v) in u {
        x[slot(c)] = *v;
    }
    let p = prog.parameter_vector(&HashMap::new()).unwrap();
    let mut du = vec![0.0; x.len()];
    let mut ws = prog.new_workspace();
    prog.eval_into(&x, &aux, &p, t, &prog.default_forcing(), &mut du, &mut ws)
        .unwrap();
    move |c: &str| du[slot(c)]
}

/// The derived model's name for a cell of a source variable, flattened.
fn io_cell(model: &str, io: &IoMap, cell: &str) -> String {
    match cell.find('[') {
        None => format!("{model}.{}", io[cell]),
        Some(k) => format!("{model}.{}{}", io[&cell[..k]], &cell[k..]),
    }
}

fn layout_ok(doc: &Value, model: &str) -> bool {
    let ms = doc["models"].as_object().unwrap();
    let mut keys: Vec<&str> = ms.keys().map(String::as_str).collect();
    keys.sort();
    let mut want = vec!["Advection", "SimpleOzone", model];
    want.sort();
    keys == want
        && ms.values().all(|m| {
            m["variables"]
                .as_object()
                .unwrap()
                .keys()
                .all(|n| !n.contains('.'))
        })
}

/// The product documents (spec §9) of a flattened system are laid out as its
/// component tree too, their inputs and outputs unscoped in the derived model
/// (`_v_SimpleOzone_O3`); each loads and gives `J v` and `Jᵀ λ` of the
/// golden Jacobian (the reference's flat_evaldoc_test.jl).
#[test]
fn product_documents() {
    let flat = earthsci_ast::flatten(&load(NAME)).unwrap();
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let g: Value = serde_json::from_str(&golden(&format!("{NAME}.jvals.json"))).unwrap();
    let u: HashMap<String, f64> = g["u"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
        .collect();
    let t = g["t"].as_f64().unwrap();
    let want: Vec<(String, String, f64)> = g["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e[0].as_str().unwrap().to_string(),
                e[1].as_str().unwrap().to_string(),
                e[2].as_f64().unwrap(),
            )
        })
        .collect();
    let mut cells: Vec<&String> = u.keys().collect();
    cells.sort();
    let v: HashMap<&str, f64> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), (1.7 * (i + 1) as f64).sin()))
        .collect();
    let lam: HashMap<&str, f64> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), (0.9 * (i + 1) as f64).cos()))
        .collect();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * b.abs() + 1e-13;
    for cse in [true, false] {
        let opts = ProductOptions {
            cse,
            ..Default::default()
        };
        let (jv, io) = jvp_document(&sv, &entries, Wrt::States, &opts).unwrap();
        assert!(layout_ok(&jv, "JvpEval"));
        assert_eq!(io.v["SimpleOzone.O3"], "_v_SimpleOzone_O3");
        assert_eq!(io.jvp["SimpleOzone.O3"], "_jvp_SimpleOzone_O3");
        let mut x = u.clone();
        for c in &cells {
            x.insert(io_cell(&io.model, &io.v, c), v[c.as_str()]);
        }
        let out = doc_rhs(&jv, &x, t, Compiler::Native);
        for r in &cells {
            let exp: f64 = want
                .iter()
                .filter(|(rr, _, _)| rr == *r)
                .map(|(_, c, w)| w * v[c.as_str()])
                .sum();
            let got = out(&io_cell(&io.model, &io.jvp, r));
            assert!(close(got, exp), "cse={cse} Jv[{r}]: {got} vs {exp}");
        }
        let (vj, io) = vjp_document(&sv, &entries, Wrt::States, &opts).unwrap();
        assert!(layout_ok(&vj, "VjpEval"));
        let mut x = u.clone();
        for r in &cells {
            x.insert(io_cell(&io.model, &io.lam, r), lam[r.as_str()]);
        }
        let out = doc_rhs(&vj, &x, t, Compiler::Interpreter);
        for c in &cells {
            let exp: f64 = want
                .iter()
                .filter(|(_, cc, _)| cc == *c)
                .map(|(r, _, w)| lam[r.as_str()] * w)
                .sum();
            let got = out(&io_cell(&io.model, &io.vjp, c));
            assert!(close(got, exp), "cse={cse} Jᵀλ[{c}]: {got} vs {exp}");
        }
    }
}

/// Refused (spec §8.1): a flattened system mixing scoped and unscoped
/// names, a scoped name that is both a variable and a component, and a
/// source variable named like a band field.
#[test]
fn refuses_bad_sources() {
    let flat = earthsci_ast::flatten(&load(NAME)).unwrap();
    let sv = sysview_flat(&flat).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let par = sv
        .variables
        .keys()
        .filter(|n| n.starts_with("Advection."))
        .min()
        .unwrap()
        .clone();
    let with = |extra: &[&str]| -> SysView {
        let mut variables = sv.variables.clone();
        for n in extra {
            variables.insert(n.to_string(), sv.variables[&par].clone());
        }
        SysView {
            variables,
            equations: sv.equations,
            index_sets: sv.index_sets.clone(),
            domain: sv.domain,
        }
    };
    let err = evaluation_document(&with(&["x"]), &entries, true, 12).unwrap_err();
    assert!(err.0.contains("`x`"), "{}", err.0);
    let err = evaluation_document(
        &with(&["Advection.Inner", "Advection.Inner.q"]),
        &entries,
        true,
        12,
    )
    .unwrap_err();
    assert!(
        err.0
            .contains("`Advection.Inner` is both a variable and a component"),
        "{}",
        err.0
    );
    let clash = earthsci_ast::load_document(&json!({
        "esm": "1.1.0",
        "metadata": {"name": "Clash"},
        "models": {"JacobianEval": {
            "variables": {"J_1": {"type": "unknown", "default": 1.0}},
            "equations": [{"lhs": {"op": "D", "args": ["J_1"], "wrt": "t"},
                           "rhs": {"op": "*", "args": [-1.0, "J_1", "J_1"]}}]}}
    }))
    .unwrap();
    let sv = earthsci_astdiff::sysview(&clash, None).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let err = evaluation_document(&sv, &entries, true, 12).unwrap_err();
    assert!(err.0.contains("band field 1"), "{}", err.0);
}

/// Every product document carries the source's `domain`, as the evaluation
/// document does (spec §8.1, §9.1).
#[test]
fn product_documents_carry_the_domain() {
    let src = json!({
        "esm": "1.1.0",
        "metadata": {"name": "Diurnal"},
        "domain": {"independent_variable": "t",
                   "temporal": {"start": "2024-05-01T06:00:00Z", "end": "2024-05-02T06:00:00Z"}},
        "models": {"Chem": {
            "variables": {
                "c": {"type": "unknown", "default": 1.0},
                "k": {"type": "parameter", "default": 0.5}},
            "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                           "rhs": {"op": "*", "args": [
                               {"op": "-", "args": ["k"]},
                               {"op": "fn", "name": "datetime.hour", "args": ["t"]}, "c"]}}]}}
    });
    let file = earthsci_ast::load_document(&src).unwrap();
    let flat = earthsci_ast::flatten(&file).unwrap();
    let want = r#"{"temporal":{"end":"2024-05-02T06:00:00Z","start":"2024-05-01T06:00:00Z"}}"#;
    let opts = ProductOptions::default();
    for sv in [
        earthsci_astdiff::sysview(&file, None).unwrap(),
        sysview_flat(&flat).unwrap(),
    ] {
        let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
        let (jv, _) = jvp_document(&sv, &entries, Wrt::States, &opts).unwrap();
        let (vj, _) = vjp_document(&sv, &entries, Wrt::States, &opts).unwrap();
        assert_eq!(stable_json(&jv["domain"]), want);
        assert_eq!(stable_json(&vj["domain"]), want);
    }
    let bd = load("bd_chem");
    let sv = earthsci_astdiff::sysview(&bd, None).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (jv, _) = jvp_document(&sv, &entries, Wrt::States, &opts).unwrap();
    assert!(jv.get("domain").is_none());
}

// ── a lifted state on one of two same-size index sets ─────────────────────
//
// `lift_same_size_index_sets` (EarthSciAST's pointwise-lift conformance case
// of that name): `y` and `x` both have size 3, `y` declared first, and the
// operand `Wind.u`, declared on `[x, y]` and gathered at the lift's loops,
// names the grid. EarthSciAST's `flatten` records `Chem.c` on `[x, y]` from
// 0.8 (esm-libraries-spec §4.7.5); the view, its context, the bands and the
// evaluation document take that shape as recorded, byte-identical to the
// reference's goldens. A `flatten` that records no lifted shape (before 0.8)
// leaves the view to its fallback, which cannot tell `x` from `y`, so the
// goldens do not apply there and the test returns early.

const LIFT: &str = "lift_same_size_index_sets";

/// The flattened fixture, or `None` when this EarthSciAST's `flatten`
/// records no lifted shape.
fn lift_flat() -> Option<FlattenedSystem> {
    let flat = earthsci_ast::flatten(&load(LIFT)).unwrap();
    if flat.state_variables["Chem.c"]
        .shape
        .as_ref()
        .is_none_or(|s| s.is_empty())
    {
        eprintln!("skipped: this earthsci-ast's flatten records no lifted shape (before 0.8)");
        return None;
    }
    Some(flat)
}

#[test]
fn lifted_shape_is_flattens() {
    let Some(flat) = lift_flat() else { return };
    let xy = json!(["x", "y"]);
    assert_eq!(json!(flat.state_variables["Chem.c"].shape), xy);
    let sv = sysview_flat(&flat).unwrap();
    assert_eq!(json!(sv.variables["Chem.c"].shape), xy);
    assert_eq!(ctx_of(&sv).unwrap().shapes.get("Chem.c"), Some(&vec![3, 3]));
    assert!(sv.index_sets.keys().all(|k| !k.starts_with("_liftdim")));
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    // nothing to hoist: the unhoisted document is the same (no nocse golden)
    for cse in [true, false] {
        let (doc, fields) = evaluation_document(&sv, &entries, cse, 12).unwrap();
        assert_eq!(
            stable_json(&doc),
            golden(&format!("{LIFT}.evaldoc.json")),
            "cse = {cse} diverges"
        );
        assert_eq!(doc["models"]["Chem"]["variables"]["c"]["shape"], xy);
        for f in &fields {
            assert_eq!(doc["models"]["JacobianEval"]["variables"][f]["shape"], xy);
        }
    }
}

#[test]
fn lifted_shape_values() {
    let Some(flat) = lift_flat() else { return };
    let g: Value = serde_json::from_str(&golden(&format!("{LIFT}.jvals.json"))).unwrap();
    let u: HashMap<String, f64> = g["u"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
        .collect();
    assert_eq!(u.len(), 9, "the golden pins every state cell");
    let want: BTreeMap<(String, String), f64> = g["entries"]
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
        .collect();
    let t = g["t"].as_f64().unwrap();
    for compiler in [Compiler::Native, Compiler::Interpreter] {
        let got = band_jacobian(&flat, &u, t, true, compiler);
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>()
        );
        for (k, w) in &want {
            assert!(
                (got[k] - w).abs() <= 1e-10 * w.abs() + 1e-13,
                "{compiler:?}: {k:?}"
            );
        }
    }
}
