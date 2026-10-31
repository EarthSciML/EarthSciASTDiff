//! Product documents (spec §9): `J v`, `Jᵀ λ` and the §4.4 contraction as
//! derived ESM documents. Pinned byte-for-byte by the reference's
//! tests/goldens/<fixture>[.<axis>].{jvp,vjp}.evaldoc.json and
//! <fixture>.<a1>-<a2>.{hvp,hvp-sym}.evaldoc.json, and checked by value: each
//! document is evaluated by EarthSciAST and compared with the product formed
//! from the value goldens themselves (`*.jvals.json`, `*.hvals.json`), so no
//! evaluator of this crate takes part. Mirrors the reference's
//! test/products_test.jl.

use std::collections::HashMap;
use std::fs;

use earthsci_ast::{esm_problem, rhs_program, Compiler, ProblemOptions};
use earthsci_astdiff::{
    column_names, hvp_document, jacobian_bands2, jacobian_bands_view, jacobian_columns,
    jvp_document, stable_json, sysview, vjp_document, HvpForm, IoMap, ProductOptions, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

const SS: (Wrt, Wrt) = (Wrt::States, Wrt::States);

/// Every axis with a value golden (spec §9.6).
const PRODUCT_AXES: [(&str, Wrt); 13] = [
    ("bd_chem", Wrt::States),
    ("adv_interior", Wrt::States),
    ("contracted_ops", Wrt::States),
    ("flux_form_adv", Wrt::States),
    ("minmod_adv", Wrt::States),
    ("contracting_observed", Wrt::States),
    ("tgrad_photolysis", Wrt::States),
    ("forcing_bracket", Wrt::States),
    ("filtered_scan", Wrt::States),
    ("semiring_reduce", Wrt::States),
    ("tgrad_photolysis", Wrt::Time),
    ("forcing_bracket", Wrt::Forcing),
    ("forcing_bracket", Wrt::Parameters),
];

const SECOND_ORDER: [(&str, (Wrt, Wrt)); 8] = [
    ("bd_chem", SS),
    ("arrhenius_chem", SS),
    ("arrhenius_chem", (Wrt::States, Wrt::Parameters)),
    ("forcing_bracket", (Wrt::States, Wrt::Forcing)),
    ("forcing_bracket", (Wrt::Time, Wrt::States)),
    ("minmod_adv", SS),
    ("filtered_scan", SS),
    ("semiring_reduce", SS),
];

fn axistag(wrt: Wrt) -> String {
    if wrt == Wrt::States {
        String::new()
    } else {
        format!(".{}", wrt.as_str())
    }
}

fn pairtag((a1, a2): (Wrt, Wrt)) -> String {
    format!("{}-{}", a1.as_str(), a2.as_str())
}

fn golden(file: &str) -> String {
    fs::read_to_string(root().join("tests/goldens").join(file)).unwrap()
}

fn forms(wrt: (Wrt, Wrt)) -> &'static [HvpForm] {
    if wrt == SS {
        &[HvpForm::Contract, HvpForm::Symmetric]
    } else {
        &[HvpForm::Contract]
    }
}

#[test]
fn product_goldens() {
    let opts = ProductOptions::default();
    for (name, wrt) in PRODUCT_AXES {
        let file = load(name);
        let sv = sysview(&file, None).unwrap();
        let entries = jacobian_bands_view(&sv, wrt, None).unwrap();
        let (doc, _) = jvp_document(&sv, &entries, wrt, &opts).unwrap();
        let tag = axistag(wrt);
        assert_eq!(
            stable_json(&doc),
            golden(&format!("{name}{tag}.jvp.evaldoc.json")),
            "jvp golden diverges: {name}{tag}"
        );
        let (doc, _) = vjp_document(&sv, &entries, wrt, &opts).unwrap();
        assert_eq!(
            stable_json(&doc),
            golden(&format!("{name}{tag}.vjp.evaldoc.json")),
            "vjp golden diverges: {name}{tag}"
        );
    }
    for (name, wrt) in SECOND_ORDER {
        let file = load(name);
        let sv = sysview(&file, None).unwrap();
        let entries = jacobian_bands2(&file, None, wrt).unwrap();
        for &form in forms(wrt) {
            let (doc, _) = hvp_document(&sv, &entries, wrt, 2, form, &opts).unwrap();
            let kind = if form == HvpForm::Contract {
                "hvp"
            } else {
                "hvp-sym"
            };
            let tag = pairtag(wrt);
            assert_eq!(
                stable_json(&doc),
                golden(&format!("{name}.{tag}.{kind}.evaldoc.json")),
                "{kind} golden diverges: {name}.{tag}"
            );
        }
    }
}

// ── evaluating a document with EarthSciAST ──────────────────────────────────

/// A cell name (`"c[2]"`, `"S"`) split into its variable and 1-based index.
fn split_cell(cell: &str) -> (&str, Vec<usize>) {
    match cell.find('[') {
        None => (cell, Vec::new()),
        Some(k) => (
            &cell[..k],
            cell[k + 1..cell.len() - 1]
                .split(',')
                .map(|x| x.parse().unwrap())
                .collect(),
        ),
    }
}

/// The document cell carrying `cell` of the variable `names` maps.
fn doc_cell(names: &IoMap, cell: &str) -> String {
    let (v, idx) = split_cell(cell);
    let nm = &names[v];
    if idx.is_empty() {
        nm.clone()
    } else {
        format!("{nm}{}", &cell[v.len()..])
    }
}

/// Evaluate `doc`'s right-hand side once with EarthSciAST's interpreter (an
/// equality-filtered `faq` is refused by the native compiler until it
/// implements §9.3's equality-filter scatter) at the cell-named states `u`
/// plus `inputs`, the declared parameter defaults (which the goldens pin)
/// and time `t`. Returns `du` by cell name.
fn eval_product(
    doc: &Value,
    model: &str,
    u: &Value,
    inputs: &HashMap<String, f64>,
    params: Option<&Value>,
    t: f64,
) -> impl Fn(&str) -> f64 {
    let file = earthsci_ast::load_document(doc).unwrap();
    let opts = ProblemOptions {
        compiler: Some(Compiler::Interpreter),
        ..Default::default()
    };
    let prog = rhs_program(&esm_problem(&file, (0.0, 1.0), opts).unwrap()).unwrap();
    let blocks: HashMap<String, (usize, Vec<usize>)> = prog
        .state_layout()
        .iter()
        .map(|b| {
            let nm = b.name.strip_prefix(&format!("{model}.")).unwrap_or(&b.name);
            (nm.to_string(), (b.flat_offset, b.shape.clone()))
        })
        .collect();
    let owner = model.to_string();
    let slot = move |cell: &str| -> usize {
        let (v, idx) = split_cell(cell);
        let (off, shape) = blocks
            .get(v)
            .unwrap_or_else(|| panic!("no state `{v}` in {owner}"));
        // column-major, first index fastest
        let (mut k, mut stride) = (0, 1);
        for (i, n) in idx.iter().zip(shape) {
            k += (i - 1) * stride;
            stride *= n;
        }
        off + k
    };
    let (mut x, aux) = prog.default_state();
    for (c, v) in u.as_object().unwrap() {
        x[slot(c)] = v.as_f64().unwrap();
    }
    for (c, v) in inputs {
        x[slot(c)] = *v;
    }
    let p = prog.parameter_vector(&HashMap::new()).unwrap();
    if let Some(params) = params {
        for (k, v) in params.as_object().unwrap() {
            let i = prog
                .parameters()
                .iter()
                .position(|s| s.name == format!("{model}.{k}") || s.name == *k)
                .unwrap_or_else(|| panic!("no parameter `{k}`"));
            assert_eq!(p[i], v.as_f64().unwrap(), "the goldens pin the defaults");
        }
    }
    let mut du = vec![0.0; x.len()];
    let mut ws = prog.new_workspace();
    prog.eval_into(&x, &aux, &p, t, &prog.default_forcing(), &mut du, &mut ws)
        .unwrap();
    move |cell: &str| du[slot(cell)]
}

fn seed(n: usize, a: f64) -> Vec<f64> {
    (1..=n).map(|k| (a * k as f64 + 0.4).sin()).collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-13 + 1e-10 * want.abs(),
        "{what}: {got} vs {want}"
    );
}

#[test]
fn jvp_vjp_against_value_goldens() {
    let opts = ProductOptions::default();
    for (name, wrt) in PRODUCT_AXES {
        for sfx in ["", ".w0"] {
            let path = root()
                .join("tests/goldens")
                .join(format!("{name}{}{sfx}.jvals.json", axistag(wrt)));
            if !path.exists() {
                continue;
            }
            let g: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
            let file = load(name);
            let sv = sysview(&file, None).unwrap();
            let mut rows: Vec<String> = g["u"].as_object().unwrap().keys().cloned().collect();
            rows.sort();
            let cols = match wrt {
                Wrt::States => rows.clone(),
                Wrt::Time => vec!["t".to_string()],
                _ => column_names(&jacobian_columns(&sv, wrt, None).unwrap()),
            };
            let ri: HashMap<&str, usize> = rows
                .iter()
                .enumerate()
                .map(|(i, n)| (n.as_str(), i))
                .collect();
            let ci: HashMap<&str, usize> = cols
                .iter()
                .enumerate()
                .map(|(i, n)| (n.as_str(), i))
                .collect();
            let mut jac = vec![vec![0.0; cols.len()]; rows.len()];
            for e in g["entries"].as_array().unwrap() {
                let (r, c) = (e[0].as_str().unwrap(), e[1].as_str().unwrap());
                jac[ri[r]][ci[c]] += e[2].as_f64().unwrap();
            }
            let v = seed(cols.len(), 1.3);
            let lam = seed(rows.len(), 0.7);
            let t = g["t"].as_f64().unwrap();
            let entries = jacobian_bands_view(&sv, wrt, None).unwrap();

            let (doc, io) = jvp_document(&sv, &entries, wrt, &opts).unwrap();
            let inputs = cols
                .iter()
                .zip(&v)
                .map(|(c, x)| (doc_cell(&io.v, c), *x))
                .collect();
            let jv = eval_product(&doc, &io.model, &g["u"], &inputs, Some(&g["params"]), t);
            for (i, r) in rows.iter().enumerate() {
                let want: f64 = jac[i].iter().zip(&v).map(|(a, b)| a * b).sum();
                close(
                    jv(&doc_cell(&io.jvp, r)),
                    want,
                    &format!("{name}{sfx} Jv {r}"),
                );
            }

            let (doc, io) = vjp_document(&sv, &entries, wrt, &opts).unwrap();
            let inputs = rows
                .iter()
                .zip(&lam)
                .map(|(r, x)| (doc_cell(&io.lam, r), *x))
                .collect();
            let vj = eval_product(&doc, &io.model, &g["u"], &inputs, Some(&g["params"]), t);
            for (j, c) in cols.iter().enumerate() {
                let want: f64 = (0..rows.len()).map(|i| jac[i][j] * lam[i]).sum();
                close(
                    vj(&doc_cell(&io.vjp, c)),
                    want,
                    &format!("{name}{sfx} Jᵀλ {c}"),
                );
            }
        }
    }
}

#[test]
fn contraction_against_value_goldens() {
    let opts = ProductOptions::default();
    for (name, wrt) in SECOND_ORDER {
        let gd: Value =
            serde_json::from_str(&golden(&format!("{name}.{}.hvals.json", pairtag(wrt)))).unwrap();
        let file = load(name);
        let sv = sysview(&file, None).unwrap();
        let entries = jacobian_bands2(&file, None, wrt).unwrap();
        let pairs = gd["pairs"].as_array().unwrap();
        for &form in forms(wrt) {
            let (doc, io) = hvp_document(&sv, &entries, wrt, pairs.len(), form, &opts).unwrap();
            let mut inputs = HashMap::new();
            for (i, pr) in pairs.iter().enumerate() {
                for (r, x) in pr["mu"].as_object().unwrap() {
                    inputs.insert(doc_cell(&io.mu[i], r), x.as_f64().unwrap());
                }
                match &pr["w"] {
                    Value::Object(w) => {
                        for (c, x) in w {
                            inputs.insert(doc_cell(&io.w[i], c), x.as_f64().unwrap());
                        }
                    }
                    x => {
                        inputs.insert(io.w[i]["t"].clone(), x.as_f64().unwrap());
                    }
                }
            }
            let h = eval_product(
                &doc,
                &io.model,
                &gd["u"],
                &inputs,
                None,
                gd["t"].as_f64().unwrap(),
            );
            // `g`: [cell, value] pairs
            let want: HashMap<&str, f64> = gd["g"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| (e[0].as_str().unwrap(), e[1].as_f64().unwrap()))
                .collect();
            let cells = if wrt.1 == Wrt::States {
                let ctx = earthsci_astdiff::ctx_of(&sv).unwrap();
                let vars: Vec<(String, Vec<i64>)> = io
                    .hvp
                    .keys()
                    .map(|n| (n.clone(), ctx.shapes.get(n).cloned().unwrap_or_default()))
                    .collect();
                column_names(&vars)
            } else {
                column_names(&jacobian_columns(&sv, wrt.1, None).unwrap())
            };
            for k in want.keys() {
                assert!(
                    cells.iter().any(|c| c == k),
                    "{name}: golden cell {k} not an output"
                );
            }
            for c in &cells {
                close(
                    h(&doc_cell(&io.hvp, c)),
                    want.get(c.as_str()).copied().unwrap_or(0.0),
                    &format!("{name}.{} {form:?} {c}", pairtag(wrt)),
                );
            }
        }
    }
}

// ── form ────────────────────────────────────────────────────────────────────

fn rhs_of<'a>(doc: &'a Value, model: &str, out: &str) -> &'a Value {
    doc["models"][model]["equations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|eq| eq["lhs"].get("args") == Some(&json!([out])))
        .map(|eq| &eq["rhs"])
        .unwrap()
}

#[test]
fn product_form() {
    let opts = ProductOptions::default();
    let file = load("adv_interior");
    let sv = sysview(&file, None).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, io) = vjp_document(&sv, &entries, Wrt::States, &opts).unwrap();
    assert_eq!(io.model, "VjpEval");
    assert_eq!(io.lam, IoMap::from([("u".into(), "_lam_u".into())]));
    assert_eq!(io.vjp, IoMap::from([("u".into(), "_vjp_u".into())]));
    // the inputs are states with a zero derivative; the source's D(u) is zeroed
    let m = &doc["models"]["VjpEval"];
    for n in ["_lam_u", "_vjp_u"] {
        assert_eq!(
            m["variables"][n],
            json!({"type": "unknown", "default": 0.0, "shape": ["x"]})
        );
    }
    // two entries, a left fold of their scatters, each filtered by its column
    let rhs = rhs_of(&doc, "VjpEval", "_vjp_u");
    assert_eq!(rhs["op"], "+");
    assert_eq!(rhs["args"].as_array().unwrap().len(), 2);
    let f = &rhs["args"][0];
    assert_eq!(f["op"], "faq");
    assert_eq!(f["output_idx"], json!(["_j1"]));
    assert_eq!(f["ranges"], json!({"_j1": [1, 8], "i": [2, 7]}));
    assert_eq!(f["filter"]["op"], "==");
    assert_eq!(f["filter"]["args"][1], "_j1");
    // names already taken get `_` appended
    let mut d2: Value = serde_json::from_str(
        &fs::read_to_string(root().join("tests/valid/adv_interior.esm")).unwrap(),
    )
    .unwrap();
    d2["models"]["Adv"]["variables"]["_v_u"] = json!({"type": "parameter", "default": 0.0});
    let file2 = earthsci_ast::load_document(&d2).unwrap();
    let sv2 = sysview(&file2, None).unwrap();
    let entries2 = jacobian_bands_view(&sv2, Wrt::States, None).unwrap();
    let (_, io2) = jvp_document(&sv2, &entries2, Wrt::States, &opts).unwrap();
    assert_eq!(io2.v["u"], "_v_u_");
    assert_eq!(io2.jvp["u"], "_jvp_u");
}

#[test]
fn ifelse_guard_makes_every_reduced_name_occur() {
    // contracting_observed's third entry reads its contracted name only in
    // its column index: Jᵀλ's body must still mention it
    let file = load("contracting_observed");
    let sv = sysview(&file, None).unwrap();
    let entries = jacobian_bands_view(&sv, Wrt::States, None).unwrap();
    let (doc, io) = vjp_document(&sv, &entries, Wrt::States, &ProductOptions::default()).unwrap();
    let text = stable_json(rhs_of(&doc, &io.model, &io.vjp["u"]));
    assert!(text.contains("\"ifelse\""));
}

#[test]
fn product_refusals() {
    let opts = ProductOptions::default();
    let file = load("bd_chem");
    let sv = sysview(&file, None).unwrap();
    let entries = jacobian_bands2(&file, None, SS).unwrap();
    assert!(hvp_document(&sv, &entries, SS, 0, HvpForm::Contract, &opts).is_err());
    let sp = (Wrt::States, Wrt::Parameters);
    let entries = jacobian_bands2(&file, None, sp).unwrap();
    assert!(hvp_document(&sv, &entries, sp, 2, HvpForm::Symmetric, &opts).is_err());
    // entries built under another forcing set than the document's: with no
    // forcing, `F` is a parameter column; the inferred set makes it forcing
    let file = load("forcing_bracket");
    let sv = sysview(&file, None).unwrap();
    let none: [String; 0] = [];
    let entries = jacobian_bands_view(&sv, Wrt::Parameters, Some(&none)).unwrap();
    assert!(entries.iter().any(|en| en.v == "F"));
    let err = vjp_document(&sv, &entries, Wrt::Parameters, &opts).unwrap_err();
    assert!(err.0.contains("forcing set"), "{}", err.0);
    assert!(jvp_document(&sv, &entries, Wrt::Parameters, &opts).is_err());
    let agree = ProductOptions {
        forcing: Some(&none),
        ..Default::default()
    };
    assert!(vjp_document(&sv, &entries, Wrt::Parameters, &agree).is_ok());
}
