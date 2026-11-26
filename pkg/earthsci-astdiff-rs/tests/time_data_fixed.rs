//! Time bands with data held fixed (spec §4.2): only observed that move with
//! `t` are followed, and an observed declared without a shape is read at a
//! cell. Byte-identical to the Julia reference's golden. And the time
//! companions (`tgrad_named_steps`): the chain through the moving observed
//! kept as named steps, checked against a central difference in `t` of the
//! source right-hand side.

use std::fs;

use earthsci_astdiff::{
    jacobian_bands, jacobian_bands2, jacobian_block, jacobian_block2, jacobian_blocks,
    serialize_band, stable_json, BlockWrt, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

#[test]
fn tgrad_data_fixed_time_bands() {
    let file = load("tgrad_data_fixed");
    let entries = jacobian_bands(&file, Some("Box"), Wrt::Time).unwrap();
    assert_eq!(entries.len(), 1);
    let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
    let want =
        fs::read_to_string(root().join("tests/goldens/tgrad_data_fixed.time.bands.json")).unwrap();
    assert_eq!(text, want);
}

/// tgrad_data_fixed with `edit` applied to the `Box` model.
fn variant(edit: impl FnOnce(&mut Value)) -> earthsci_ast::EsmFile {
    let text = fs::read_to_string(root().join("tests/valid/tgrad_data_fixed.esm")).unwrap();
    let mut d: Value = serde_json::from_str(&text).unwrap();
    edit(&mut d["models"]["Box"]);
    earthsci_ast::load_string(&d.to_string()).expect("variant loads")
}

fn jr_t() -> Value {
    json!({"op": "*", "args": ["J0", {"op": "cos", "args": [{"op": "*", "args": ["omega", "t"]}]}]})
}

#[test]
fn time_pair_follows_what_its_second_axis_seeds() {
    // kc = c·c moves only with the states: the plain time pass leaves
    // `index(kc, i, j)` an opaque read, the (time, states) pair follows it.
    // Each pair keeps its own pass, so the order of emission is immaterial.
    let file = variant(|m| {
        m["variables"]["kc"] = json!({"type": "unknown"});
        m["equations"]
            .as_array_mut()
            .unwrap()
            .push(json!({"lhs": "kc", "rhs": {"op": "*", "args": ["c", "c"]}}));
        let body = m["equations"][3]["rhs"]["expr"].clone();
        m["equations"][3]["rhs"]["expr"] = json!({"op": "+", "args": [body, {"op": "*", "args": [
            {"op": "index", "args": ["kc", "i", "j"]},
            {"op": "cos", "args": [{"op": "*", "args": ["omega", "t"]}]}
        ]}]});
    });
    assert!(
        !jacobian_bands2(&file, Some("Box"), (Wrt::Time, Wrt::States))
            .unwrap()
            .is_empty()
    );
    let pair = jacobian_blocks(
        &file,
        Some("Box"),
        &[(Wrt::Time, Wrt::States)],
        None,
        true,
        12,
    )
    .unwrap();
    let time = jacobian_blocks(&file, Some("Box"), &[Wrt::Time], None, true, 12).unwrap();
    let both = jacobian_blocks(
        &file,
        Some("Box"),
        &[
            BlockWrt::Axis(Wrt::Time),
            BlockWrt::Pair(Wrt::Time, Wrt::States),
        ],
        None,
        true,
        12,
    )
    .unwrap();
    let one = |v: &Value| v.as_array().map_or(v.clone(), |a| a[0].clone());
    assert_eq!(both[1], one(&pair));
    assert_eq!(both[0], one(&time));
}

#[test]
fn elementwise_read_at_a_cell_refuses_what_it_cannot_read() {
    // a partial index (still an array) inside the definition
    let partial = variant(|m| {
        m["equations"][0]["rhs"] =
            json!({"op": "+", "args": [jr_t(), {"op": "index", "args": ["Eb", 1]}]});
    });
    assert!(jacobian_bands(&partial, Some("Box"), Wrt::Time).is_err());
    // a global named like the read site's index `j`: the read is a column
    // (spec §4.2, companions), not inlined, and the companion row of `jr`
    // names its own indices clear of `j`, so nothing is captured
    let capture = variant(|m| {
        m["variables"]["j"] = json!({"type": "parameter", "default": 2.0});
        m["equations"][0]["rhs"] = json!({"op": "*", "args": [jr_t(), "j"]});
    });
    assert_eq!(
        jacobian_bands(&capture, Some("Box"), Wrt::Time)
            .unwrap()
            .len(),
        1
    );
    let block = jacobian_block(&capture, Some("Box"), Wrt::Time, true, 12).unwrap();
    let rhs = &block["companions"][0]["rhs"];
    assert_eq!(rhs["output_idx"], json!(["_r1", "_r2"]));
    assert!(stable_json(rhs).contains("\"j\""));
    // the full index of an array inside the definition is a scalar
    let full = variant(|m| {
        m["equations"][0]["rhs"] =
            json!({"op": "*", "args": [jr_t(), {"op": "index", "args": ["Eb", 1, 2]}]});
    });
    assert_eq!(
        jacobian_bands(&full, Some("Box"), Wrt::Time).unwrap().len(),
        1
    );
}

/// The names `_dt_<o>` an expression's free variables hold, sorted.
fn companion_reads(e: &Value) -> Vec<String> {
    fn walk(e: &Value, acc: &mut std::collections::BTreeSet<String>) {
        match e {
            Value::String(s) if s.starts_with("_dt_") => {
                acc.insert(s.clone());
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, acc)),
            Value::Object(m) => m.values().for_each(|x| walk(x, acc)),
            _ => {}
        }
    }
    let mut acc = std::collections::BTreeSet::new();
    walk(e, &mut acc);
    acc.into_iter().collect()
}

#[test]
fn companions_keep_the_chain_as_named_steps() {
    // tgrad_named_steps (spec §4.2): w(t) → Tb → kr → c (at its cell) and s
    // (under a sum over k). The bands read `_dt_kr`; each companion's
    // definition reads only the next one down the chain; Eb (data and a
    // parameter weight) gets none.
    let file = load("tgrad_named_steps");
    let entries = jacobian_bands(&file, Some("Column"), Wrt::Time).unwrap();
    let rows: Vec<(&str, &str)> = entries
        .iter()
        .map(|en| (en.u.as_str(), en.v.as_str()))
        .collect();
    assert_eq!(rows, [("c", "t"), ("s", "t")]);
    let bands: Vec<Value> = entries.iter().map(serialize_band).collect();
    assert_eq!(companion_reads(&bands[0]["coef"]), ["_dt_kr"]);
    assert_eq!(bands[1]["coef"]["op"], "faq");
    let text = stable_json(&Value::Array(bands));
    let want =
        fs::read_to_string(root().join("tests/goldens/tgrad_named_steps.time.bands.json")).unwrap();
    assert_eq!(text, want);
    let block = jacobian_block(&file, Some("Column"), Wrt::Time, true, 12).unwrap();
    let comps = block["companions"].as_array().unwrap();
    let names: Vec<&str> = comps.iter().map(|c| c["name"].as_str().unwrap()).collect();
    let of: Vec<&str> = comps.iter().map(|c| c["of"].as_str().unwrap()).collect();
    assert_eq!(names, ["_dt_Tb", "_dt_kr", "_dt_w"]);
    assert_eq!(of, ["Tb", "kr", "w"]);
    assert_eq!(companion_reads(&comps[1]["rhs"]), ["_dt_Tb"]);
    assert_eq!(companion_reads(&comps[0]["rhs"]), ["_dt_w"]);
    assert!(companion_reads(&comps[2]["rhs"]).is_empty());
    // a time pair takes `t` second (spec §4.4): the time bands of each state
    // entry's coefficient, through the same companions, never inlined
    let e2 = jacobian_bands2(&file, Some("Column"), (Wrt::Time, Wrt::States)).unwrap();
    assert!(!e2.is_empty());
    for en in &e2 {
        assert_eq!(en.v, "t");
        assert!(!companion_reads(&serde_json::to_value(&en.band.coef).unwrap()).is_empty());
    }
    let pblk = jacobian_block2(
        &file,
        Some("Column"),
        (Wrt::Time, Wrt::States),
        None,
        true,
        12,
    )
    .unwrap();
    let pnames: Vec<&str> = pblk["companions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(pnames, ["_dt_Tb", "_dt_kr", "_dt_w"]);
}

#[test]
fn companion_name_is_clear_of_the_source() {
    let text = fs::read_to_string(root().join("tests/valid/tgrad_named_steps.esm")).unwrap();
    let mut d: Value = serde_json::from_str(&text).unwrap();
    d["models"]["Column"]["variables"]["_dt_w"] = json!({"type": "parameter", "default": 1.0});
    let file = earthsci_ast::load_string(&d.to_string()).expect("variant loads");
    let block = jacobian_block(&file, Some("Column"), Wrt::Time, true, 12).unwrap();
    let names: Vec<&str> = block["companions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["_dt_Tb", "_dt_kr", "_dt_w_"]);
}

#[test]
fn companions_match_a_central_difference_in_t() {
    // The composed f_t at the value golden's point equals the derivative in
    // t of the source right-hand side, which EarthSciAST evaluates.
    let (env, g) = common::jvals_env("tgrad_named_steps.time");
    let src: Value = serde_json::from_str(
        &fs::read_to_string(root().join("tests/valid/tgrad_named_steps.esm")).unwrap(),
    )
    .unwrap();
    let h = 1e-5;
    let (mut ep, mut em) = (env.clone(), env.clone());
    ep.t += h;
    em.t -= h;
    let up = common::docvals::DocEval::new(&src, "Column", &ep);
    let dn = common::docvals::DocEval::new(&src, "Column", &em);
    let (up, dn) = (up.call(&Default::default()), dn.call(&Default::default()));
    let entries = g["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 12);
    for e in entries {
        let (row, j) = (e[0].as_str().unwrap(), e[2].as_f64().unwrap());
        let fd = (up.at(row) - dn.at(row)) / (2.0 * h);
        assert!(
            (j - fd).abs() <= 1e-6 * (1e-6 + fd.abs()),
            "f_t[{row}] = {j}, central difference {fd}"
        );
    }
}
