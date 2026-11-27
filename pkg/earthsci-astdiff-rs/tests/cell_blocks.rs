//! Per-cell blocks over caller-named cell axes (`cell_blocks`,
//! `"cell_block"`, `FactorizationPlan::CellBlock`; spec §4, §6) and the rank
//! rule (a band whose column and row have different ranks is never
//! diagonal): the shared golden, written by the Julia reference, is
//! byte-compared.

use std::fs;

use earthsci_astdiff::{
    cell_blocks, ctx_of, detect_structure, detect_structure_with, jacobian_bands,
    plan_factorization_with, serialize_plan, stable_json, sysview, FactorizationPlan, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

#[test]
fn the_shared_golden() {
    let file = load("cell_block_mixed_rank");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    let choices: [Option<Vec<String>>; 4] = [
        None,
        Some(vec!["lev".into(), "bin".into()]),
        Some(vec!["lev".into()]),
        Some(vec!["salt".into()]),
    ];
    let mut cases = Vec::new();
    for axes in &choices {
        let axes_ref = axes.as_deref();
        let st = detect_structure_with(&entries, &sv, &ctx, None, axes_ref).unwrap();
        let plan = if st == "block_diagonal" || st == "cell_block" {
            let p = plan_factorization_with(&entries, &sv, &ctx, Some(st), None, axes_ref)
                .unwrap()
                .unwrap();
            serialize_plan(&p)
        } else {
            Value::Null
        };
        cases.push(json!({"cell_axes": axes, "structure": st, "factorization": plan}));
    }
    let golden =
        fs::read_to_string(root().join("tests/goldens/cell_block_mixed_rank.cell_blocks.json"))
            .unwrap();
    assert_eq!(stable_json(&Value::Array(cases)), golden);
}

#[test]
fn a_column_of_another_rank_is_not_diagonal() {
    let file = load("cell_block_mixed_rank");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    assert!(entries
        .iter()
        .any(|en| en.band.cidx.len() != en.band.ridx.len()));
    assert_eq!(detect_structure(&entries, &ctx), "general");
}

#[test]
fn what_the_plan_is() {
    let file = load("cell_block_mixed_rank");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    let axes = vec!["bin".to_string(), "lev".to_string()];
    let cb = cell_blocks(&entries, &sv, &ctx, &axes).unwrap().unwrap();
    assert_eq!(cb.block_vars, vec!["s", "w"]);
    assert_eq!(
        cb.elements,
        vec![
            ("s".to_string(), vec![1]),
            ("s".to_string(), vec![2]),
            ("s".to_string(), vec![3]),
            ("w".to_string(), vec![]),
        ]
    );
    let plan = plan_factorization_with(&entries, &sv, &ctx, None, None, Some(&axes)).unwrap();
    assert!(matches!(plan, Some(FactorizationPlan::CellBlock(_))));
    for bad in [
        vec!["salt".to_string()],
        vec!["lev".into(), "lev".into()],
        vec![],
    ] {
        assert!(
            cell_blocks(&entries, &sv, &ctx, &bad).unwrap().is_none(),
            "{bad:?}"
        );
    }
    assert!(plan_factorization_with(&entries, &sv, &ctx, Some("cell_block"), None, None).is_err());
    let salt = vec!["salt".to_string()];
    assert!(
        plan_factorization_with(&entries, &sv, &ctx, Some("cell_block"), None, Some(&salt))
            .is_err()
    );
}

#[test]
fn other_classes_are_kept() {
    let file = load("adv_interior");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    let x = vec!["x".to_string()];
    assert_eq!(detect_structure(&entries, &ctx), "banded");
    assert_eq!(
        detect_structure_with(&entries, &sv, &ctx, None, Some(&x)).unwrap(),
        "banded"
    );
    let file = load("bd_chem");
    let ctx = ctx_of(&sysview(&file, None).unwrap()).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    assert_eq!(detect_structure(&entries, &ctx), "block_diagonal");
}

#[test]
fn the_positional_classes_shared_golden() {
    let file = load("named_dims");
    let mut out = serde_json::Map::new();
    let mut names: Vec<&String> = file.models.as_ref().unwrap().keys().collect();
    names.sort();
    for m in names {
        let sv = sysview(&file, Some(m)).unwrap();
        let ctx = ctx_of(&sv).unwrap();
        let entries = jacobian_bands(&file, Some(m), Wrt::States).unwrap();
        let st = detect_structure(&entries, &ctx);
        let plan = earthsci_astdiff::plan_factorization(&entries, &sv, &ctx, Some(st)).unwrap();
        out.insert(
            m.clone(),
            json!({"structure": st, "factorization": plan.map_or(Value::Null, |p| serialize_plan(&p))}),
        );
    }
    let golden =
        fs::read_to_string(root().join("tests/goldens/named_dims.structure.json")).unwrap();
    assert_eq!(stable_json(&Value::Object(out)), golden);
}

#[test]
fn the_emitted_block_takes_the_cell_axes() {
    let file = load("cell_block_mixed_rank");
    let axes = vec!["lev".to_string(), "bin".to_string()];
    let opts = earthsci_astdiff::BlockOptions {
        cell_axes: Some(&axes),
        ..Default::default()
    };
    let block = earthsci_astdiff::jacobian_block_with(&file, None, Wrt::States, &opts).unwrap();
    let golden = fs::read_to_string(
        root().join("tests/goldens/cell_block_mixed_rank.cell-lev-bin.block.json"),
    )
    .unwrap();
    assert_eq!(stable_json(&block), golden);
    let b0 = earthsci_astdiff::jacobian_block_with(
        &file,
        None,
        Wrt::States,
        &earthsci_astdiff::BlockOptions::default(),
    )
    .unwrap();
    assert_eq!(b0["structure"], json!("general"));
    assert!(b0.get("factorization").is_none());
}

/// p over [x, y] reading q over [y, x] at q[j, i], x and y of one size: diagonal
/// by name only, and the classes are positional, so the lane-block scatter
/// plan (which would place dp[i, j]/dq[j, i] in lane (i, j)'s slot for
/// q[i, j]) is refused; over the cell axes [x, y] the axes pair by name.
#[test]
fn block_diagonal_is_positional_cell_block_pairs_by_name() {
    let file = load("named_dims");
    let sv = sysview(&file, Some("Transposed")).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, Some("Transposed"), Wrt::States).unwrap();
    assert_eq!(detect_structure(&entries, &ctx), "general");
    let Err(err) = earthsci_astdiff::ScatterPlan::from_file(
        &file,
        Some("Transposed"),
        &earthsci_astdiff::Layout::LaneBlocks,
    ) else {
        panic!("the lane-block plan is refused");
    };
    assert!(format!("{err}").contains("block_diagonal"), "{err}");
    let axes = vec!["x".to_string(), "y".to_string()];
    assert_eq!(
        detect_structure_with(&entries, &sv, &ctx, None, Some(&axes)).unwrap(),
        "cell_block"
    );
    let cb = cell_blocks(&entries, &sv, &ctx, &axes).unwrap().unwrap();
    assert_eq!(cb.block_vars, vec!["p".to_string(), "q".to_string()]);
    assert_eq!(
        cb.elements,
        vec![("p".to_string(), vec![]), ("q".to_string(), vec![])]
    );
}

/// A variable shaped over an undeclared index set: the context refuses it,
/// and `cell_blocks` given a context that lacks it does too.
#[test]
fn a_shape_over_an_undeclared_index_set() {
    let mut file = load("cell_block_mixed_rank");
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    let ctx = ctx_of(&sysview(&file, None).unwrap()).unwrap();
    for m in file.models.as_mut().unwrap().values_mut() {
        let w = m.variables.get_mut("w").unwrap();
        w.shape.as_mut().unwrap().push("foo".into());
    }
    let sv = sysview(&file, None).unwrap();
    assert_eq!(
        ctx_of(&sv).err().unwrap().0,
        "unknown index set `foo` in shape"
    );
    let err = cell_blocks(&entries, &sv, &ctx, &["lev".to_string()])
        .err()
        .unwrap();
    assert_eq!(err.0, "unknown index set `foo`");
}
