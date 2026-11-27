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
