//! `plan_factorization` and `serialize_plan` (the reference's
//! factorization.jl, the Python binding's factorization.py): the serialized
//! plan of every shared fixture equals the `factorization` field of its
//! Julia-written state block golden byte for byte, and a fixture whose
//! golden has no plan gets none.

use std::fs;

use earthsci_astdiff::{
    block_diagonal_lu, ctx_of, jacobian_bands, plan_factorization, serialize_plan, stable_json,
    sysview, FactorizationPlan, Wrt,
};
use serde_json::Value;

mod common;
use common::{load, root};

const NAMES: [&str; 10] = [
    "bd_chem",
    "adv_interior",
    "contracted_ops",
    "flux_form_adv",
    "minmod_adv",
    "contracting_observed",
    "tgrad_photolysis",
    "forcing_bracket",
    "recurrence_sweep",
    "pbl_rank1",
];

fn golden_block(name: &str) -> Value {
    let path = root()
        .join("tests/goldens")
        .join(format!("{name}.block.json"));
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn plans_match_the_block_goldens() {
    let mut planned = 0;
    for name in NAMES {
        let file = load(name);
        let sv = sysview(&file, None).unwrap();
        let ctx = ctx_of(&sv).unwrap();
        let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
        let golden = golden_block(name);
        assert_eq!(golden["wrt"], "states", "{name}: a state block");
        let plan = plan_factorization(&entries, &sv, &ctx, None).unwrap();
        match (plan, golden.get("factorization")) {
            (Some(p), Some(g)) => {
                assert_eq!(
                    stable_json(&serialize_plan(&p)),
                    stable_json(g),
                    "{name}: plan diverges from the golden"
                );
                planned += 1;
            }
            (None, None) => {}
            (p, g) => panic!("{name}: plan {p:?}, golden factorization {g:?}"),
        }
    }
    assert_eq!(planned, 2, "bd_chem and tgrad_photolysis carry plans");
}

#[test]
fn what_the_plan_is() {
    let file = load("bd_chem");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    let Some(FactorizationPlan::BlockDiagonal(plan)) =
        plan_factorization(&entries, &sv, &ctx, None).unwrap()
    else {
        panic!("a block LU")
    };
    assert_eq!(plan, block_diagonal_lu(&entries));
    let mut vars = plan.block_vars.clone();
    vars.sort();
    assert_eq!(vars, ["A", "B", "C"]);
    assert_eq!(plan.pattern.len(), 3);
    for (p, l) in plan.pattern.iter().flatten().zip(plan.lu.iter().flatten()) {
        assert!(!p || *l, "the LU pattern contains the pattern");
    }
    let ser = serialize_plan(&FactorizationPlan::BlockDiagonal(plan.clone()));
    assert_eq!(ser["type"], "block_diagonal_lu");
    let nnz = plan.lu.iter().flatten().filter(|x| **x).count();
    assert_eq!(ser["lu_pattern"].as_array().unwrap().len(), nnz);
    // 1-based pairs and ordering.
    assert!(ser["pattern"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|p| p.as_array().unwrap())
        .all(|i| (1..=3).contains(&i.as_u64().unwrap())));
    let mut ord = plan.ordering.clone();
    ord.sort();
    assert_eq!(ord, [1, 2, 3]);
}

#[test]
fn structure_argument_decides() {
    let file = load("bd_chem");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    assert_eq!(
        plan_factorization(&entries, &sv, &ctx, Some("banded")).unwrap(),
        None
    );
    assert_eq!(
        plan_factorization(&entries, &sv, &ctx, Some("block_diagonal")).unwrap(),
        Some(FactorizationPlan::BlockDiagonal(block_diagonal_lu(
            &entries
        )))
    );
    // A class without a plan yet gives none.
    let adv = load("adv_interior");
    let asv = sysview(&adv, None).unwrap();
    let actx = ctx_of(&asv).unwrap();
    let aent = jacobian_bands(&adv, None, Wrt::States).unwrap();
    assert_eq!(plan_factorization(&aent, &asv, &actx, None).unwrap(), None);
}

/// A misspelled class is an error, never a silent "no plan".
#[test]
fn unknown_structure_class_is_refused() {
    let file = load("bd_chem");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    for typo in ["block-diagonal", "blockdiagonal", "Block_Diagonal", ""] {
        let err = plan_factorization(&entries, &sv, &ctx, Some(typo)).unwrap_err();
        assert!(
            err.0.contains("unknown structure class"),
            "{typo}: {}",
            err.0
        );
    }
}

/// A parameters band list is a rectangular matrix: no block LU over a mix
/// of states and parameters, whatever class the caller names or detects.
#[test]
fn rectangular_band_lists_are_refused() {
    let file = load("bd_chem");
    let sv = sysview(&file, None).unwrap();
    let ctx = ctx_of(&sv).unwrap();
    let entries = jacobian_bands(&file, None, Wrt::Parameters).unwrap();
    assert!(!entries.is_empty());
    for structure in [None, Some("block_diagonal")] {
        let err = plan_factorization(&entries, &sv, &ctx, structure).unwrap_err();
        assert!(err.0.contains("rectangular"), "{}", err.0);
    }
}
