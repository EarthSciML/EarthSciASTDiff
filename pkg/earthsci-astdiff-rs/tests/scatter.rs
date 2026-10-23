//! Scatter plans (spec §8): byte-identical plan goldens, values a plan
//! scatters equal to the oracle's, refusals, and the plan's thread-safety.

use std::fs;

use earthsci_astdiff::{
    assemble_jacobian, band_fields, jacobian_bands, stable_json, sysview, Layout,
    ScatterPlan, Wrt,
};

mod common;
use common::{jvals_env, load, root};

fn cb(axis: &str) -> Layout {
    Layout::ColumnBanded { axis: axis.into() }
}

/// Kept in step with test/scatter_test.jl and scripts/regenerate-goldens.jl.
fn plan_layouts() -> Vec<(&'static str, &'static str, Layout)> {
    vec![
        ("bd_chem", "lane_blocks", Layout::LaneBlocks),
        ("bd_chem", "lane_sparse_blocks", Layout::LaneSparseBlocks),
        ("bd_chem", "column_banded", cb("z")),
        ("bd_chem", "csr", Layout::Csr),
        ("adv_interior", "column_banded", cb("x")),
        ("adv_interior", "csr", Layout::Csr),
        ("flux_form_adv", "csr", Layout::Csr),
        ("minmod_adv", "csr", Layout::Csr),
        ("contracted_ops", "csr", Layout::Csr),
        ("contracting_observed", "csr", Layout::Csr),
    ]
}

#[test]
fn plan_goldens() {
    for (name, tag, layout) in plan_layouts() {
        let plan = ScatterPlan::from_file(&load(name), None, &layout).unwrap();
        let golden = fs::read_to_string(
            root().join("tests/goldens").join(format!("{name}.{tag}.plan.json"))).unwrap();
        assert_eq!(stable_json(&plan.to_json()), golden, "plan golden diverges: {name}.{tag}");
    }
}

#[test]
fn scattered_values_match_the_oracle() {
    for (name, tag, layout) in plan_layouts() {
        let file = load(name);
        let (env, _) = jvals_env(name);
        let oracle = assemble_jacobian(&file, None, Wrt::States, &env).unwrap();
        let fields = band_fields(&file, None, &env).unwrap();
        let plan = ScatterPlan::from_file(&file, None, &layout).unwrap();
        let nl = plan.lanes();
        for l in [nl, nl + 3] {
            let mut out = vec![f64::NAN; plan.out_len(l)];
            plan.scatter(&fields, l, &mut out).unwrap();
            let targets = plan.targets(l);
            let mut covered = std::collections::HashSet::new();
            for (pos, t) in targets.iter().enumerate() {
                match t {
                    None => assert_eq!(out[pos], 0.0, "{name}.{tag}: padding slot {pos}"),
                    Some(rc) => {
                        let want = oracle.get(rc).copied().unwrap_or(0.0);
                        let tol = 1e-13 * want.abs().max(1.0);
                        assert!((out[pos] - want).abs() <= tol,
                                "{name}.{tag} L={l}: {rc:?} {} vs {want}", out[pos]);
                        covered.insert(rc.clone());
                    }
                }
            }
            for k in oracle.keys() {
                assert!(covered.contains(k), "{name}.{tag}: {k:?} has no slot");
            }
        }
    }
}

#[test]
fn refusals() {
    let refused = |name: &str, layout: Layout| {
        ScatterPlan::from_file(&load(name), None, &layout).is_err()
    };
    assert!(refused("adv_interior", Layout::LaneBlocks));
    assert!(refused("adv_interior", Layout::LaneSparseBlocks));
    assert!(refused("minmod_adv", cb("x")));            // periodic wrap
    assert!(refused("adv_react_2d", cb("lon")));        // scalar state
    assert!(refused("bd_chem", cb("nope")));
    let file = load("bd_chem");
    let sv = sysview(&file, None).unwrap();
    // an empty band list on a model with array states is never a valid plan
    assert!(ScatterPlan::build(&sv, &[], &Layout::Csr).is_err());
    // columns that are not states
    let pe = jacobian_bands(&file, None, Wrt::Parameters).unwrap();
    assert!(ScatterPlan::build(&sv, &pe, &Layout::Csr).is_err());
    // scatter-time checks: short capacity, wrong output length, wrong fields
    let plan = ScatterPlan::from_file(&file, None, &Layout::LaneBlocks).unwrap();
    let (env, _) = jvals_env("bd_chem");
    let fields = band_fields(&file, None, &env).unwrap();
    assert!(plan.scatter(&fields, 2, &mut vec![0.0; plan.out_len(2)]).is_err());
    assert!(plan.scatter(&fields, 4, &mut vec![0.0; plan.out_len(4) + 1]).is_err());
    assert!(plan.scatter(&fields[1..], 4, &mut vec![0.0; plan.out_len(4)]).is_err());
}

#[test]
fn lane_templates_are_per_cell() {
    let file = load("bd_chem");
    let plan = ScatterPlan::from_file(&file, None, &Layout::LaneBlocks).unwrap();
    match &plan.kind {
        earthsci_astdiff::PlanKind::LaneBlocks { block_vars, items, .. } => {
            assert_eq!(items.len(), 6);                // one per band, any grid
            assert_eq!(block_vars, &["C", "A", "B"]);  // Markowitz order
        }
        _ => unreachable!(),
    }
}

#[test]
fn plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ScatterPlan>();
    assert_send_sync::<Layout>();
}
