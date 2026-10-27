//! Differentiate a model, read its sparsity and structure, and evaluate the Jacobian.

use std::collections::HashMap;

use earthsci_ast::load_path;
use earthsci_astdiff::numeric::Env;
use earthsci_astdiff::{
    assemble_jacobian, block_diagonal_lu, detect_structure, jacobian_block, jacobian_bands, sysview,
    ctx_of, Wrt,
};
use earthsci_astdiff_api_examples::fixture;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // region: bands ops=jacobian_bands,sysview
    let file = load_path(fixture("bd_chem.esm"))?;
    let entries = jacobian_bands(&file, None, Wrt::States)?;
    // One entry per (row variable, column variable) pair that occurs in the model.
    let mut pairs: Vec<(&str, &str)> =
        entries.iter().map(|en| (en.u.as_str(), en.v.as_str())).collect();
    pairs.sort();
    pairs.dedup();
    assert_eq!(
        pairs,
        [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
    );
    let view = sysview(&file, None)?;
    // endregion

    // region: structure ops=detect_structure,plan_factorization
    let ctx = ctx_of(&view)?;
    assert_eq!(detect_structure(&entries, &ctx), "block_diagonal");
    let plan = block_diagonal_lu(&entries);
    let mut vars = plan.block_vars.clone();
    vars.sort();
    assert_eq!(vars, ["A", "B", "C"]);
    // endregion

    // region: assemble ops=assemble_jacobian
    let env = Env {
        states: HashMap::from([
            ("A".to_string(), json!([2.0, 2.0, 2.0, 2.0])),
            ("B".to_string(), json!([0.5, 0.5, 0.5, 0.5])),
            ("C".to_string(), json!([0.1, 0.1, 0.1, 0.1])),
        ]),
        params: HashMap::from([
            ("k1".to_string(), json!(0.04)),
            ("k2".to_string(), json!(3.0)),
            ("k3".to_string(), json!(1.0)),
        ]),
        t: 0.0,
    };
    let values = assemble_jacobian(&file, None, Wrt::States, &env)?;
    assert_eq!(values[&("A[1]".to_string(), "A[1]".to_string())], -0.04); // -k1
    assert_eq!(values[&("B[1]".to_string(), "B[1]".to_string())], -3.0); // -2 k2 B
    // endregion

    // region: block ops=jacobian_block
    let block = jacobian_block(&file, None, Wrt::States, true, 12)?;
    assert_eq!(block["structure"], "block_diagonal");
    assert_eq!(block["factorization"]["type"], "block_diagonal_lu");
    // endregion

    Ok(())
}
