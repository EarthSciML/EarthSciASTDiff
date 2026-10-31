//! Differentiate a model, read its sparsity and structure, and emit the Jacobian
//! as a document EarthSciAST evaluates.

use earthsci_ast::load_path;
use earthsci_astdiff::{
    ctx_of, detect_structure, evaluation_document, jacobian_block, jacobian_bands,
    plan_factorization, sysview, FactorizationPlan, Wrt,
};
use earthsci_astdiff_api_examples::fixture;

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
    let Some(FactorizationPlan::BlockDiagonal(plan)) =
        plan_factorization(&entries, &view, &ctx, None)?
    else {
        unreachable!("a block-diagonal Jacobian has a block LU plan")
    };
    let mut vars = plan.block_vars.clone();
    vars.sort();
    assert_eq!(vars, ["A", "B", "C"]);
    // endregion

    // region: evaluate ops=evaluation_document
    // EarthSciASTDiff evaluates nothing: it emits the band fields `J_k` as a
    // derived ESM model that EarthSciAST evaluates.
    let (doc, fields) = evaluation_document(&view, &entries, true, 12)?;
    assert!(doc["models"]["JacobianEval"].is_object());
    assert_eq!(fields.len(), entries.len());
    // endregion

    // region: block ops=jacobian_block
    let block = jacobian_block(&file, None, Wrt::States, true, 12)?;
    assert_eq!(block["structure"], "block_diagonal");
    assert_eq!(block["factorization"]["type"], "block_diagonal_lu");
    // endregion

    Ok(())
}
