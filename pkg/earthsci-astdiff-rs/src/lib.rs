//! Analytical sparse Jacobians for EarthSciAST (`.esm`) models — the Rust
//! binding of EarthSciASTDiff. Normative semantics: `esm-jacobian-spec.md`
//! (§3 derivative rules, §4 band entries). Gated byte-for-byte by the same
//! goldens as the Julia reference and the Python binding.

pub mod bands;
pub mod branches;
pub mod clip;
pub mod emit;
pub mod eval_cse;
pub mod helpers;
pub mod inline;
pub mod jacobian;
pub mod points;
pub mod reductions;
pub mod rules;
pub mod scatter;
pub mod second_order;
pub mod simplify_port;
pub mod system;

pub use bands::{bands, merge_bands, normalize_band, Band, BandError, Ctx};
pub use branches::{canon_lits, simp, simplify_branches};
pub use clip::{clip_regions, RIdx};
pub use emit::{
    block_diagonal_lu, block_diagonal_plan, column_block_plan, column_lines, cse_templates,
    detect_structure, detect_structure_along, evaluation_document, jacobian_block,
    jacobian_block_with, jacobian_block_with_forcing, jacobian_blocks, jacobian_blocks_with,
    lu_fill, markowitz_ordering, parse_band, parse_jacobian_block, parse_jacobian_block2,
    parse_jacobian_blocks, plan_factorization, plan_factorization_along, serialize_band,
    serialize_plan, BlockLu, BlockOptions, BlockWrt, ColumnLines, FactorizationPlan, ParsedBlock,
    STRUCTURE_CLASSES,
};
pub use eval_cse::{hoist_observed, Hoisted};
pub use helpers::{ckey, skey, stable_json, Site};
pub use inline::{inline_observed, InlineError};
pub use jacobian::{
    column_names, forcing_names, jacobian_bands, jacobian_bands_view, jacobian_bands_with_forcing,
    jacobian_columns, JacEntry, Wrt,
};
pub use rules::{dscalar, DerivativeRuleError};
pub use scatter::{BoundScatter, Layout, PlanKind, ScatterPlan};
pub use second_order::{
    axis_pair, jacobian_bands2, jacobian_bands2_with_forcing, jacobian_block2, parse_band2,
    serialize_band2, JacEntry2,
};
pub use system::{ctx_of, sysview, sysview_flat, SysView};
