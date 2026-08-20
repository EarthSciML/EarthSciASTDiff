//! Analytical sparse Jacobians for EarthSciAST (`.esm`) models — the Rust
//! binding of EarthSciASTDiff. Normative semantics: `esm-jacobian-spec.md`
//! (§3 derivative rules, §4 band entries). Gated byte-for-byte by the same
//! goldens as the Julia reference and the Python binding.

pub mod bands;
pub mod branches;
pub mod clip;
pub mod emit;
pub mod helpers;
pub mod inline;
pub mod jacobian;
pub mod numeric;
pub mod rules;
pub mod simplify_port;
pub mod system;

pub use bands::{bands, merge_bands, normalize_band, Band, BandError, Ctx};
pub use branches::{canon_lits, simp, simplify_branches};
pub use clip::{clip_regions, RIdx};
pub use emit::{cse_templates, detect_structure, jacobian_block, lu_fill,
               markowitz_ordering, serialize_band};
pub use helpers::{ckey, skey, stable_json, Site};
pub use inline::{inline_observed, InlineError};
pub use jacobian::{jacobian_bands, JacEntry, Wrt};
pub use numeric::assemble_jacobian;
pub use rules::{dscalar, DerivativeRuleError};
pub use system::{ctx_of, sysview, SysView};
