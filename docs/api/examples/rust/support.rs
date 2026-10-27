//! Shared helper for the examples: locate the `.esm` fixtures by this
//! directory, not the working directory.

use std::path::PathBuf;

/// Path of a fixture document in the repository's `tests/valid/`.
pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../tests/valid")
        .join(name)
}
