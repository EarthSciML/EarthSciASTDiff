"""Analytical sparse Jacobians for EarthSciAST (.esm) models.

Python binding of the EarthSciASTDiff band calculus — see
``esm-jacobian-spec.md`` at the repository root for the normative derivative
semantics (§3) and the band-entry format (§4). Gated by the same
deterministic-JSON goldens as the Julia reference.
"""

from .bands import Band, BandError, Ctx, bands, merge_bands, normalize_band
from .clip import clip_regions
from .cse import cse_templates, expand_templates
from .emit import (
    column_names,
    jacobian_block,
    jacobian_pattern,
    parse_band,
    parse_jacobian_block,
    parse_jacobian_blocks,
    scatter_pairs,
    serialize_band,
)
from .expr_helpers import Site, ckey, skey, stable_json
from .factorization import (
    BlockDiagonalPlan,
    ColumnBlockPlan,
    lu_fill,
    markowitz_ordering,
    plan_factorization,
    serialize_plan,
)
from .inline import InlineError, inline_observed
from .jacobian import JacEntry, forcing_names, jacobian_bands, jacobian_columns
from .scalar_rules import DerivativeRuleError, dscalar
from .second_order import JacEntry2, jacobian_bands2, parse_band2
from .simplify_branches import canon_lits, simp, simplify_branches
from .structure import column_lines, detect_structure
from .system import SysView, ctx_of, sysview

__all__ = [
    "Band",
    "BandError",
    "BlockDiagonalPlan",
    "ColumnBlockPlan",
    "Ctx",
    "DerivativeRuleError",
    "InlineError",
    "JacEntry",
    "JacEntry2",
    "Site",
    "SysView",
    "bands",
    "canon_lits",
    "ckey",
    "clip_regions",
    "column_lines",
    "column_names",
    "cse_templates",
    "ctx_of",
    "detect_structure",
    "dscalar",
    "expand_templates",
    "forcing_names",
    "inline_observed",
    "jacobian_bands",
    "jacobian_bands2",
    "jacobian_block",
    "jacobian_columns",
    "jacobian_pattern",
    "lu_fill",
    "markowitz_ordering",
    "merge_bands",
    "normalize_band",
    "parse_band",
    "parse_band2",
    "parse_jacobian_block",
    "parse_jacobian_blocks",
    "plan_factorization",
    "scatter_pairs",
    "serialize_band",
    "serialize_plan",
    "simp",
    "simplify_branches",
    "skey",
    "stable_json",
    "sysview",
]
