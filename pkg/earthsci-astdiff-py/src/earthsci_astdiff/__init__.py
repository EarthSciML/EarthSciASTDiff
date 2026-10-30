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
    jacobian_block,
    jacobian_pattern,
    parse_band,
    parse_jacobian_block,
    scatter_pairs,
    serialize_band,
)
from .expr_helpers import Site, ckey, skey, stable_json
from .factorization import (
    BlockDiagonalPlan,
    lu_fill,
    markowitz_ordering,
    plan_factorization,
    serialize_plan,
)
from .inline import InlineError, inline_observed
from .jacobian import JacEntry, jacobian_bands
from .numeric import EvalError, assemble_jacobian
from .scalar_rules import DerivativeRuleError, dscalar
from .simplify_branches import canon_lits, simp, simplify_branches
from .structure import detect_structure
from .system import SysView, ctx_of, sysview

__all__ = [
    "Band",
    "BandError",
    "BlockDiagonalPlan",
    "Ctx",
    "DerivativeRuleError",
    "EvalError",
    "InlineError",
    "JacEntry",
    "Site",
    "SysView",
    "assemble_jacobian",
    "bands",
    "canon_lits",
    "ckey",
    "clip_regions",
    "cse_templates",
    "ctx_of",
    "detect_structure",
    "dscalar",
    "expand_templates",
    "inline_observed",
    "jacobian_bands",
    "jacobian_block",
    "jacobian_pattern",
    "lu_fill",
    "markowitz_ordering",
    "merge_bands",
    "normalize_band",
    "parse_band",
    "parse_jacobian_block",
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
