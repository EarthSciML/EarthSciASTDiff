"""Analytical sparse Jacobians for EarthSciAST (.esm) models.

Python binding of the EarthSciASTDiff band calculus — see
``esm-jacobian-spec.md`` at the repository root for the normative derivative
semantics (§3) and the band-entry format (§4). Gated by the same
deterministic-JSON goldens as the Julia reference.
"""
from .bands import Band, BandError, Ctx, bands, merge_bands, normalize_band
from .clip import clip_regions
from .cse import cse_templates, expand_templates
from .emit import (jacobian_block, jacobian_pattern, parse_band,
                   parse_jacobian_block, scatter_pairs, serialize_band)
from .structure import detect_structure
from .expr_helpers import Site, ckey, skey, stable_json
from .inline import InlineError, inline_observed
from .jacobian import JacEntry, jacobian_bands
from .numeric import EvalError, assemble_jacobian
from .scalar_rules import DerivativeRuleError, dscalar
from .simplify_branches import canon_lits, simp, simplify_branches
from .system import SysView, ctx_of, sysview

__all__ = [
    "Band", "BandError", "Ctx", "bands", "merge_bands", "normalize_band",
    "clip_regions", "jacobian_pattern", "parse_band", "scatter_pairs",
    "jacobian_block", "parse_jacobian_block", "cse_templates",
    "expand_templates", "detect_structure",
    "serialize_band", "Site", "ckey", "skey", "stable_json",
    "InlineError", "inline_observed", "JacEntry", "jacobian_bands",
    "EvalError", "assemble_jacobian",
    "DerivativeRuleError", "dscalar", "canon_lits", "simp",
    "simplify_branches", "SysView", "ctx_of", "sysview",
]
