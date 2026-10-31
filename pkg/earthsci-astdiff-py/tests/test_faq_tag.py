"""The ``faq`` node tag — the regression guard for a SILENT ZERO (port of the
reference's test/faq_tag_test.jl).

EarthSciAST renamed the unified query node's wire tag from ``aggregate`` to
``faq`` at esm 1.1.0; the loader rewrites the deprecated alias at the wire
boundary. Matching the old tag throws nothing: ``lhs_state`` stops recognising
the pointwise-lifted ``faq{ expr: D(index(u, …)) }`` LHS, every array state
equation is classified as non-differential, and the band list comes back
EMPTY. So these assertions are about non-emptiness as much as correctness."""

from types import SimpleNamespace

import earthsci_ast as ea
import pytest
from earthsci_ast.esm_types import ExprNode
from earthsci_astdiff import jacobian_bands, serialize_band, stable_json, sysview
from earthsci_astdiff.system import lhs_state, view_states


def test_lifted_lhs_is_an_array_state_equation():
    lifted = ExprNode(
        op="faq",
        args=[],
        output_idx=["i"],
        ranges={"i": [1, 4]},
        expr=ExprNode(op="D", wrt="t", args=[ExprNode(op="index", args=["u", "i"])]),
    )
    assert lhs_state(SimpleNamespace(lhs=lifted, rhs=None)) == "u"
    plain = ExprNode(op="D", wrt="t", args=["u"])
    assert lhs_state(SimpleNamespace(lhs=plain, rhs=None)) == "u"


@pytest.mark.parametrize(
    "name",
    [
        "adv_interior",
        "flux_form_adv",
        "minmod_adv",
        "contracted_ops",
        "adv_react_2d",
        "bd_chem",
        "contracting_observed",
        "forcing_bracket",
        "filtered_scan",
        "semiring_reduce",
    ],
)
def test_array_models_differentiate_to_a_non_empty_band_set(valid_dir, name):
    """The CI gate: a tag mismatch makes every one of these silently empty."""
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    assert view_states(sysview(file))
    entries = jacobian_bands(file)
    assert entries, f"{name}: EMPTY band list (the silent zero)"
    assert any(e.band.rows for e in entries)


def _alias_doc(tag: str, version: str):
    body = {"op": "*", "args": [-0.5, {"op": "index", "args": ["u", "i"]}]}
    return ea.load_document(
        {
            "esm": version,
            "metadata": {"name": "FaqAlias"},
            "index_sets": {"x": {"kind": "interval", "size": 4}},
            "models": {
                "FaqAlias": {
                    "variables": {"u": {"type": "unknown", "shape": ["x"], "default": 1.0}},
                    "equations": [
                        {
                            "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                            "rhs": {
                                "op": tag,
                                "args": [],
                                "output_idx": ["i"],
                                "ranges": {"i": [1, 4]},
                                "expr": body,
                            },
                        }
                    ],
                }
            },
        }
    )


def test_deprecated_alias_differentiates_identically():
    def ser(f):
        return [(e.u, e.v, stable_json(serialize_band(e))) for e in jacobian_bands(f)]

    new = ser(_alias_doc("faq", "1.1.0"))
    with pytest.warns(Warning, match="deprecated_op_alias"):
        old = ser(_alias_doc("aggregate", "1.0.0"))
    assert new
    assert new == old
