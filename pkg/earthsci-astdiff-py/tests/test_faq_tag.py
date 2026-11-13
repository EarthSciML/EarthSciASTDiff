"""The ``faq`` node tag — the regression guard for a SILENT ZERO (port of the
reference's test/faq_tag_test.jl).

EarthSciAST renamed the unified query node's wire tag from ``aggregate`` to
``faq`` at esm 1.1.0, and esm 2.0.0 removed the old spelling (the loader
rejects it, ``removed_op``). Matching the old tag threw nothing: ``lhs_state`` stops recognising
the pointwise-lifted ``faq{ expr: D(index(u, …)) }`` LHS, every array state
equation is classified as non-differential, and the band list comes back
EMPTY. So these assertions are about non-emptiness as much as correctness."""

from types import SimpleNamespace

import earthsci_ast as ea
import pytest
from earthsci_ast.esm_types import ExprNode
from earthsci_astdiff import jacobian_bands, sysview
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
        "pre_operator",
        "recurrence_sweep",
        "pbl_rank1",
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


def test_removed_alias_is_refused_at_load():
    assert jacobian_bands(_alias_doc("faq", "1.1.0"))
    # Even a 1.0.0 document, where `aggregate` was the current spelling: the
    # 2.0.0 loader rejects every removed op whatever `esm` it declares.
    with pytest.raises(Exception, match="removed_op"):
        _alias_doc("aggregate", "1.0.0")
