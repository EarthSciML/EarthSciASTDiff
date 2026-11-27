"""Per-cell blocks over caller-named cell axes (structure.jl ``cell_blocks``,
``"cell_block"``, ``CellBlockPlan``; spec §4, §6) and the rank rule (a band
whose column and row have different ranks is never diagonal): the shared
golden, written by the Julia reference, is byte-compared."""

import earthsci_ast as ea
import pytest
from earthsci_astdiff import (
    CellBlockPlan,
    cell_blocks,
    detect_structure,
    jacobian_bands,
    plan_factorization,
    serialize_plan,
    stable_json,
    sysview,
)


def _load(valid_dir, name):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    sv = sysview(file, next(iter(file.models)))
    return sv, jacobian_bands(sv)


def test_the_shared_golden(valid_dir, golden_dir):
    sv, entries = _load(valid_dir, "cell_block_mixed_rank")
    cases = []
    for axes in (None, ["lev", "bin"], ["lev"], ["salt"]):
        st = detect_structure(entries, sv, cell_axes=axes)
        plan = None
        if st in ("block_diagonal", "cell_block"):
            plan = serialize_plan(plan_factorization(entries, sv, structure=st, cell_axes=axes))
        cases.append({"cell_axes": axes, "structure": st, "factorization": plan})
    golden = (golden_dir / "cell_block_mixed_rank.cell_blocks.json").read_text()
    assert stable_json(cases) == golden


def test_a_column_of_another_rank_is_not_diagonal(valid_dir):
    sv, entries = _load(valid_dir, "cell_block_mixed_rank")
    assert any(len(en.band.cidx) != len(en.band.ridx) for en in entries)
    assert detect_structure(entries, sv) == "general"
    assert plan_factorization(entries, sv) is None


def test_what_the_plan_is(valid_dir):
    sv, entries = _load(valid_dir, "cell_block_mixed_rank")
    cb = cell_blocks(entries, sv, ["bin", "lev"])
    assert cb["block_vars"] == ["s", "w"]
    assert cb["elements"] == [("s", [1]), ("s", [2]), ("s", [3]), ("w", [])]
    p = plan_factorization(entries, sv, cell_axes=["lev", "bin"])
    assert isinstance(p, CellBlockPlan)
    for bad in (["salt"], ["lev", "lev"], []):
        assert cell_blocks(entries, sv, bad) is None
    with pytest.raises(ValueError, match="needs its `cell_axes`"):
        plan_factorization(entries, sv, structure="cell_block")
    with pytest.raises(ValueError, match="not cell_block"):
        plan_factorization(entries, sv, structure="cell_block", cell_axes=["salt"])


def test_other_classes_are_kept(valid_dir):
    sv, entries = _load(valid_dir, "adv_interior")
    assert detect_structure(entries, sv) == "banded"
    assert detect_structure(entries, sv, cell_axes=["x"]) == "banded"
    sv, entries = _load(valid_dir, "bd_chem")
    assert detect_structure(entries, sv) == "block_diagonal"
