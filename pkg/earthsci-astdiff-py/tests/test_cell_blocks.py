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


def test_the_positional_classes_shared_golden(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "named_dims.esm"))
    out = {}
    for m in sorted(file.models):
        sv = sysview(file, m)
        entries = jacobian_bands(sv)
        st = detect_structure(entries, sv)
        plan = plan_factorization(entries, sv, structure=st)
        out[m] = {
            "structure": st,
            "factorization": None if plan is None else serialize_plan(plan),
        }
    assert stable_json(out) == (golden_dir / "named_dims.structure.json").read_text()


def test_the_emitted_block_takes_the_cell_axes(valid_dir, golden_dir):
    from earthsci_astdiff import jacobian_block

    file = ea.load_path(str(valid_dir / "cell_block_mixed_rank.esm"))
    block = jacobian_block(file, cell_axes=["lev", "bin"])
    golden = (golden_dir / "cell_block_mixed_rank.cell-lev-bin.block.json").read_text()
    assert stable_json(block) == golden
    b0 = jacobian_block(file)
    assert b0["structure"] == "general"
    assert "factorization" not in b0


def test_block_diagonal_is_positional_cell_block_pairs_by_name(valid_dir):
    from earthsci_astdiff import LaneBlocks, scatter_plan

    file = ea.load_path(str(valid_dir / "named_dims.esm"))
    sv = sysview(file, "Transposed")
    entries = jacobian_bands(sv)
    # p over [x, y] reading q over [y, x] at q[j, i], x and y of one size:
    # diagonal by name only, and the classes are positional
    assert detect_structure(entries, sv) == "general"
    # the lane-block scatter plan would place dp[i, j]/dq[j, i] in lane
    # (i, j)'s slot for q[i, j]: refused
    with pytest.raises(ValueError, match="block_diagonal"):
        scatter_plan(file, LaneBlocks(), model_name="Transposed")
    # over the cell axes [x, y] the axes pair by name
    assert detect_structure(entries, sv, cell_axes=["x", "y"]) == "cell_block"
    cb = cell_blocks(entries, sv, ["x", "y"])
    assert cb["block_vars"] == ["p", "q"]
    assert [(v, list(i)) for v, i in cb["elements"]] == [("p", []), ("q", [])]
    assert [list(map(bool, r)) for r in cb["pattern"]] == [[True, True], [False, True]]


def test_second_order_structure_is_positional(valid_dir):
    from earthsci_astdiff import jacobian_bands2
    from earthsci_astdiff.jacobian import _FirstOrder
    from earthsci_astdiff.second_order import structure2

    # Mismatch: d2a/da db has a second column on b over [cell, el], the row
    # over [cell, salt]: by position the column is the row's, by index set
    # it is not. Even with the first-order class taken as block diagonal,
    # the pair block is not.
    file = ea.load_path(str(valid_dir / "named_dims.esm"))
    sv = sysview(file, "Mismatch")
    e1 = jacobian_bands(sv)
    e2 = jacobian_bands2(sv)
    assert any(en.w == "b" for en in e2)
    memo = _FirstOrder()
    memo._structure[id(e1)] = (e1, "block_diagonal")
    assert structure2(e1, e2, sv, "states", memo) is None
    # a positional pair block keeps the class
    sv0, e10 = _load(valid_dir, "bd_chem")
    assert structure2(e10, jacobian_bands2(sv0), sv0, "states") == "block_diagonal"


def test_a_lone_cell_axis_name_is_one_axis(valid_dir):
    from earthsci_astdiff import jacobian_block

    sv, entries = _load(valid_dir, "cell_block_mixed_rank")
    # the axis `lev`, not the axes `l`, `e`, `v`
    assert cell_blocks(entries, sv, "lev") == cell_blocks(entries, sv, ["lev"])
    assert detect_structure(entries, sv, cell_axes="lev") == detect_structure(
        entries, sv, cell_axes=["lev"]
    )
    file = ea.load_path(str(valid_dir / "cell_block_mixed_rank.esm"))
    assert stable_json(jacobian_block(file, cell_axes="lev")) == stable_json(
        jacobian_block(file, cell_axes=["lev"])
    )
    # a one-axis cell_block Jacobian, planned from a lone name
    file = ea.load_path(str(valid_dir / "named_dims.esm"))
    tsv = sysview(file, "Transposed")
    te = jacobian_bands(tsv)
    assert detect_structure(te, tsv, cell_axes="x") == "cell_block"
    assert serialize_plan(plan_factorization(te, tsv, cell_axes="x")) == serialize_plan(
        plan_factorization(te, tsv, cell_axes=["x"])
    )


def test_a_shape_over_an_undeclared_index_set(valid_dir):
    import copy

    from earthsci_astdiff import BandError

    sv, entries = _load(valid_dir, "cell_block_mixed_rank")
    w = copy.copy(sv.variables["w"])
    w.shape = [*w.shape, "foo"]
    sv.variables["w"] = w
    with pytest.raises(BandError, match=r"^unknown index set `foo` in shape$"):
        cell_blocks(entries, sv, ["lev"])
