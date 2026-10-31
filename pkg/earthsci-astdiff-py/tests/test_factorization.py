"""plan_factorization's refusals (factorization.jl's testset): a misspelled
structure class and a rectangular (parameters) band list are errors."""

import earthsci_ast as ea
import pytest
from earthsci_astdiff import BlockDiagonalPlan, jacobian_bands, plan_factorization, sysview


@pytest.fixture
def bd_chem(valid_dir):
    file = ea.load_path(str(valid_dir / "bd_chem.esm"))
    return file, sysview(file, "Chem")


def test_block_diagonal_plan(bd_chem):
    file, sv = bd_chem
    plan = plan_factorization(jacobian_bands(file, "Chem"), sv)
    assert isinstance(plan, BlockDiagonalPlan)
    assert sorted(plan.block_vars) == ["A", "B", "C"]


@pytest.mark.parametrize("typo", ["block-diagonal", "blockdiagonal", "Block_Diagonal", ""])
def test_unknown_structure_class_is_refused(bd_chem, typo):
    file, sv = bd_chem
    with pytest.raises(ValueError, match="unknown structure class"):
        plan_factorization(jacobian_bands(file, "Chem"), sv, structure=typo)


@pytest.mark.parametrize("structure", [None, "block_diagonal"])
def test_rectangular_band_lists_are_refused(bd_chem, structure):
    file, sv = bd_chem
    entries = jacobian_bands(file, "Chem", wrt="parameters")
    assert entries
    with pytest.raises(ValueError, match="rectangular"):
        plan_factorization(entries, sv, structure=structure)
