"""Block emission against the Julia reference's block goldens
(tests/goldens/*.block.json): byte-identical after stripping the
`factorization` key (that plan is not ported yet)."""
import json

import earthsci_ast as ea
import pytest

from earthsci_astdiff import (jacobian_bands, jacobian_block,
                              parse_jacobian_block, skey, stable_json)

NAMES = ("bd_chem", "adv_interior", "contracted_ops", "flux_form_adv",
         "minmod_adv")


@pytest.mark.parametrize("name", NAMES)
def test_block_golden(name, valid_dir, golden_dir):
    g = json.loads((golden_dir / f"{name}.block.json").read_text())
    g.pop("factorization", None)
    file = ea.load(str(valid_dir / f"{name}.esm"))
    block = jacobian_block(file, next(iter(file.models)))
    assert stable_json(block) == stable_json(g)


@pytest.mark.parametrize("name", NAMES)
def test_block_roundtrip(name, valid_dir):
    file = ea.load(str(valid_dir / f"{name}.esm"))
    mname = next(iter(file.models))
    block = jacobian_block(file, mname)
    wrt, back = parse_jacobian_block(block)
    ref = jacobian_bands(file, mname)
    assert wrt == "states" and len(back) == len(ref)
    for a, b in zip(ref, back):
        assert skey(a.band.coef) == skey(b.band.coef)
        assert [skey(c) for c in a.band.cidx] == [skey(c) for c in b.band.cidx]
