"""Block emission against the Julia reference's block goldens
(tests/goldens/*.block.json): byte-identical, factorization plan included."""

import earthsci_ast as ea
import pytest
from earthsci_astdiff import (
    jacobian_bands,
    jacobian_block,
    parse_jacobian_block,
    parse_jacobian_blocks,
    skey,
    stable_json,
)

NAMES = (
    "bd_chem",
    "adv_interior",
    "contracted_ops",
    "flux_form_adv",
    "minmod_adv",
    "contracting_observed",
    "tgrad_photolysis",
    "forcing_bracket",
    "filtered_scan",
    "semiring_reduce",
)


@pytest.mark.parametrize("name", NAMES)
def test_block_golden(name, valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    block = jacobian_block(file, next(iter(file.models)))
    assert stable_json(block) == (golden_dir / f"{name}.block.json").read_text()


@pytest.mark.parametrize("name", NAMES)
def test_block_roundtrip(name, valid_dir):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    mname = next(iter(file.models))
    block = jacobian_block(file, mname)
    wrt, back = parse_jacobian_block(block)
    ref = jacobian_bands(file, mname)
    assert wrt == "states" and len(back) == len(ref)
    for a, b in zip(ref, back):
        assert skey(a.band.coef) == skey(b.band.coef)
        assert [skey(c) for c in a.band.cidx] == [skey(c) for c in b.band.cidx]


AXIS_GOLDENS = (
    ("tgrad_photolysis", "time"),
    ("forcing_bracket", "forcing"),
    ("forcing_bracket", "parameters"),
)


@pytest.mark.parametrize("name,wrt", AXIS_GOLDENS)
def test_axis_block_golden(name, wrt, valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    mname = next(iter(file.models))
    block = jacobian_block(file, mname, wrt=wrt)
    assert block["wrt"] == wrt
    # a parameters / forcing block carries its column space, never a
    # structure class or factorization plan (spec §4)
    rect = wrt in ("parameters", "forcing")
    assert ("columns" in block) == rect and ("structure" in block) != rect
    assert stable_json(block) == (golden_dir / f"{name}.{wrt}.block.json").read_text()
    w, back = parse_jacobian_block(block)
    assert w == wrt and len(back) == len(jacobian_bands(file, mname, wrt=wrt))


def test_several_axes_in_one_entry(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "forcing_bracket.esm"))
    blocks = jacobian_block(file, "Bracket", wrt=["states", "forcing"])
    assert isinstance(blocks, list) and [b["wrt"] for b in blocks] == ["states", "forcing"]
    assert stable_json(blocks[0]) == (golden_dir / "forcing_bracket.block.json").read_text()
    assert stable_json(blocks[1]) == (golden_dir / "forcing_bracket.forcing.block.json").read_text()
    assert [w for w, _ in parse_jacobian_blocks(blocks)] == ["states", "forcing"]
    assert [w for w, _ in parse_jacobian_blocks(blocks[1])] == ["forcing"]
    with pytest.raises(ValueError):
        jacobian_block(file, "Bracket", wrt=["states", "states"])
    with pytest.raises(ValueError):
        parse_jacobian_blocks([blocks[1], blocks[1]])
