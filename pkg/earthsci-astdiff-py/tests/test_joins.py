"""Bands through a relational join (spec §3.5): a regrid whose only gate is a
spatial overlap join. Byte-identical to the Julia reference's goldens."""

import earthsci_ast as ea
from earthsci_astdiff import (
    jacobian_bands,
    jacobian_block,
    jvp_document,
    serialize_band,
    stable_json,
    sysview,
    vjp_document,
)
from earthsci_astdiff.emit import _evaluation_document


def test_join_regrid_goldens(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "join_regrid.esm"))
    sv = sysview(file, "Regrid")
    for wrt, tag in (("states", ""), ("parameters", ".parameters")):
        entries = jacobian_bands(sv, wrt=wrt)
        text = stable_json([serialize_band(e) for e in entries])
        assert text == (golden_dir / f"join_regrid{tag}.bands.json").read_text()
        (block,) = jacobian_block(file, "Regrid", wrt=[wrt])
        assert stable_json(block) == (golden_dir / f"join_regrid{tag}.block.json").read_text()
        ev, _ = _evaluation_document(sv, entries)
        assert stable_json(ev) == (golden_dir / f"join_regrid{tag}.evaldoc.json").read_text()
        jv, _ = jvp_document(file, "Regrid", wrt=wrt)
        assert stable_json(jv) == (golden_dir / f"join_regrid{tag}.jvp.evaldoc.json").read_text()
        vj, _ = vjp_document(file, "Regrid", wrt=wrt)
        assert stable_json(vj) == (golden_dir / f"join_regrid{tag}.vjp.evaldoc.json").read_text()
        col = "q" if wrt == "states" else "F"
        (jb,) = [e for e in entries if e.u == "c" and e.v == col]
        assert [n for n, _ in jb.band.join[1]] == ["i", "r"]


def test_join_two_levels_goldens(valid_dir, golden_dir):
    # a derivative through two joined sums (spec §3.5): the coarse field's
    # bands carry both joins' clauses
    file = ea.load_path(str(valid_dir / "join_two_levels.esm"))
    sv = sysview(file, "Regrid")
    for wrt, tag in (("states", ""), ("parameters", ".parameters")):
        entries = jacobian_bands(sv, wrt=wrt)
        text = stable_json([serialize_band(e) for e in entries])
        assert text == (golden_dir / f"join_two_levels{tag}.bands.json").read_text()
        assert any(
            e.band.join is not None and len(e.band.join[0]) == 2 for e in entries if e.u == "z"
        )
