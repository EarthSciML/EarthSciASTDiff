"""The cross-language conformance gate: stable_json of the serialized band
list must be byte-identical to the Julia reference's goldens."""
import earthsci_ast as ea
import pytest

from earthsci_astdiff import jacobian_bands, serialize_band, stable_json

GOLDEN_NAMES = ("bd_chem", "adv_interior", "contracted_ops",
                "flux_form_adv", "minmod_adv")


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_band_golden(name, valid_dir, golden_dir):
    file = ea.load(str(valid_dir / f"{name}.esm"))
    entries = jacobian_bands(file, next(iter(file.models)))
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / f"{name}.bands.json").read_text()


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_band_roundtrip(name, valid_dir):
    from earthsci_astdiff import parse_band
    from earthsci_astdiff.expr_helpers import skey

    file = ea.load(str(valid_dir / f"{name}.esm"))
    entries = jacobian_bands(file, next(iter(file.models)))
    for en in entries:
        back = parse_band(serialize_band(en))
        assert back.u == en.u and back.v == en.v
        assert back.band.rows == en.band.rows
        assert back.band.contracted == en.band.contracted
        assert skey(back.band.coef) == skey(en.band.coef)
