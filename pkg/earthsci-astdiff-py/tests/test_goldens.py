"""The cross-language conformance gate: stable_json of the serialized band
list must be byte-identical to the Julia reference's goldens."""

import earthsci_ast as ea
import pytest
from earthsci_astdiff import jacobian_bands, serialize_band, stable_json

GOLDEN_NAMES = (
    "bd_chem",
    "adv_interior",
    "contracted_ops",
    "flux_form_adv",
    "minmod_adv",
    "contracting_observed",
    "tgrad_photolysis",
    "forcing_bracket",
)


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_band_golden(name, valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    entries = jacobian_bands(file, next(iter(file.models)))
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / f"{name}.bands.json").read_text()


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_band_roundtrip(name, valid_dir):
    from earthsci_astdiff import parse_band
    from earthsci_astdiff.expr_helpers import skey

    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    entries = jacobian_bands(file, next(iter(file.models)))
    for en in entries:
        back = parse_band(serialize_band(en))
        assert back.u == en.u and back.v == en.v
        assert back.band.rows == en.band.rows
        assert back.band.contracted == en.band.contracted
        assert skey(back.band.coef) == skey(en.band.coef)


# Fixtures with `<name>.<axis>.*` goldens beyond the default states axis:
# time (spec §4.2), forcing and parameters (spec §4, §4.3).
AXIS_GOLDENS = (
    ("tgrad_photolysis", "time"),
    ("forcing_bracket", "forcing"),
    ("forcing_bracket", "parameters"),
)
TIME_NAMES = tuple(n for n, w in AXIS_GOLDENS if w == "time")
# Time-dependent fixtures pinned on other axes only (no `.time.*` goldens).
NONAUTONOMOUS = ("forcing_bracket",)


@pytest.mark.parametrize("name", TIME_NAMES)
def test_time_band_golden(name, valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    entries = jacobian_bands(file, next(iter(file.models)), wrt="time")
    assert entries
    # one scalar column `t`, no contracted dimensions
    assert all(e.v == "t" and not e.band.cidx and not e.band.contracted for e in entries)
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / f"{name}.time.bands.json").read_text()


@pytest.mark.parametrize(
    "name", [n for n in GOLDEN_NAMES if n not in TIME_NAMES and n not in NONAUTONOMOUS]
)
def test_time_bands_empty_for_autonomous(name, valid_dir):
    # spec §4.2: an empty time band list means f_t == 0; every fixture
    # without `.time.*` goldens is autonomous
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    assert jacobian_bands(file, next(iter(file.models)), wrt="time") == []


@pytest.mark.parametrize(
    "value,want",
    [
        (1.0e-9, "1.0e-9"),
        (1.5e-5, "1.5e-5"),
        (0.000123456, "0.000123456"),
        (2.5, "2.5"),
        (999999.5, "999999.5"),
        (1000000.5, "1.0000005e6"),
        (123456789.25, "1.2345678925e8"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (1.0, "1.0"),
        (100.0, "100.0"),
        (1.0e6, "1.0e6"),
        (1.0e22, "1.0e22"),
        (0.0001, "0.0001"),
        (1.0e-5, "1.0e-5"),
        (-2.5e-7, "-2.5e-7"),
        (2440587.5, "2.4405875e6"),
        (7.27220521664304e-5, "7.27220521664304e-5"),
        (5e-324, "5.0e-324"),
        (1.7976931348623157e308, "1.7976931348623157e308"),
    ],
)
def test_stable_json_float_layout(value, want):
    # EarthSciAST CONFORMANCE_SPEC §5.5.3.1's probe table plus Julia `show`
    # values: the goldens are written in the reference's float layout
    assert stable_json([value]) == f"[{want}]"


def test_stable_json_non_string_keys():
    # coerced the way json.dumps does, then sorted as the written strings
    assert stable_json({10: 1, 2: 2, "a": 3.0}) == '{"10":1,"2":2,"a":3.0}'


def test_skey_survives_non_finite_constants():
    # equality keys never go through the golden writer, so a folded inf
    # still keys (and keys differently from a finite constant)
    from earthsci_astdiff import skey

    assert skey(float("inf")) != skey(1.0)


@pytest.mark.parametrize("name,wrt", [a for a in AXIS_GOLDENS if a[1] != "time"])
def test_axis_band_golden(name, wrt, valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    entries = jacobian_bands(file, next(iter(file.models)), wrt=wrt)
    assert entries
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / f"{name}.{wrt}.bands.json").read_text()
