"""The cross-language conformance gate: stable_json of the serialized band
list must be byte-identical to the Julia reference's goldens."""

import json
import struct

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
    "filtered_scan",
    "semiring_reduce",
    "faq_operand_list",
    "pre_operator",
    "recurrence_sweep",
    "pbl_rank1",
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
    ("recurrence_sweep", "parameters"),
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


def test_stable_json_float_goldens(golden_dir):
    # [bits, text] pairs every binding's writer must reproduce, decimal ties
    # (written with the even last digit, as `repr` does) among them
    pairs = json.loads((golden_dir / "canonical_floats.json").read_text())
    for bits, want in pairs:
        x = struct.unpack(">d", bytes.fromhex(bits))[0]
        assert stable_json([x]) == f"[{want}]", bits


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


# Structure-only entries (spec §2.1): `<name>.structure.bands.json`, the state
# entries without coefficients. The reference writes those of the first ten
# fixtures, of the flattened coupled document and of `onehot_clipped` (a
# one-hot column split by clipping, a state of two equations); the Rust
# binding those of the two recurrence fixtures, whose pass runs in full (§2.1).
@pytest.mark.parametrize("name", (*GOLDEN_NAMES, "coupled_chem_advection", "onehot_clipped"))
def test_structure_golden(name, valid_dir, golden_dir):
    from earthsci_astdiff import sysview

    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    if name == "coupled_chem_advection":
        sv = sysview(ea.flatten(file))
    else:
        sv = sysview(file, next(iter(file.models)))
    entries = jacobian_bands(sv, coefficients=False)
    assert all(e.band.coef is None for e in entries)
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / f"{name}.structure.bands.json").read_text()
    # a structure-only entry reads back as it was written
    from earthsci_astdiff import parse_band

    assert stable_json([serialize_band(parse_band(serialize_band(e))) for e in entries]) == text


def test_structure_merges_on_structure_alone(valid_dir):
    """`onehot_clipped` (spec §2.1 item 2): the one-hot column of `c` that
    clipping splits into rows 1..2 and 3..5 is two full bands (their one-hot
    selections differ) and ONE structure-only band; `e`'s two equations keep
    their bands on `c` apart in both modes."""
    from earthsci_astdiff import sysview

    file = ea.load_path(str(valid_dir / "onehot_clipped.esm"))
    sv = sysview(file, next(iter(file.models)))
    full = jacobian_bands(sv)
    so = jacobian_bands(sv, coefficients=False)

    def rows(entries, u, v, onehot):
        return [
            e.band.rows
            for e in entries
            if e.u == u and e.v == v and bool(e.band.contracted) == onehot
        ]

    assert rows(full, "c", "c", True) == [[(1, 2)], [(3, 5)]]
    assert rows(so, "c", "c", True) == [[(1, 5)]]
    assert rows(full, "e", "c", False) == rows(so, "e", "c", False) == [[(1, 2)], [(3, 5)]]


def test_structure_keeps_zero_coefficient():
    """d/dx[i] of x[i] - x[i] simplifies to 0: the full mode drops the band,
    the structure-only mode keeps it (spec §2.1)."""
    from earthsci_ast.esm_types import ExprNode
    from earthsci_astdiff.bands import Ctx, bands

    ctx = Ctx({"i": 4}, {"x": [4]}, set())
    xi = ExprNode(op="index", args=["x", "i"])
    rhs = ExprNode(
        op="faq",
        args=[],
        output_idx=["i"],
        expr=ExprNode(op="-", args=[xi, xi]),
        ranges={"i": [1, 4]},
    )
    assert bands(rhs, "x", ctx, shape_u=[4]) == []
    so = bands(rhs, "x", ctx, shape_u=[4], coefficients=False)
    assert len(so) == 1 and so[0].coef is None and so[0].rows == [(1, 4)]


def test_structure_refuses_what_the_full_mode_refuses():
    """An op with no derivative rule (spec §3) is refused in both modes."""
    from earthsci_ast.esm_types import ExprNode
    from earthsci_astdiff.bands import Ctx, bands
    from earthsci_astdiff.scalar_rules import DerivativeRuleError

    ctx = Ctx({"i": 4}, {"x": [4]}, set())
    xi = ExprNode(op="index", args=["x", "i"])
    # ops with no derivative rule, alone and under a product
    xx = ExprNode(op="grad", args=[xi], dim="x")
    for body in (
        ExprNode(op="D", args=[xi], wrt="lon"),
        xx,
        ExprNode(op="*", args=[2, xx]),
    ):
        rhs = ExprNode(op="faq", args=[], output_idx=["i"], expr=body, ranges={"i": [1, 4]})
        for co in (True, False):
            with pytest.raises(DerivativeRuleError):
                bands(rhs, "x", ctx, shape_u=[4], coefficients=co)


def test_drule_check_agrees_with_dscalar():
    from earthsci_ast.esm_types import ExprNode
    from earthsci_astdiff.expr_helpers import Site, is_zero
    from earthsci_astdiff.scalar_rules import DerivativeRuleError, drule_check, dscalar

    def o(op, *a, **kw):
        return ExprNode(op=op, args=list(a), **kw)

    s = Site.of("x")
    tab = o("const", value=[0.0, 1.0, 4.0])
    for e in (
        "x",
        "y",
        0,
        o("+", "x", "y"),
        o("-", "x"),
        o("-", "y", "x"),
        o("neg", "y"),
        o("*", "x", "y"),
        o("*", 0, "x"),
        o("*", o("sign", "x"), "y"),
        o("/", "y", "x"),
        o("/", o("floor", "x"), "y"),
        o("^", "x", 0),
        o("^", "x", 2),
        o("^", 2, "x"),
        o("^", o("sign", "x"), "y"),
        o("exp", "x"),
        o("log", o("sign", "x")),
        o("sqrt", "x"),
        o("cos", o("ceil", "x")),
        o("atan", "y", "x"),
        o("abs", "x"),
        o("ifelse", o("<", "x", "y"), "y", 1),
        o("max", "x", "y"),
        o("min", o("sign", "x")),
        o("<", "x", "y"),
        o("index", tab, o("floor", "x")),
        o("+", o("index", tab, o("min", o("max", o("floor", "x"), 1), 2)), o("*", 3, "x")),
        o("faq", output_idx=[], reduce="max", ranges={"k": [1, 3]}, expr=o("*", "x", "k")),
    ):
        assert drule_check(e, s) == (not is_zero(dscalar(e, s))), e

    def m2(body):
        return o("faq", output_idx=[], reduce="max", ranges={"j": [1, 3], "k": [1, 3]}, expr=body)

    with pytest.raises(DerivativeRuleError):
        dscalar(m2("x"), s)
    with pytest.raises(DerivativeRuleError):
        drule_check(m2("x"), s)
    assert not drule_check(m2(o("sign", "x")), s)
    assert is_zero(dscalar(m2(o("sign", "x")), s))


@pytest.mark.parametrize("name", ["split_sum_2d", "split_diff_2d"])
def test_split_sum_golden(valid_dir, golden_dir, name):
    """Each summand clipped on its own axes (spec §2.2): the band form the
    reference writes for ``split_sum_2d.esm`` and, through a difference with
    an uncut summand, ``split_diff_2d.esm``."""
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    text = stable_json([serialize_band(e) for e in jacobian_bands(file, "S")])
    assert text == (golden_dir / f"{name}.bands.json").read_text()


def test_split_diff_pieces(valid_dir):
    """``a - b`` splits into ``a`` and ``-b`` (spec §2.2 rule 1), and the
    uncut summand is a group of its own: 3 + 3 + 1 pieces, not 3 × 3."""
    from earthsci_astdiff import clip_regions, clip_summands

    file = ea.load_path(str(valid_dir / "split_diff_2d.esm"))
    body = file.models["S"].equations[0].rhs.expr
    box = [(1, 6), (1, 5)]
    assert len(clip_regions(body, ["i", "j"], box)) == 9
    assert [p[0] for p in clip_summands(body, ["i", "j"], box)] == [
        [(1, 1), (1, 5)],
        [(2, 5), (1, 5)],
        [(6, 6), (1, 5)],
        [(1, 6), (1, 1)],
        [(1, 6), (2, 4)],
        [(1, 6), (5, 5)],
        [(1, 6), (1, 5)],
    ]
