"""Filtered faq nodes, non-``+`` reductions and value-indexed reads
(esm-jacobian-spec.md §3.5; port of the reference's
test/reductions_test.jl): every case is checked against central differences
of EarthSciAST's own evaluator (tests/oracle.py), and the guarded cases for
an EXACT pattern — the window's points, no padding."""

import earthsci_ast as ea
import pytest
from earthsci_astdiff import (
    BandError,
    assemble_jacobian,
    column_lines,
    detect_structure,
    jacobian_bands,
    jacobian_block,
    sysview,
)
from earthsci_astdiff.emit import check_state_column
from oracle import check_against_fd


def idx(v, *i):
    return {"op": "index", "args": [v, *i]}


def op(o, *a):
    return {"op": o, "args": list(a)}


def faq(oidx, ranges, body, **kw):
    return {
        "op": "faq",
        "args": [],
        "output_idx": list(oidx),
        "ranges": dict(ranges),
        "expr": body,
        **kw,
    }


def doc(rhs, n=5, nz=0, ushape=("x",), variables=None, eqs=()):
    """One model over `x` (size n) and, optionally, `z`; `u` is the state."""
    sets = {"x": {"kind": "interval", "size": n}}
    if nz > 0:
        sets["z"] = {"kind": "interval", "size": nz}
    return ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "U9"},
            "index_sets": sets,
            "models": {
                "U9": {
                    "variables": {
                        "u": {"type": "unknown", "shape": list(ushape), "default": 0.5},
                        "w": {"type": "parameter", "default": 0.7},
                        **(variables or {}),
                    },
                    "equations": [
                        *eqs,
                        {"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": rhs},
                    ],
                }
            },
        }
    )


def oi(body, n=5, **kw):
    return faq(["i"], {"i": [1, n], "j": [1, n]}, body, **kw)


def le(a, b):
    return op("<=", a, b)


def lt(a, b):
    return op("<", a, b)


def ge(a, b):
    return op(">=", a, b)


def gt(a, b):
    return op(">", a, b)


def neg(x):
    return op("-", x)


U5 = [0.3, 0.9, 0.2, 0.6, 0.45]
U34 = [0.1 + k * (1.1 / 11) for k in range(12)]


def scan(f):
    return neg(oi(op("*", "w", idx("u", "j")), filter=f))


# ── filtered `+` at array level ──
FILTERED = {
    "inclusive forward scan": (scan(le("j", "i")), 15),
    "exclusive forward scan": (scan(lt("j", "i")), 10),
    "inclusive reverse scan": (scan(ge("j", "i")), 15),
    "mirrored spelling": (scan(gt("i", "j")), 10),
    "constant window (box only)": (scan(le("j", 2)), 10),
    "window clipped by a constant": (scan(op("and", le("j", "i"), ge("j", 2))), 10),
    "value-dependent filter conjunct": (
        neg(oi(op("^", idx("u", "j"), 2), filter=op("and", le("j", "i"), gt(idx("u", "j"), 0.4)))),
        15,
    ),
    "column with and without the contracted index": (
        neg(oi(op("*", idx("u", "i"), idx("u", "j")), filter=lt("j", "i"))),
        None,
    ),
    # max / min
    "running max": (
        neg(oi(op("*", idx("u", "j"), idx("u", "j")), filter=le("j", "i"), reduce="max")),
        15,
    ),
    "suffix min": (
        neg(oi(op("*", idx("u", "j"), idx("u", "j")), filter=ge("j", "i"), reduce="min")),
        15,
    ),
    "max_product semiring": (neg(oi(idx("u", "j"), semiring="max_product")), 25),
    # nested: D(u)[i] = -u[i]·max_j u[j]
    "nested max": (
        faq(
            ["i"],
            {"i": [1, 5]},
            neg(op("*", idx("u", "i"), faq([], {"j": [1, 5]}, idx("u", "j"), reduce="max"))),
        ),
        None,
    ),
    # products
    "prefix product": (neg(oi(op("+", 1, idx("u", "j")), filter=le("j", "i"), reduce="*")), 15),
}


@pytest.mark.parametrize("label", list(FILTERED))
def test_against_finite_differences(label):
    rhs, npat = FILTERED[label]
    check_against_fd(doc(rhs), U5, npat=npat)


def test_prefix_product_with_a_zero_factor():
    rhs, npat = FILTERED["prefix product"]
    check_against_fd(doc(rhs), [0.3, -1.0, 0.2, 0.6, 0.45], npat=npat)


def test_mz_exclusive_level_scan():
    # the Mz shape: (gi, gke) rows over a level contraction gk < gke
    mz = faq(
        ["gi", "gke"],
        {"gi": [1, 3], "gke": [1, 4], "gk": [1, 4]},
        op("*", "w", idx("u", "gi", "gk")),
        filter=lt("gk", "gke"),
    )
    file = doc(neg(mz), n=3, nz=4, ushape=("x", "z"))
    check_against_fd(file, U34, npat=3 * 6)
    assert any(en.band.guard is not None for en in jacobian_bands(file))


def test_row_only_filter_clips_without_a_guard():
    file = doc(faq(["i"], {"i": [1, 5]}, op("*", "w", idx("u", "i")), filter=ge("i", 3)))
    check_against_fd(file, U5, npat=3)
    assert all(en.band.guard is None for en in jacobian_bands(file))


def test_nested_filtered_sum():
    # a filtered sum read through an observed (nested scalar sum)
    svar = {"s": {"type": "unknown", "shape": ["x"]}}
    seq = [{"lhs": "s", "rhs": oi(op("^", idx("u", "j"), 2), filter=le("j", "i"))}]
    file = doc(
        faq(["i"], {"i": [1, 5]}, neg(op("*", idx("s", "i"), idx("u", "i")))),
        variables=svar,
        eqs=seq,
    )
    check_against_fd(file, U5, npat=15)
    assert any(en.band.guard is not None for en in jacobian_bands(file))


def test_exact_ties_take_the_first_extremum():
    rhs, _ = FILTERED["running max"]
    utie = [0.3, 0.9, 0.2, 0.9, 0.45]
    J = assemble_jacobian(doc(rhs), states={"u": utie})
    assert J[("u[4]", "u[2]")] == pytest.approx(-2 * 0.9)
    assert J[("u[4]", "u[4]")] == 0
    assert J[("u[5]", "u[2]")] == pytest.approx(-2 * 0.9)
    assert J[("u[5]", "u[4]")] == 0


def test_max_over_two_contracted_indices_is_refused():
    mx2 = faq(["i"], {"i": [1, 3], "j": [1, 3], "k": [1, 4]}, idx("u", "j", "k"), reduce="max")
    with pytest.raises(BandError):
        jacobian_bands(doc(mx2, n=3, nz=4, ushape=("x", "z")), "U9")


def test_level_search_as_an_index():
    # κ = min{k : u[k] > 0.5}; D(u)[i] = -u[κ]·u[i]
    kvar = {"kap": {"type": "unknown"}}
    keq = [
        {
            "lhs": "kap",
            "rhs": faq([], {"k": [1, 5]}, "k", reduce="min", filter=gt(idx("u", "k"), 0.5)),
        }
    ]
    file = doc(
        faq(["i"], {"i": [1, 5]}, neg(op("*", idx("u", "kap"), idx("u", "i")))),
        variables=kvar,
        eqs=keq,
    )
    check_against_fd(file, U5, npat=25)
    entries = jacobian_bands(file)
    assert any(any(c[0].startswith("_oh") for c in en.band.contracted) for en in entries)


def test_array_level_unary_minus_keeps_its_sign():
    file = doc(neg(faq(["i"], {"i": [1, 5]}, op("*", "w", idx("u", "i")))))
    J = check_against_fd(file, U5, npat=5)
    assert J[("u[2]", "u[2]")] == pytest.approx(-0.7)


# ── column_block structure ──

R = {"i": [1, 3], "k": [1, 4]}
LOC = faq(["i", "k"], R, op("-", idx("u", "i", "k")))


def test_column_block_structure():
    # an exclusive level scan along z plus a local term: dense lower lines
    scan_z = faq(
        ["i", "k"],
        {**R, "m": [1, 4]},
        op("*", "w", idx("u", "i", "m")),
        filter=op("<", "m", "k"),
    )
    file = doc(op("+", scan_z, LOC), n=3, nz=4, ushape=("x", "z"))
    sv = sysview(file, "U9")
    entries = jacobian_bands(sv)
    assert detect_structure(entries, sv) == "general"
    assert detect_structure(entries, sv, column_axis="z") == "column_block"
    assert detect_structure(entries, sv, column_axis="x") == "general"
    cl = column_lines(entries, sv, "z")
    assert (cl["pattern"], cl["kl"], cl["ku"], cl["line_length"], cl["block_vars"]) == (
        "lower",
        3,
        0,
        4,
        ["u"],
    )
    block = jacobian_block(file, "U9", column_axis="z")
    assert block["structure"] == "column_block"
    assert block["factorization"] == {
        "type": "column_block",
        "axis": "z",
        "line_length": 4,
        "block_vars": ["u"],
        "pattern": "lower",
        "bandwidth": [3, 0],
    }
    # without the axis the block is unchanged
    assert jacobian_block(file, "U9")["structure"] == "general"
    check_against_fd(file, U34)


def test_column_block_banded_and_block_diagonal():
    # a one-step stencil along z: banded lines
    sten = faq(["i", "k"], {"i": [1, 3], "k": [2, 4]}, op("*", "w", idx("u", "i", op("-", "k", 1))))
    file2 = doc(op("+", sten, LOC), n=3, nz=4, ushape=("x", "z"))
    sv2 = sysview(file2, "U9")
    cl2 = column_lines(jacobian_bands(sv2), sv2, "z")
    assert (cl2["pattern"], cl2["kl"], cl2["ku"]) == ("banded", 1, 0)
    assert detect_structure(jacobian_bands(sv2), sv2, column_axis="z") == "column_block"
    # a chemistry-shaped (per-cell) Jacobian stays block-diagonal
    file3 = doc(LOC, n=3, nz=4, ushape=("x", "z"))
    sv3 = sysview(file3, "U9")
    assert detect_structure(jacobian_bands(sv3), sv3, column_axis="z") == "block_diagonal"


# ── silent paths ──


def test_faq_lhs_observed_goes_through_to_the_state():
    # `s[i] ~ Σ_j u[j]·u[i]` written with a faq LHS is an observed, inlined,
    # so the bands go through it to `u` — never a column of `s`.
    svar = {"s": {"type": "unknown", "shape": ["x"]}}
    seq = [
        {
            "lhs": faq(["i"], {"i": [1, 5]}, idx("s", "i")),
            "rhs": faq([], {"j": [1, 5]}, op("*", idx("u", "j"), idx("u", "i"))),
        }
    ]
    file = doc(faq(["i"], {"i": [1, 5]}, op("-", idx("s", "i"))), variables=svar, eqs=seq)
    entries = jacobian_bands(file, "U9")
    assert entries and all(en.v == "u" for en in entries)
    check_against_fd(file, U5)


def test_unresolvable_state_column_is_refused():
    # A column whose variable the layout does not hold at all is refused at
    # assembly; a cell past a state's extent is a ghost read.
    shapes = {"u": [2]}
    assert check_state_column({"u"}, shapes, "u", [3], "u[3]", "u[1]") is False
    assert check_state_column({"u"}, shapes, "u", [2], "u[2]", "u[1]") is True
    with pytest.raises(BandError):
        check_state_column({"u"}, shapes, "V", [1], "V[1]", "u[1]")
