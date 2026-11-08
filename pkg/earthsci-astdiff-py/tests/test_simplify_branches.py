"""Same-condition pruning through a rewritten condition (the reference's
test/cse_test.jl): inside ``q``'s true side, ``ifelse(q, x, y) < y`` is
``x < y``, which an enclosing test of ``x < y`` must still prune."""

from earthsci_astdiff import simplify_branches
from earthsci_astdiff.expr_helpers import op, skey


def _ife(c, a, b):
    return op("ifelse", c, a, b)


def test_prunes_a_condition_that_assume_rewrote():
    c = op("<", "x", "y")
    q = op(">", "z", 0)
    cq = op("<", _ife(q, "x", "y"), "y")
    e = _ife(c, _ife(q, _ife(cq, "a", "b"), "d"), "f")
    want = _ife(c, _ife(q, "a", "d"), "f")
    assert skey(simplify_branches(e)) == skey(want)
