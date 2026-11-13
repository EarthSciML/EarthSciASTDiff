"""The extraction rule of esm-jacobian-spec §4.1 (the reference's
cse_test.jl "the extraction rule" testset)."""

from earthsci_ast.esm_types import ExprNode
from earthsci_astdiff import cse_templates, skey


def N(op, *args):
    return ExprNode(op=op, args=list(args))


def ref(n):
    return ExprNode(op="apply_expression_template", args=[], name=n)


def test_counts_after_larger_extractions():
    B = N("exp", N("+", "x", "y"))
    C = N("log", N("*", "x", 2.0))
    A = N("+", N("*", B, B), C)
    tmpl, rw = cse_templates([N("-", A, "a"), N("/", A, "b")], min_nodes=4)
    # A occurs twice; B occurs twice in A's body, which is kept once, so it
    # still counts 2; C occurs once in A's body, so it counts 1 and stays.
    assert list(tmpl) == ["jt1", "jt2"]
    assert skey(tmpl["jt1"]) == skey(N("+", N("*", ref("jt2"), ref("jt2")), C))
    assert skey(tmpl["jt2"]) == skey(B)
    assert skey(rw[0]) == skey(N("-", ref("jt1"), "a"))
    assert skey(rw[1]) == skey(N("/", ref("jt1"), "b"))
    # structurally equal copies count like the shared object
    A2 = N("+", N("*", B, N("exp", N("+", "x", "y"))), N("log", N("*", "x", 2.0)))
    tmpl2, rw2 = cse_templates([N("-", A, "a"), N("/", A2, "b")], min_nodes=4)
    assert [skey(v) for v in tmpl2.values()] == [skey(v) for v in tmpl.values()]
    assert [skey(e) for e in rw2] == [skey(e) for e in rw]


def test_equal_sizes_are_named_in_first_encounter_order():
    D = N("sin", N("+", "x", "z"))
    E = N("cos", N("+", "y", "z"))
    tmpl, _ = cse_templates([N("*", E, D), N("-", D, E)], min_nodes=4)
    assert [skey(tmpl["jt1"]), skey(tmpl["jt2"])] == [skey(E), skey(D)]


def test_an_integer_leaf_is_not_a_float_leaf():
    P = N("exp", N("+", "x", 1))
    Q = N("exp", N("+", "x", 1.0))
    tmpl, _ = cse_templates([N("*", P, Q)], min_nodes=4)
    assert tmpl == {}


def test_an_operator_with_no_children_is_extracted():
    Z = N("zarg")
    tmpl, rw = cse_templates([N("*", Z, Z)], min_nodes=1)
    assert list(tmpl) == ["jt1"]
    assert skey(tmpl["jt1"]) == skey(Z)
    assert skey(rw[0]) == skey(N("*", ref("jt1"), ref("jt1")))


def test_a_deep_chain_needs_no_stack_per_level():
    e = "x"
    for _ in range(5000):
        e = N("+", e, 1)
    tmpl, rw = cse_templates([e, e], min_nodes=12)
    assert list(tmpl) == ["jt1"]
    assert tmpl["jt1"] is e  # nothing below it changed: the node itself
    assert rw[0].op == "apply_expression_template"
