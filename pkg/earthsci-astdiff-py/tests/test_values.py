"""Numeric assembly against the Julia reference's value goldens
(tests/goldens/*.jvals.json): the (row, col) key set must match EXACTLY and
every value to 1e-10 relative (bit-equality across runtimes is unrealistic —
evaluation order differs)."""
import json

import earthsci_ast as ea
import pytest

from earthsci_astdiff import assemble_jacobian

NAMES = ("bd_chem", "adv_interior", "contracted_ops", "flux_form_adv",
         "minmod_adv")


def _cells_to_arrays(u: dict):
    """{'u[3]': v, 's': v} → {'u': [..1-based..], 's': v}"""
    arrays: dict = {}
    for cellname, v in u.items():
        if "[" not in cellname:
            arrays[cellname] = v
            continue
        name, rest = cellname.split("[", 1)
        cells = [int(x) for x in rest[:-1].split(",")]
        cur = arrays.setdefault(name, {})
        for c in cells[:-1]:
            cur = cur.setdefault(c, {})
        cur[cells[-1]] = v
    def to_list(d):
        if not isinstance(d, dict):
            return d
        n = max(d)
        return [to_list(d.get(i)) for i in range(1, n + 1)]
    return {k: to_list(v) for k, v in arrays.items()}


@pytest.mark.parametrize("name", NAMES)
def test_values_golden(name, golden_dir, valid_dir):
    g = json.loads((golden_dir / f"{name}.jvals.json").read_text())
    file = ea.load(str(valid_dir / f"{name}.esm"))
    got = assemble_jacobian(file, next(iter(file.models)),
                            states=_cells_to_arrays(g["u"]),
                            params=g["params"], t=g["t"])
    want = {(r, c): v for r, c, v in g["entries"]}
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-10, abs=1e-13), k
