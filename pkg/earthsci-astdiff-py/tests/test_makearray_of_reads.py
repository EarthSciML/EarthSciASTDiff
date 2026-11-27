"""A makearray whose region values are cell reads (scalars), read at a cell
through an observed: each value is read as itself, never wrapped in an
`index` with no subscripts. Byte-identical to the Julia reference's golden."""

import earthsci_ast as ea
from earthsci_astdiff import jacobian_bands, serialize_band, stable_json


def test_makearray_of_reads_bands(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "makearray_of_reads.esm"))
    entries = jacobian_bands(file, "M")
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / "makearray_of_reads.bands.json").read_text()
    assert [(e.u, e.v) for e in entries] == [("u", "u"), ("u", "v"), ("v", "v")]
