"""Time bands with data held fixed (spec §4.2): only observed that move with
``t`` are followed, and an observed declared without a shape is read at a
cell. Byte-identical to the Julia reference's golden."""

import earthsci_ast as ea
from earthsci_astdiff import jacobian_bands, serialize_band, stable_json


def test_tgrad_data_fixed_time_bands(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "tgrad_data_fixed.esm"))
    entries = jacobian_bands(file, "Box", wrt="time")
    assert len(entries) == 1
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / "tgrad_data_fixed.time.bands.json").read_text()
