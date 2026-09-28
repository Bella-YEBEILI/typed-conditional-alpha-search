from quantaalpha.factors.tq_runner import _extract_minute_inputs


def test_extract_minute_inputs_detects_direct_h5_dataset_reads():
    module_text = """
def prepare_minute_datas():
    with h5py.File("dummy.h5", "r") as handle:
        highs_ds = handle["data/highs"]
        lows_ds = handle["data/lows"]
        closes_ds = handle["data/closes"]
    return {"demo_feature": None}
"""
    assert _extract_minute_inputs(module_text) == {"highs", "lows", "closes", "demo_feature"}
