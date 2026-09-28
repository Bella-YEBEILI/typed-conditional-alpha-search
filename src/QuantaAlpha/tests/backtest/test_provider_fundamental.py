import pandas as pd
from pathlib import Path

from quantaalpha.backtest.provider import FilesystemDataProvider


def test_provider_loads_fundamental_raw_subdirectory():
    data_root = Path("data/test_tmp/provider_fundamental_case")
    fundamental_raw = data_root / "fundamental" / "raw"
    fundamental_raw.mkdir(parents=True)

    frame = pd.DataFrame(
        [[0.12, 0.08], [0.14, 0.09]],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        columns=["000001.SZ", "000002.SZ"],
    )
    frame.to_pickle(fundamental_raw / "roe.pkl")

    provider = FilesystemDataProvider(data_root, start_date="20190101")

    assert "roe" in provider.list_datas()
    loaded = provider.get_single_data("roe")
    pd.testing.assert_frame_equal(loaded, frame.rename_axis("datetime"))
