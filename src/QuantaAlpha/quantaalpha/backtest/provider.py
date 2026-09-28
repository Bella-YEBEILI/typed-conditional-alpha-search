from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantaalpha.runtime import (
    data_root,
    fundamental_dir,
    industry_dir,
    limit_dir,
    minute_data_root,
    minute_vwap_dir,
    pv_dir,
    status_dir,
    style_derived_dir,
    universe_dir,
)

PV_FIELD_FILES = {
    "opens": "opens",
    "closes": "closes",
    "highs": "highs",
    "lows": "lows",
    "hfq_opens": "hfq_opens",
    "hfq_closes": "hfq_closes",
    "hfq_highs": "hfq_highs",
    "hfq_lows": "hfq_lows",
    "volumes": "volumes",
    "vwaps": "vwaps",
    "turnovers": "turnovers",
    "ctc_returns": "ctc_returns",
    "adj_factors": "adj_factors",
}

LEGACY_PV_FIELD_ALIASES = {
    "volume": "volumes",
    "vwap": "vwaps",
    "turnover": "turnovers",
}

PV_FIELD_ALIASES = {
    **PV_FIELD_FILES,
    **LEGACY_PV_FIELD_ALIASES,
}

MINUTE_FIELD_ALIASES = {
    "minute_vwap": "open_vwaps_1m",
}

STYLE_FIELDS = ("Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol")
STATUS_BOOL_FIELDS = {
    "tradables",
}
LIMIT_BOOL_FIELDS = {
    "limit_up_cto",
    "limit_down_cto",
    "limit_up_ctc",
    "limit_down_ctc",
}
UNIVERSE_FALLBACK_FIELDS = (
    "all",
    "stables",
    "standards",
    "smalls",
    "hs300s",
    "sz50s",
    "zz500s",
    "zz800s",
    "zz1000s",
    "zzhls",
)


def _read_pickle_compat(path: Path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise

    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _ensure_datetime_index(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index.astype(str))
    return data.sort_index()


def _frame_from_any(obj, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        frame = obj.copy()
    elif isinstance(obj, pd.Series):
        frame = obj.to_frame(name=name)
    else:
        frame = pd.DataFrame(obj)
    frame = _ensure_datetime_index(frame)
    frame.index.name = "datetime"
    return frame.sort_index(axis=1)


class FilesystemDataProvider:
    """Filesystem provider for QuantaAlpha test-data directories."""

    def __init__(self, data_dir: str | Path, start_date: str = "20150101"):
        self.data_dir = Path(data_dir)
        self.start_date = str(start_date)
        self._cache: dict[str, pd.DataFrame] = {}
        self._pv_anchor_frame: pd.DataFrame | None = None

        self._pv_dir = pv_dir() if self.data_dir == Path(pv_dir()).parents[0] else self.data_dir / "pv"
        self._fundamental_dir = (
            fundamental_dir() if self.data_dir == Path(fundamental_dir()).parents[0] else self.data_dir / "fundamental"
        )
        self._universe_dir = universe_dir() if self.data_dir == Path(universe_dir()).parents[0] else self.data_dir / "universe"
        self._status_dir = status_dir() if self.data_dir == Path(status_dir()).parents[0] else self.data_dir / "status"
        self._limit_dir = limit_dir() if self.data_dir == Path(limit_dir()).parents[0] else self.data_dir / "limit"
        self._industry_dir = industry_dir() if self.data_dir == Path(industry_dir()).parents[0] else self.data_dir / "industry"
        self._style_dir = (
            style_derived_dir() if self.data_dir == Path(style_derived_dir()).parents[1] else self.data_dir / "styles" / "derived"
        )
        daily_root = data_root()
        minute_root = minute_data_root()
        if self.data_dir in {Path(minute_root), Path(daily_root)}:
            self._minute_vwap_dir = minute_vwap_dir()
        else:
            self._minute_vwap_dir = self.data_dir / "minute_vwaps"

        self._fundamental_inventory = self._scan_fundamental_fields(self._fundamental_dir)
        self._universe_inventory = self._scan_dir(self._universe_dir)
        self._status_inventory = self._scan_dir(self._status_dir)
        self._limit_inventory = self._scan_dir(self._limit_dir)
        self._style_inventory = self._scan_dir(self._style_dir)

    @staticmethod
    def _scan_dir(folder: Path) -> set[str]:
        if not folder.exists():
            return set()
        return {file_path.stem for file_path in folder.glob("*.pkl")}

    @classmethod
    def _scan_fundamental_fields(cls, folder: Path) -> set[str]:
        fields = cls._scan_dir(folder)
        fields.update(cls._scan_dir(folder / "raw"))
        return fields

    def _fundamental_field_path(self, field_name: str) -> Path:
        raw_path = self._fundamental_dir / "raw" / f"{field_name}.pkl"
        if raw_path.exists():
            return raw_path
        return self._fundamental_dir / f"{field_name}.pkl"

    def _list_supported_fields(self) -> list[str]:
        fields = set(PV_FIELD_FILES.keys())
        fields.update(LEGACY_PV_FIELD_ALIASES.keys())
        fields.update(self._fundamental_inventory)
        fields.update(self._universe_inventory or set(UNIVERSE_FALLBACK_FIELDS))
        fields.update(self._status_inventory)
        fields.update(self._limit_inventory)
        fields.update(STATUS_BOOL_FIELDS)
        fields.update(self._style_inventory)
        fields.update(MINUTE_FIELD_ALIASES.keys())
        fields.update({"returns", "tradables", "industrys"})
        return sorted(fields)

    def list_datas(self) -> list[str]:
        return self._list_supported_fields()

    def _load_pickle_frame(self, path: Path, field_name: str) -> pd.DataFrame:
        if not path.exists():
            raise KeyError(f"field '{field_name}' not found in {path.parent}")
        return _frame_from_any(_read_pickle_compat(path), field_name)

    def _get_pv_anchor_frame(self) -> pd.DataFrame:
        if self._pv_anchor_frame is not None:
            return self._pv_anchor_frame

        cached_closes = self._cache.get("closes")
        if cached_closes is not None:
            self._pv_anchor_frame = cached_closes
            return self._pv_anchor_frame

        closes_path = self._pv_dir / f"{PV_FIELD_ALIASES['closes']}.pkl"
        closes_frame = self._load_pickle_frame(closes_path, "closes").loc[self.start_date :]
        self._cache["closes"] = closes_frame
        self._pv_anchor_frame = closes_frame
        return self._pv_anchor_frame

    def _align_pv_frame(self, field_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        if field_name == "closes":
            self._pv_anchor_frame = frame
            return frame

        anchor = self._get_pv_anchor_frame()
        if frame.index.equals(anchor.index) and frame.columns.equals(anchor.columns):
            return frame
        return frame.reindex(index=anchor.index, columns=anchor.columns)

    def _base_shape_frame(self) -> pd.DataFrame:
        return self._get_pv_anchor_frame().copy()

    def _empty_bool_frame(self) -> pd.DataFrame:
        base = self._base_shape_frame()
        return pd.DataFrame(False, index=base.index, columns=base.columns, dtype=bool)

    def _load_field(self, field_name: str) -> pd.DataFrame:
        if field_name in self._cache:
            return self._cache[field_name]

        frame: pd.DataFrame
        if field_name in PV_FIELD_ALIASES:
            frame = self._load_pickle_frame(self._pv_dir / f"{PV_FIELD_ALIASES[field_name]}.pkl", field_name)
            frame = frame.loc[self.start_date :]
            frame = self._align_pv_frame(field_name, frame)
        elif field_name in self._fundamental_inventory:
            frame = self._load_pickle_frame(self._fundamental_field_path(field_name), field_name)
        elif field_name in self._universe_inventory or field_name in UNIVERSE_FALLBACK_FIELDS:
            frame = self._load_pickle_frame(self._universe_dir / f"{field_name}.pkl", field_name).astype(bool)
        elif field_name in STATUS_BOOL_FIELDS:
            status_path = self._status_dir / f"{field_name}.pkl"
            if status_path.exists():
                frame = self._load_pickle_frame(status_path, field_name).astype(bool)
            elif field_name == "tradables":
                standards = self.get_single_data("standards").astype(bool)
                opens = self.get_single_data("opens")
                closes = self.get_single_data("closes")
                frame = (standards & opens.notna() & closes.notna()).astype(bool)
            else:
                frame = self._empty_bool_frame()
        elif field_name in self._limit_inventory:
            frame = self._load_pickle_frame(self._limit_dir / f"{field_name}.pkl", field_name).astype(bool)
        elif field_name in self._status_inventory:
            frame = self._load_pickle_frame(self._status_dir / f"{field_name}.pkl", field_name)
        elif field_name == "industrys":
            frame = self._load_pickle_frame(self._industry_dir / "industrys.pkl", field_name)
        elif field_name in self._style_inventory or field_name in STYLE_FIELDS:
            frame = self._load_pickle_frame(self._style_dir / f"{field_name}.pkl", field_name)
        elif field_name in MINUTE_FIELD_ALIASES:
            frame = self._load_pickle_frame(self._minute_vwap_dir / f"{MINUTE_FIELD_ALIASES[field_name]}.pkl", field_name)
        elif field_name == "returns":
            closes = self.get_single_data("closes")
            frame = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            frame.index.name = "datetime"
        else:
            raise KeyError(f"field '{field_name}' not found in {self.data_dir}")

        if field_name not in PV_FIELD_ALIASES:
            frame = frame.loc[self.start_date :]
        self._cache[field_name] = frame
        return frame

    def get_single_data(self, fld: str) -> pd.DataFrame:
        return self._load_field(fld)

    def get_datas(self, data_needed: Iterable[str]) -> dict[str, pd.DataFrame]:
        return {fld: self.get_single_data(fld) for fld in data_needed}


class StaticFieldDataProvider:
    """List-only provider used when runtime data is unavailable."""

    def __init__(self, fields: Iterable[str]):
        self._fields = sorted({str(field) for field in fields})

    def list_datas(self) -> list[str]:
        return list(self._fields)

    def get_single_data(self, fld: str) -> pd.DataFrame:
        raise RuntimeError(
            f"StaticFieldDataProvider only supports list_datas(); field '{fld}' requires a real tq.data_dir"
        )

    def get_datas(self, data_needed: Iterable[str]) -> dict[str, pd.DataFrame]:
        raise RuntimeError(
            "StaticFieldDataProvider only supports list_datas(); get_datas() requires a real tq.data_dir"
        )
