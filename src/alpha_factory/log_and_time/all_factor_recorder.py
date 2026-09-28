import sqlite3
from datetime import datetime
from pathlib import Path

from ..core.factor import Factor


class AllFactorRecorder:
    COLUMNS = (
        "source",
        "formula",
        "decay",
        "neutralize",
        "ic",
        "rankic",
        "rankicir",
        "coverage",
        "turnover",
        "fitness",
        "evaluated_at",
    )

    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.db_path = root/"excavate_data"/"all_factors.sqlite"
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path),timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS all_factors (
                source TEXT,
                formula TEXT,
                decay INTEGER,
                neutralize TEXT,
                ic REAL,
                rankic REAL,
                rankicir REAL,
                coverage REAL,
                turnover REAL,
                fitness REAL,
                evaluated_at TEXT
            )
        """)
        self.conn.commit()

    def save_batch(self,source:str,factors:list[Factor]):
        if len(factors)==0:
            return
        evaluated_at = datetime.now().isoformat(timespec="seconds")
        rows = [
            (
                source,
                factor.spec.formula,
                factor.spec.decay,
                factor.spec.neutralize,
                factor.result.ic,
                factor.result.rankic,
                factor.result.rankicir,
                factor.result.coverage,
                factor.result.turnover,
                factor.result.fitness,
                evaluated_at,
            )
            for factor in factors
        ]
        placeholders = ",".join(["?"]*len(self.COLUMNS))
        self.conn.executemany(
            f"INSERT INTO all_factors ({','.join(self.COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        self.conn.commit()
