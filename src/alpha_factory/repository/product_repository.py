import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.factor import Factor,FactorMeta,FactorSpec,FactorResult
from ..core.id_manager import IdManager


class ProductRepository:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.db_path = root/"product_data"/"product.sqlite"
        self.rankics_dir = root/"product_data"/"rankics"
        self.rankics_is_dir = self.rankics_dir/"is"
        self.rankics_full_dir = self.rankics_dir/"full"
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.rankics_is_dir.mkdir(parents=True,exist_ok=True)
        self.rankics_full_dir.mkdir(parents=True,exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS spec (
                spec_id     TEXT PRIMARY KEY,
                formula_id  TEXT NOT NULL,
                formula     TEXT NOT NULL,
                decay       INTEGER NOT NULL,
                neutralize  TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS product_is (
                spec_id      TEXT PRIMARY KEY,
                formula      TEXT NOT NULL,
                decay        INTEGER NOT NULL,
                neutralize   TEXT,
                perf_score   REAL,
                structure_score REAL,
                ic           REAL,
                rankic       REAL,
                rankicir     REAL,
                longret      REAL,
                turnover     REAL,
                coverage     REAL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS product_os (
                spec_id      TEXT PRIMARY KEY,
                formula      TEXT NOT NULL,
                decay        INTEGER NOT NULL,
                neutralize   TEXT,
                perf_score   REAL,
                structure_score REAL,
                ic           REAL,
                rankic       REAL,
                rankicir     REAL,
                longret      REAL,
                turnover     REAL,
                coverage     REAL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS product_full (
                spec_id    TEXT PRIMARY KEY,
                formula    TEXT NOT NULL,
                decay      INTEGER NOT NULL,
                neutralize TEXT,
                ic         REAL,
                rankic     REAL,
                rankicir   REAL,
                longret    REAL,
                turnover   REAL,
                coverage   REAL,
                max_corr   REAL,
                max_corr_id TEXT,
                avg_corr   REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)
            self._ensure_columns("product_is",{
                "formula":"TEXT",
                "decay":"INTEGER",
                "neutralize":"TEXT",
                "perf_score":"REAL",
                "structure_score":"REAL",
            })
            self._ensure_columns("product_os",{
                "formula":"TEXT",
                "decay":"INTEGER",
                "neutralize":"TEXT",
                "perf_score":"REAL",
                "structure_score":"REAL",
            })
            self._ensure_columns("product_full",{
                "formula":"TEXT",
                "decay":"INTEGER",
                "neutralize":"TEXT",
                "max_corr":"REAL",
                "max_corr_id":"TEXT",
                "avg_corr":"REAL",
            })

    def _ensure_columns(self,table:str,columns:dict):
        existing = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name,decl in columns.items():
            if name not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    @staticmethod
    def _bool_to_int(value):
        if value is None:
            return None
        return int(bool(value))

    @staticmethod
    def _float_or_none(value):
        if value is None:
            return None
        return float(value)

    def bind_spec_id(self,factor:Factor)->str:
        spec = factor.spec
        spec.formula_id = IdManager.get_formula_id(spec.formula)
        spec.spec_id = IdManager.get_spec_id(spec.formula,spec.decay,spec.neutralize)
        return spec.spec_id

    def save_is_batch(self,factors:list[Factor]):
        self._save_product_eval_batch(factors,"product_is",save_rankics=True,period="is")

    def save_os_batch(self,factors:list[Factor]):
        self._save_product_eval_batch(factors,"product_os",save_rankics=False)

    def save_full_batch(self,factors:list[Factor]):
        self._save_eval_batch(factors,"product_full",save_rankics=True,period="full")

    def _save_product_eval_batch(self,factors:list[Factor],table:str,save_rankics:bool,period:str="is"):
        if len(factors)==0:
            return

        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            for factor in factors:
                self._save_spec(factor,now)
                r = factor.result
                self.conn.execute(f"""
                INSERT INTO {table} (
                    spec_id,formula,decay,neutralize,perf_score,structure_score,ic,rankic,rankicir,longret,turnover,coverage,
                    created_at,updated_at
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(spec_id) DO UPDATE SET
                    formula=excluded.formula,
                    decay=excluded.decay,
                    neutralize=excluded.neutralize,
                    perf_score=excluded.perf_score,
                    structure_score=excluded.structure_score,
                    ic=excluded.ic,
                    rankic=excluded.rankic,
                    rankicir=excluded.rankicir,
                    longret=excluded.longret,
                    turnover=excluded.turnover,
                    coverage=excluded.coverage,
                    updated_at=excluded.updated_at
                """,(
                    factor.spec.spec_id,
                    factor.spec.formula,
                    int(factor.spec.decay),
                    factor.spec.neutralize,
                    self._float_or_none(r.perf_score),
                    self._float_or_none(r.structure_score),
                    self._float_or_none(r.ic),
                    self._float_or_none(r.rankic),
                    self._float_or_none(r.rankicir),
                    self._float_or_none(r.longret),
                    self._float_or_none(r.turnover),
                    self._float_or_none(r.coverage),
                    now,now,
                ))
                if save_rankics:
                    self.save_rankics(factor.spec.spec_id,r.rankics,period=period)

    def _save_eval_batch(self,factors:list[Factor],table:str,save_rankics:bool,period:str="is"):
        if len(factors)==0:
            return

        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            for factor in factors:
                self._save_spec(factor,now)
                r = factor.result
                self.conn.execute(f"""
                INSERT INTO {table} (
                    spec_id,formula,decay,neutralize,ic,rankic,rankicir,longret,turnover,coverage,
                    max_corr,max_corr_id,avg_corr,created_at,updated_at
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(spec_id) DO UPDATE SET
                    formula=excluded.formula,
                    decay=excluded.decay,
                    neutralize=excluded.neutralize,
                    ic=excluded.ic,
                    rankic=excluded.rankic,
                    rankicir=excluded.rankicir,
                    longret=excluded.longret,
                    turnover=excluded.turnover,
                    coverage=excluded.coverage,
                    updated_at=excluded.updated_at
                """,(
                    factor.spec.spec_id,
                    factor.spec.formula,
                    int(factor.spec.decay),
                    factor.spec.neutralize,
                    self._float_or_none(r.ic),
                    self._float_or_none(r.rankic),
                    self._float_or_none(r.rankicir),
                    self._float_or_none(r.longret),
                    self._float_or_none(r.turnover),
                    self._float_or_none(r.coverage),
                    None,None,None,
                    now,now,
                ))
                if save_rankics:
                    self.save_rankics(factor.spec.spec_id,r.rankics,period=period)

    def _save_spec(self,factor:Factor,now:str):
        self.bind_spec_id(factor)
        spec = factor.spec
        self.conn.execute("""
        INSERT INTO spec (
            spec_id,formula_id,formula,decay,neutralize,created_at,updated_at
        )
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(spec_id) DO UPDATE SET
            formula_id=excluded.formula_id,
            formula=excluded.formula,
            decay=excluded.decay,
            neutralize=excluded.neutralize,
            updated_at=excluded.updated_at
        """,(
            spec.spec_id,spec.formula_id,spec.formula,
            int(spec.decay),spec.neutralize,now,now
        ))

    def save_rankics(self,spec_id:str,rankics:np.ndarray,period:str="is"):
        np.save(self._rankics_dir(period)/f"{spec_id}.npy",rankics)

    def load_rankics(self,spec_id:str,period:str="is"):
        return np.load(self._rankics_dir(period)/f"{spec_id}.npy")

    def _rankics_dir(self,period:str):
        if period=="is":
            return self.rankics_is_dir
        if period=="full":
            return self.rankics_full_dir
        raise ValueError("period must be 'is' or 'full'")

    def load_factors(self)->list[Factor]:
        rows = self.conn.execute("""
        SELECT i.spec_id,s.formula_id,i.formula,i.decay,i.neutralize,
               i.perf_score,i.structure_score,i.ic,i.rankic,i.rankicir,i.longret,i.turnover,i.coverage
        FROM product_is i
        INNER JOIN product_full f ON i.spec_id=f.spec_id
        INNER JOIN spec s ON i.spec_id=s.spec_id
        """).fetchall()

        factors = []
        for row in rows:
            factors.append(self._row_to_factor(row,period="is"))
        return factors

    def load_full_factors(self)->list[Factor]:
        rows = self.conn.execute("""
        SELECT f.spec_id,s.formula_id,f.formula,f.decay,f.neutralize,
               f.ic,f.rankic,f.rankicir,f.longret,f.turnover,f.coverage
        FROM product_full f
        INNER JOIN spec s ON f.spec_id=s.spec_id
        """).fetchall()

        factors = []
        for row in rows:
            factors.append(self._row_to_factor(row,period="full"))
        return factors

    def load_factor(self,spec_id:str,period:str="is")->Factor:
        table = "product_is" if period=="is" else "product_full"
        score_cols = "e.perf_score,e.structure_score," if period=="is" else ""
        row = self.conn.execute(f"""
        SELECT e.spec_id,s.formula_id,e.formula,e.decay,e.neutralize,
               {score_cols}e.ic,e.rankic,e.rankicir,e.longret,e.turnover,e.coverage
        FROM {table} e
        INNER JOIN spec s ON e.spec_id=s.spec_id
        WHERE e.spec_id=?
        """,(spec_id,)).fetchone()
        return self._row_to_factor(row,period=period)

    def _row_to_factor(self,row,period:str="is")->Factor:
        spec = FactorSpec(
            formula=row["formula"],
            decay=row["decay"],
            neutralize=row["neutralize"],
            formula_id=row["formula_id"],
            spec_id=row["spec_id"],
        )
        result = FactorResult(
            ic=row["ic"],
            rankic=row["rankic"],
            rankicir=row["rankicir"],
            longret=row["longret"],
            turnover=row["turnover"],
            coverage=row["coverage"],
            rankics=self.load_rankics(row["spec_id"],period=period),
        )
        if "perf_score" in row.keys():
            result.perf_score = row["perf_score"]
            result.structure_score = row["structure_score"]
        return Factor(
            meta=FactorMeta(atom=("product",)),
            spec=spec,
            result=result,
        )

    def update_corr_fields(self,spec_id:str,max_corr,max_corr_id,avg_corr,tables:tuple[str,...]):
        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            for table in tables:
                self.conn.execute(f"""
                UPDATE {table}
                SET max_corr=?,max_corr_id=?,avg_corr=?,updated_at=?
                WHERE spec_id=?
                """,(
                    self._float_or_none(max_corr),
                    max_corr_id,
                    self._float_or_none(avg_corr),
                    now,
                    spec_id,
                ))

    def delete_factor(self,spec_id:str):
        with self.conn:
            self.conn.execute("DELETE FROM product_full WHERE spec_id=?",(spec_id,))
            self.conn.execute("DELETE FROM product_os WHERE spec_id=?",(spec_id,))
            self.conn.execute("DELETE FROM product_is WHERE spec_id=?",(spec_id,))
            self.conn.execute("DELETE FROM spec WHERE spec_id=?",(spec_id,))

        for period in ("is","full"):
            path = self._rankics_dir(period)/f"{spec_id}.npy"
            if path.exists():
                path.unlink()
