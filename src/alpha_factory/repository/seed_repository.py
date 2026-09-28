import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.factor import Factor,FactorMeta,FactorSpec,FactorResult
from ..core.id_manager import IdManager


class SeedRepository:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.db_path = root/"excavate_data"/"seed.sqlite"
        self.rankics_dir = root/"excavate_data"/"rankics"
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.rankics_dir.mkdir(parents=True,exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS spec (
                spec_id     TEXT    PRIMARY KEY,
                formula_id  TEXT    NOT NULL,
                formula     TEXT    NOT NULL,
                decay       INTEGER NOT NULL,
                neutralize  TEXT,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS eval (
                spec_id                         TEXT PRIMARY KEY,
                ic                              REAL,
                rankic                          REAL,
                rankicir                        REAL,
                longret                         REAL,
                turnover                        REAL,
                coverage                        REAL,
                is_candidate                    INTEGER,
                is_seed                         INTEGER,
                perf_score                      REAL,
                structure_score                 REAL,
                batch_corr                      REAL,
                prod_corr                       REAL,
                fitness                         REAL,
                candidate_prod_prune_passed     INTEGER,
                candidate_prod_max_corr         REAL,
                candidate_prod_max_id           TEXT,
                candidate_self_prune_passed     INTEGER,
                candidate_self_max_corr         REAL,
                candidate_self_max_id           TEXT,
                seed_prune_passed               INTEGER,
                seed_max_corr                   REAL,
                seed_max_id                     TEXT,
                created_at                      TEXT NOT NULL,
                updated_at                      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS path (
                spec_id     TEXT    NOT NULL,
                atom        TEXT    NOT NULL,
                layer       INTEGER NOT NULL,
                path        TEXT    NOT NULL,
                factory     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL,
                UNIQUE(spec_id,atom,layer,path,factory)
            );
            """)
            self._ensure_columns("eval",{
                "candidate_prod_prune_passed":"INTEGER",
                "candidate_prod_max_corr":"REAL",
                "candidate_prod_max_id":"TEXT",
                "candidate_self_prune_passed":"INTEGER",
                "candidate_self_max_corr":"REAL",
                "candidate_self_max_id":"TEXT",
                "seed_prune_passed":"INTEGER",
                "seed_max_corr":"REAL",
                "seed_max_id":"TEXT",
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
    def _int_to_bool(value):
        if value is None:
            return None
        return bool(value)

    @staticmethod
    def _float_or_none(value):
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _atom_text(atom:tuple[str,...]):
        if len(atom)==1:
            return atom[0]
        return json.dumps(list(atom),ensure_ascii=False)

    def bind_spec_id(self,factor:Factor)->str:
        spec = factor.spec
        spec.formula_id = IdManager.get_formula_id(spec.formula)
        spec.spec_id = IdManager.get_spec_id(spec.formula,spec.decay,spec.neutralize)
        return spec.spec_id

    def save_eval_batch(self,factors:list[Factor]):
        if len(factors)==0:
            return

        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            for factor in factors:
                self._save_spec(factor,now)
                self._save_eval(factor,now)
                self._save_path(factor,now)
                self.save_rankics(factor.spec.spec_id,factor.result.rankics,period="is")

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

    def _save_eval(self,factor:Factor,now:str):
        r = factor.result
        self.conn.execute("""
        INSERT INTO eval (
            spec_id,ic,rankic,rankicir,longret,turnover,coverage,is_candidate,is_seed,
            perf_score,structure_score,batch_corr,prod_corr,fitness,
            candidate_prod_prune_passed,candidate_prod_max_corr,candidate_prod_max_id,
            candidate_self_prune_passed,candidate_self_max_corr,candidate_self_max_id,
            seed_prune_passed,seed_max_corr,seed_max_id,created_at,updated_at
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(spec_id) DO UPDATE SET
            ic=excluded.ic,
            rankic=excluded.rankic,
            rankicir=excluded.rankicir,
            longret=excluded.longret,
            turnover=excluded.turnover,
            coverage=excluded.coverage,
            is_candidate=excluded.is_candidate,
            is_seed=excluded.is_seed,
            perf_score=excluded.perf_score,
            structure_score=excluded.structure_score,
            batch_corr=excluded.batch_corr,
            prod_corr=excluded.prod_corr,
            fitness=excluded.fitness,
            candidate_prod_prune_passed=excluded.candidate_prod_prune_passed,
            candidate_prod_max_corr=excluded.candidate_prod_max_corr,
            candidate_prod_max_id=excluded.candidate_prod_max_id,
            candidate_self_prune_passed=excluded.candidate_self_prune_passed,
            candidate_self_max_corr=excluded.candidate_self_max_corr,
            candidate_self_max_id=excluded.candidate_self_max_id,
            seed_prune_passed=excluded.seed_prune_passed,
            seed_max_corr=excluded.seed_max_corr,
            seed_max_id=excluded.seed_max_id,
            updated_at=excluded.updated_at
        """,(
            factor.spec.spec_id,
            self._float_or_none(r.ic),
            self._float_or_none(r.rankic),
            self._float_or_none(r.rankicir),
            self._float_or_none(r.longret),
            self._float_or_none(r.turnover),
            self._float_or_none(r.coverage),
            self._bool_to_int(r.is_candidate),
            self._bool_to_int(r.is_seed),
            self._float_or_none(r.perf_score),
            self._float_or_none(r.structure_score),
            self._float_or_none(r.batch_corr),
            self._float_or_none(r.prod_corr),
            self._float_or_none(r.fitness),
            self._bool_to_int(r.candidate_prod_prune_passed),
            self._float_or_none(r.candidate_prod_max_corr),
            r.candidate_prod_max_id,
            self._bool_to_int(r.candidate_self_prune_passed),
            self._float_or_none(r.candidate_self_max_corr),
            r.candidate_self_max_id,
            self._bool_to_int(r.seed_prune_passed),
            self._float_or_none(r.seed_max_corr),
            r.seed_max_id,
            now,now,
        ))

    def _save_path(self,factor:Factor,now:str):
        meta = factor.meta
        self.conn.execute("""
        INSERT INTO path (
            spec_id,atom,layer,path,factory,created_at,updated_at
        )
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(spec_id,atom,layer,path,factory) DO UPDATE SET
            updated_at=excluded.updated_at
        """,(
            factor.spec.spec_id,
            self._atom_text(meta.atom),
            int(meta.layer),
            meta.path,
            meta.factory,
            now,now,
        ))

    def save_rankics(self,spec_id:str,rankics:np.ndarray,period:str="is"):
        np.save(self.rankics_dir/f"{spec_id}.npy",rankics)

    def load_rankics(self,spec_id:str,period:str="is"):
        return np.load(self.rankics_dir/f"{spec_id}.npy")

    def load_seeds(self,
                   atoms:list[str],
                   layers:list[int]=None,
                   paths:list[str]=None,
                   factories:list[str]=None,
                   seed_prune_passed:bool=True,
                   low_rankic=None,
                   limit=None)->list[Factor]:
        sql = """
        SELECT
            s.spec_id,s.formula_id,s.formula,s.decay,s.neutralize,
            e.ic,e.rankic,e.rankicir,e.longret,e.turnover,e.coverage,
            e.is_candidate,e.is_seed,e.perf_score,e.structure_score,e.batch_corr,e.prod_corr,e.fitness,
            e.candidate_prod_prune_passed,e.candidate_prod_max_corr,e.candidate_prod_max_id,
            e.candidate_self_prune_passed,e.candidate_self_max_corr,e.candidate_self_max_id,
            e.seed_prune_passed,e.seed_max_corr,e.seed_max_id,
            MIN(p.atom) AS atom,
            MIN(p.layer) AS layer,
            MIN(p.path) AS path,
            MIN(p.factory) AS factory
        FROM path p
        INNER JOIN spec s ON p.spec_id=s.spec_id
        INNER JOIN eval e ON p.spec_id=e.spec_id
        WHERE e.is_seed=1
        """
        params = []
        sql,params = self._append_in(sql,params,"p.atom",atoms)
        sql,params = self._append_in(sql,params,"p.layer",layers)
        sql,params = self._append_in(sql,params,"p.path",paths)
        sql,params = self._append_in(sql,params,"p.factory",factories)
        if seed_prune_passed:
            sql += " AND e.seed_prune_passed=1"
        if low_rankic is not None:
            sql += " AND e.rankic>=?"
            params.append(float(low_rankic))
        sql += " GROUP BY s.spec_id ORDER BY e.fitness DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = self.conn.execute(sql,params).fetchall()
        factors = []
        for row in rows:
            factors.append(self._row_to_factor(row))
        return factors

    def _append_in(self,sql:str,params:list,column:str,values):
        if values is None:
            return sql,params
        if not isinstance(values,(list,tuple,set)):
            values = [values]
        placeholders = ",".join(["?"]*len(values))
        sql += f" AND {column} IN ({placeholders})"
        params.extend(list(values))
        return sql,params

    def load_factor(self,spec_id:str)->Factor:
        row = self.conn.execute("""
        SELECT s.*,e.*
        FROM spec s
        LEFT JOIN eval e ON s.spec_id=e.spec_id
        WHERE s.spec_id=?
        """,(spec_id,)).fetchone()
        spec = FactorSpec(
            formula=row["formula"],
            decay=row["decay"],
            neutralize=row["neutralize"],
            formula_id=row["formula_id"],
            spec_id=row["spec_id"],
        )
        result = FactorResult()
        if "ic" in row.keys() and row["ic"] is not None:
            self._fill_result(result,row)
            result.rankics = self.load_rankics(row["spec_id"])
        return Factor(meta=FactorMeta(atom=("seed",)),spec=spec,result=result)

    def _row_to_factor(self,row)->Factor:
        spec = FactorSpec(
            formula=row["formula"],
            decay=row["decay"],
            neutralize=row["neutralize"],
            formula_id=row["formula_id"],
            spec_id=row["spec_id"],
        )
        factor = Factor(
            meta=FactorMeta(
                atom=(row["atom"],),
                layer=row["layer"],
                path=row["path"],
                factory=row["factory"],
            ),
            spec=spec,
        )
        self._fill_result(factor.result,row)
        factor.result.rankics = self.load_rankics(row["spec_id"])
        return factor

    def _fill_result(self,result:FactorResult,row):
        result.ic = row["ic"]
        result.rankic = row["rankic"]
        result.rankicir = row["rankicir"]
        result.longret = row["longret"]
        result.turnover = row["turnover"]
        result.coverage = row["coverage"]
        result.is_candidate = self._int_to_bool(row["is_candidate"])
        result.is_seed = self._int_to_bool(row["is_seed"])
        result.perf_score = row["perf_score"]
        result.structure_score = row["structure_score"]
        result.batch_corr = row["batch_corr"]
        result.prod_corr = row["prod_corr"]
        result.fitness = row["fitness"]
        if "candidate_prod_prune_passed" in row.keys():
            result.candidate_prod_prune_passed = self._int_to_bool(row["candidate_prod_prune_passed"])
            result.candidate_prod_max_corr = row["candidate_prod_max_corr"]
            result.candidate_prod_max_id = row["candidate_prod_max_id"]
            result.candidate_self_prune_passed = self._int_to_bool(row["candidate_self_prune_passed"])
            result.candidate_self_max_corr = row["candidate_self_max_corr"]
            result.candidate_self_max_id = row["candidate_self_max_id"]
            result.seed_prune_passed = self._int_to_bool(row["seed_prune_passed"])
            result.seed_max_corr = row["seed_max_corr"]
            result.seed_max_id = row["seed_max_id"]
