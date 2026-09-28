import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.factor import Factor,FactorMeta,FactorSpec,FactorResult
from ..core.id_manager import IdManager
from ..interaction_strategy.job_key import make_job_key


class InteractionRepository:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.db_path = root/"excavate_data"/"interaction.sqlite"
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
                perf_score                      REAL,
                structure_score                 REAL,
                prod_corr                       REAL,
                fitness                         REAL,
                is_candidate                    INTEGER,
                candidate_prod_prune_passed     INTEGER,
                candidate_prod_max_corr         REAL,
                candidate_prod_max_id           TEXT,
                candidate_self_prune_passed     INTEGER,
                candidate_self_max_corr         REAL,
                candidate_self_max_id           TEXT
            );

            CREATE TABLE IF NOT EXISTS path (
                spec_id        TEXT PRIMARY KEY,
                kind           TEXT NOT NULL,
                op             TEXT NOT NULL,
                params         TEXT NOT NULL,
                seed1          TEXT NOT NULL,
                seed1_atom     TEXT,
                seed1_layer    INTEGER,
                seed1_path     TEXT,
                seed1_factory  TEXT,
                seed2          TEXT,
                seed2_layer    INTEGER,
                seed2_path     TEXT,
                seed2_factory  TEXT
            );
            """)

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

    def save_eval_batch(self,factors:list[Factor]):
        if len(factors)==0:
            return

        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            for factor in factors:
                self._save_spec(factor,now)
                self._save_eval(factor)
                if hasattr(factor,"interaction_path"):
                    self._save_path(factor)

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

    def _save_eval(self,factor:Factor):
        r = factor.result
        self.conn.execute("""
        INSERT INTO eval (
            spec_id,ic,rankic,rankicir,longret,turnover,coverage,
            perf_score,structure_score,prod_corr,fitness,is_candidate,
            candidate_prod_prune_passed,candidate_prod_max_corr,candidate_prod_max_id,
            candidate_self_prune_passed,candidate_self_max_corr,candidate_self_max_id
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(spec_id) DO UPDATE SET
            ic=excluded.ic,
            rankic=excluded.rankic,
            rankicir=excluded.rankicir,
            longret=excluded.longret,
            turnover=excluded.turnover,
            coverage=excluded.coverage,
            perf_score=excluded.perf_score,
            structure_score=excluded.structure_score,
            prod_corr=excluded.prod_corr,
            fitness=excluded.fitness,
            is_candidate=excluded.is_candidate,
            candidate_prod_prune_passed=excluded.candidate_prod_prune_passed,
            candidate_prod_max_corr=excluded.candidate_prod_max_corr,
            candidate_prod_max_id=excluded.candidate_prod_max_id,
            candidate_self_prune_passed=excluded.candidate_self_prune_passed,
            candidate_self_max_corr=excluded.candidate_self_max_corr,
            candidate_self_max_id=excluded.candidate_self_max_id
        """,(
            factor.spec.spec_id,
            self._float_or_none(r.ic),
            self._float_or_none(r.rankic),
            self._float_or_none(r.rankicir),
            self._float_or_none(r.longret),
            self._float_or_none(r.turnover),
            self._float_or_none(r.coverage),
            self._float_or_none(r.perf_score),
            self._float_or_none(r.structure_score),
            self._float_or_none(r.prod_corr),
            self._float_or_none(r.fitness),
            self._bool_to_int(r.is_candidate),
            self._bool_to_int(r.candidate_prod_prune_passed),
            self._float_or_none(r.candidate_prod_max_corr),
            r.candidate_prod_max_id,
            self._bool_to_int(r.candidate_self_prune_passed),
            self._float_or_none(r.candidate_self_max_corr),
            r.candidate_self_max_id,
        ))

    def _save_path(self,factor:Factor):
        record = factor.interaction_path
        params = record["params"]
        if not isinstance(params,str):
            params = json.dumps(list(params),sort_keys=True,ensure_ascii=False)
        self.conn.execute("""
        INSERT INTO path (
            spec_id,kind,op,params,seed1,seed1_atom,seed1_layer,seed1_path,seed1_factory,
            seed2,seed2_layer,seed2_path,seed2_factory
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(spec_id) DO UPDATE SET
            kind=excluded.kind,
            op=excluded.op,
            params=excluded.params,
            seed1=excluded.seed1,
            seed1_atom=excluded.seed1_atom,
            seed1_layer=excluded.seed1_layer,
            seed1_path=excluded.seed1_path,
            seed1_factory=excluded.seed1_factory,
            seed2=excluded.seed2,
            seed2_layer=excluded.seed2_layer,
            seed2_path=excluded.seed2_path,
            seed2_factory=excluded.seed2_factory
        """,(
            factor.spec.spec_id,
            record["kind"],
            record["op"],
            params,
            record["seed1"],
            record.get("seed1_atom"),
            record.get("seed1_layer"),
            record.get("seed1_path"),
            record.get("seed1_factory"),
            record.get("seed2"),
            record.get("seed2_layer"),
            record.get("seed2_path"),
            record.get("seed2_factory"),
        ))

    def load_done_jobs(self)->set[str]:
        rows = self.conn.execute("""
        SELECT kind,op,params,seed1,seed2
        FROM path
        """).fetchall()
        done = set()
        for row in rows:
            params = json.loads(row["params"])
            done.add(make_job_key(
                job_type=row["kind"],
                left_formula_id=row["seed1"],
                right_formula_id=row["seed2"],
                op=row["op"],
                params=params,
            ))
        return done

    def save_rankics(self,spec_id:str,rankics:np.ndarray,period:str="is"):
        np.save(self.rankics_dir/f"{spec_id}.npy",rankics)

    def load_rankics(self,spec_id:str,period:str="is"):
        return np.load(self.rankics_dir/f"{spec_id}.npy")

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
        if row["ic"] is not None:
            result.ic = row["ic"]
            result.rankic = row["rankic"]
            result.rankicir = row["rankicir"]
            result.longret = row["longret"]
            result.turnover = row["turnover"]
            result.coverage = row["coverage"]
            result.rankics = self.load_rankics(row["spec_id"])
        return Factor(meta=FactorMeta(atom=("interaction",)),spec=spec,result=result)
