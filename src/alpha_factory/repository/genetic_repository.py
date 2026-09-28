import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.factor import Factor
from ..core.id_manager import IdManager


class GeneticRepository:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.db_path = root/"excavate_data"/"genetic.sqlite"
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
                spec_id     TEXT PRIMARY KEY,
                formula_id  TEXT NOT NULL,
                formula     TEXT NOT NULL,
                decay       INTEGER NOT NULL,
                neutralize  TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS eval (
                id                              INTEGER PRIMARY KEY AUTOINCREMENT,
                spec_id                         TEXT NOT NULL,
                run_id                          TEXT NOT NULL,
                generation                      INTEGER NOT NULL,
                parent_spec_id                  TEXT,
                mutation_type                   TEXT,
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

            CREATE TABLE IF NOT EXISTS run_log (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                      TEXT NOT NULL,
                config_path                 TEXT,
                generation                  INTEGER NOT NULL,
                population_count            INTEGER,
                offspring_count             INTEGER,
                evaluated_count             INTEGER,
                seed_count                  INTEGER,
                passed_seed_count           INTEGER,
                candidate_count             INTEGER,
                candidate_prod_passed_count INTEGER,
                candidate_self_passed_count INTEGER,
                product_count               INTEGER,
                best_spec_id                TEXT,
                best_fitness                REAL,
                best_rankic                 REAL,
                started_at                  TEXT NOT NULL,
                ended_at                    TEXT NOT NULL,
                total_sec                   REAL,
                stopped_reason              TEXT
            );
            """)

    @staticmethod
    def new_run_id():
        now = datetime.now().isoformat(timespec="seconds")
        return datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+IdManager.stable_hash(now,6)

    def bind_spec_id(self,factor:Factor):
        factor.spec.formula_id = IdManager.get_formula_id(factor.spec.formula)
        factor.spec.spec_id = IdManager.get_spec_id(
            factor.spec.formula,
            factor.spec.decay,
            factor.spec.neutralize,
        )
        return factor.spec.spec_id

    def save_seed_batch(self,run_id:str,generation:int,factors:list[Factor]):
        if len(factors)==0:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            for factor in factors:
                self._save_spec(factor,now)
                self._save_eval(run_id,generation,factor,now)
                self.save_rankics(factor.spec.spec_id,factor.result.rankics)

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
            int(spec.decay),spec.neutralize,now,now,
        ))

    def _save_eval(self,run_id:str,generation:int,factor:Factor,now:str):
        r = factor.result
        self.conn.execute("""
        INSERT INTO eval (
            spec_id,run_id,generation,parent_spec_id,mutation_type,
            ic,rankic,rankicir,longret,turnover,coverage,is_candidate,is_seed,
            perf_score,structure_score,batch_corr,prod_corr,fitness,
            candidate_prod_prune_passed,candidate_prod_max_corr,candidate_prod_max_id,
            candidate_self_prune_passed,candidate_self_max_corr,candidate_self_max_id,
            seed_prune_passed,seed_max_corr,seed_max_id,created_at,updated_at
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,(
            factor.spec.spec_id,
            run_id,
            int(generation),
            getattr(factor,"genetic_parent_spec_id",None),
            getattr(factor,"genetic_mutation_type",None),
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

    def append_run_log(self,
                       run_id:str,
                       config_path:str,
                       generation:int,
                       population_count:int,
                       offspring_count:int,
                       evaluated_count:int,
                       seed_count:int,
                       passed_seed_count:int,
                       candidate_count:int,
                       candidate_prod_passed_count:int,
                       candidate_self_passed_count:int,
                       product_count:int,
                       best_spec_id,
                       best_fitness,
                       best_rankic,
                       started_at,
                       total_sec:float,
                       stopped_reason=None):
        ended_at = datetime.now()
        with self.conn:
            self.conn.execute("""
            INSERT INTO run_log (
                run_id,config_path,generation,population_count,offspring_count,evaluated_count,
                seed_count,passed_seed_count,candidate_count,candidate_prod_passed_count,
                candidate_self_passed_count,product_count,best_spec_id,best_fitness,best_rankic,
                started_at,ended_at,total_sec,stopped_reason
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,(
                run_id,config_path,int(generation),population_count,offspring_count,evaluated_count,
                seed_count,passed_seed_count,candidate_count,candidate_prod_passed_count,
                candidate_self_passed_count,product_count,best_spec_id,
                self._float_or_none(best_fitness),self._float_or_none(best_rankic),
                started_at.isoformat(timespec="seconds"),
                ended_at.isoformat(timespec="seconds"),
                float(total_sec),
                stopped_reason,
            ))

    def save_rankics(self,spec_id:str,rankics:np.ndarray):
        np.save(self.rankics_dir/f"{spec_id}.npy",rankics)

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
