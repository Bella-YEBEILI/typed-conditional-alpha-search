import sqlite3
from pathlib import Path

import pandas as pd


class DbInteractor:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.seed_db_path = root/"excavate_data"/"seed.sqlite"
        self.interaction_db_path = root/"excavate_data"/"interaction.sqlite"
        self.genetic_db_path = root/"excavate_data"/"genetic.sqlite"
        self.product_db_path = root/"product_data"/"product.sqlite"

    def read_seed(self,filters=None,**kwargs):
        return self._read_excavate_db(self.seed_db_path,filters,kwargs)

    def read_intersection(self,kind=None,filters=None,**kwargs):
        if kind is None:
            return self._read_excavate_db(self.interaction_db_path,filters,kwargs)
        return self._read_intersection_db(kind,filters,kwargs)

    def read_interaction(self,kind=None,filters=None,**kwargs):
        return self.read_intersection(kind=kind,filters=filters,**kwargs)

    def read_genetic(self,filters=None,**kwargs):
        return self._read_excavate_db(self.genetic_db_path,filters,kwargs)

    def read_excavate(self,filters=None,**kwargs):
        frames = [
            self.read_seed(filters,**kwargs),
            self.read_intersection(filters=filters,**kwargs),
            self.read_genetic(filters,**kwargs),
        ]
        return pd.concat(frames,ignore_index=True)

    def _read_excavate_db(self,db_path,filters=None,kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        filters = self._merge_filters(filters,kwargs)
        where,params = self._build_where(filters,self._excavate_filter_columns())
        sql = f"""
        SELECT DISTINCT
            s.spec_id,
            s.formula_id,
            s.formula,
            s.decay,
            s.neutralize,
            e.ic,
            e.rankic,
            e.rankicir,
            e.turnover,
            e.coverage
        FROM spec s
        INNER JOIN eval e ON s.spec_id=e.spec_id
        {where}
        ORDER BY e.rankic DESC
        """
        with sqlite3.connect(str(db_path)) as conn:
            return pd.read_sql_query(sql,conn,params=params)

    def _read_intersection_db(self,kind,filters=None,kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        filters = self._merge_filters(filters,kwargs)
        filters["kind"] = kind
        where,params = self._build_where(filters,self._intersection_filter_columns(kind))
        sql = f"""
        SELECT DISTINCT
            s.spec_id,
            s.formula_id,
            s.formula,
            s.decay,
            s.neutralize,
            e.ic,
            e.rankic,
            e.rankicir,
            e.turnover,
            e.coverage
        FROM spec s
        INNER JOIN eval e ON s.spec_id=e.spec_id
        INNER JOIN path p ON s.spec_id=p.spec_id
        {where}
        ORDER BY e.rankic DESC
        """
        with sqlite3.connect(str(self.interaction_db_path)) as conn:
            return pd.read_sql_query(sql,conn,params=params)

    def read_product(self,eval="full",filters=None,**kwargs):
        filters = self._merge_filters(filters,kwargs)
        table = self._product_table(eval)
        columns = self._product_filter_columns(eval)
        where,params = self._build_where(filters,columns)
        score_cols = ""
        corr_cols = ""
        order_col = "p.rankic"

        if eval in ("is","os"):
            score_cols = """
            p.perf_score,
            p.structure_score,
            """
            order_col = "p.perf_score"
        elif eval=="full":
            corr_cols = """
            p.max_corr,
            p.max_corr_id,
            p.avg_corr,
            """

        sql = f"""
        SELECT
            p.spec_id,
            s.formula_id,
            p.formula,
            p.decay,
            p.neutralize,
            {score_cols}
            p.ic,
            p.rankic,
            p.rankicir,
            p.longret,
            p.turnover,
            p.coverage,
            {corr_cols}
            p.created_at,
            p.updated_at
        FROM {table} p
        INNER JOIN spec s ON p.spec_id=s.spec_id
        {where}
        ORDER BY {order_col} DESC
        """
        with sqlite3.connect(str(self.product_db_path)) as conn:
            return pd.read_sql_query(sql,conn,params=params)

    @staticmethod
    def _merge_filters(filters,kwargs):
        merged = {}
        if filters is not None:
            merged.update(filters)
        merged.update(kwargs)
        return merged

    def _build_where(self,filters,columns):
        clauses = []
        params = []
        for key,value in filters.items():
            column,op = self._split_filter_key(key)
            expr = columns[column]
            if value is None:
                if op=="ne":
                    clauses.append(f"{expr} IS NOT NULL")
                else:
                    clauses.append(f"{expr} IS NULL")
                continue

            value = self._normalize_value(value)
            if op=="eq":
                if isinstance(value,(list,tuple,set)):
                    placeholders = ",".join(["?"]*len(value))
                    clauses.append(f"{expr} IN ({placeholders})")
                    params.extend(list(value))
                else:
                    clauses.append(f"{expr}=?")
                    params.append(value)
            elif op=="ne":
                clauses.append(f"{expr}<>?")
                params.append(value)
            elif op=="gt":
                clauses.append(f"{expr}>?")
                params.append(value)
            elif op=="gte":
                clauses.append(f"{expr}>=?")
                params.append(value)
            elif op=="lt":
                clauses.append(f"{expr}<?")
                params.append(value)
            elif op=="lte":
                clauses.append(f"{expr}<=?")
                params.append(value)
            elif op=="like":
                clauses.append(f"{expr} LIKE ?")
                params.append(value)
            elif op=="in":
                placeholders = ",".join(["?"]*len(value))
                clauses.append(f"{expr} IN ({placeholders})")
                params.extend(list(value))

        if len(clauses)==0:
            return "",params
        return "WHERE "+" AND ".join(clauses),params

    @staticmethod
    def _split_filter_key(key):
        if "__" not in key:
            return key,"eq"
        column,op = key.rsplit("__",1)
        return column,op

    @staticmethod
    def _normalize_value(value):
        if isinstance(value,bool):
            return int(value)
        if isinstance(value,(list,tuple,set)):
            return [int(v) if isinstance(v,bool) else v for v in value]
        return value

    @staticmethod
    def _excavate_filter_columns():
        return {
            "spec_id":"s.spec_id",
            "formula_id":"s.formula_id",
            "formula":"s.formula",
            "decay":"s.decay",
            "neutralize":"s.neutralize",
            "ic":"e.ic",
            "rankic":"e.rankic",
            "rankicir":"e.rankicir",
            "turnover":"e.turnover",
            "coverage":"e.coverage",
        }

    def _intersection_filter_columns(self,kind):
        columns = self._excavate_filter_columns()
        columns.update({
            "kind":"p.kind",
            "op":"p.op",
            "params":"p.params",
            "seed1":"p.seed1",
            "seed1_atom":"p.seed1_atom",
            "seed_atom":"p.seed1_atom",
            "seed1_layer":"p.seed1_layer",
            "seed1_path":"p.seed1_path",
            "seed1_factory":"p.seed1_factory",
            "seed2":"p.seed2",
            "seed2_layer":"p.seed2_layer",
            "seed2_path":"p.seed2_path",
            "seed2_factory":"p.seed2_factory",
        })
        return columns

    @staticmethod
    def _product_table(eval):
        return {
            "is":"product_is",
            "os":"product_os",
            "full":"product_full",
        }[eval]

    @staticmethod
    def _product_filter_columns(eval):
        columns = {
            "spec_id":"p.spec_id",
            "formula_id":"s.formula_id",
            "formula":"p.formula",
            "decay":"p.decay",
            "neutralize":"p.neutralize",
            "ic":"p.ic",
            "rankic":"p.rankic",
            "rankicir":"p.rankicir",
            "longret":"p.longret",
            "turnover":"p.turnover",
            "coverage":"p.coverage",
            "created_at":"p.created_at",
            "updated_at":"p.updated_at",
        }
        if eval in ("is","os"):
            columns.update({
                "perf_score":"p.perf_score",
                "structure_score":"p.structure_score",
            })
        elif eval=="full":
            columns.update({
                "max_corr":"p.max_corr",
                "max_corr_id":"p.max_corr_id",
                "avg_corr":"p.avg_corr",
            })
        return columns
