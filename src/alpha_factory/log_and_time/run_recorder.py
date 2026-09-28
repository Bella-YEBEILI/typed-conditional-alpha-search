import csv
import json
import time
from datetime import datetime
from pathlib import Path


class RunRecorder:
    COLUMNS = (
        "run_type",
        "config_path",
        "atom",
        "evaluated_count",
        "seed_count",
        "candidate_count",
        "final_seed_count",
        "full_product_count",
        "started_at",
        "ended_at",
        "total_sec",
    )

    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.path = root/"excavate_data"/"run_records.csv"
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def start(self):
        return datetime.now(),time.perf_counter()

    def append(self,
               run_type:str,
               config_path,
               atom,
               evaluated_count:int,
               seed_count:int,
               candidate_count:int,
               final_seed_count,
               full_product_count:int,
               started_at,
               start_perf:float):
        ended_at = datetime.now()
        row = {
            "run_type":run_type,
            "config_path":"" if config_path is None else str(config_path),
            "atom":self._atom_text(atom),
            "evaluated_count":evaluated_count,
            "seed_count":seed_count,
            "candidate_count":candidate_count,
            "final_seed_count":final_seed_count,
            "full_product_count":full_product_count,
            "started_at":started_at.isoformat(timespec="seconds"),
            "ended_at":ended_at.isoformat(timespec="seconds"),
            "total_sec":round(time.perf_counter()-start_perf,6),
        }
        exists = self.path.exists()
        if exists:
            self._ensure_header()
        with self.path.open("a",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=self.COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _ensure_header(self):
        with self.path.open("r",newline="",encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames==list(self.COLUMNS):
                return
            rows = list(reader)

        with self.path.open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=self.COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col:row.get(col,"") for col in self.COLUMNS})

    @staticmethod
    def _atom_text(atom):
        if isinstance(atom,(list,tuple)):
            if len(atom)==1:
                return atom[0]
            return json.dumps(list(atom),ensure_ascii=False)
        return atom
