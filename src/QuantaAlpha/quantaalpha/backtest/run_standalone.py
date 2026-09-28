from __future__ import annotations

import os
from pathlib import Path

import fire
import numpy as np

from .bridge import TQUpstreamBridge


def main(factor_file: str, output_path: str = "result.h5", context: str = "standalone") -> str:
    factor_path = Path(factor_file)
    out_path = Path(output_path)
    result = TQUpstreamBridge().evaluate_factor_file_value(factor_path, context=context).astype(np.float64)
    if out_path.exists():
        os.remove(out_path)
    result.to_hdf(out_path, key="data", mode="w")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(main)
