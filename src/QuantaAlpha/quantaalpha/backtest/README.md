# QuantaAlpha Backtest

This folder contains the consolidated standalone TQ-compatible backtest stack used by QuantaAlpha.

Included modules:
- `workflow.py`: session-based backtest entry used by `quantaalpha backtest`
- `provider.py`: load runtime fields from `/home/workspace/common/test_data`
- `analysis.py` / `numbafunc.py` / `minute_tools.py`: vendored TQ operators and accelerated kernels
- `evaluator.py` / `result_engine.py` / `performance_engine.py`: factor value generation and metric calculation
- `bridge.py` / `config.py`: expression rendering, runtime checks, factor-file evaluation
- `quality.py`: manual research plots, group summaries, and Barra-style style exposure summaries
- `run_backtest.py`: ad hoc expression / factor-file backtests
- `manual_analysis.py`: rerun manual quality analysis from saved backtest artifacts

Typical usage:

```bash
python -m quantaalpha.backtest.run_backtest --expression "cs_rank(ts_mean(hfq_closes, 5))" --factor-name demo --plot true
python -m quantaalpha.backtest.run_backtest --library-path data/factorlib/all_factors_library.json --library-factor <factor_id_or_name> --plot true
python -m quantaalpha.backtest.manual_analysis --factor-value-file data/tq_upstream/candidates/demo_factor_value.pkl --factor-result-file data/tq_upstream/candidates/demo_factor_result.pkl --plot true
```

Outputs are written under `data/tq_upstream/candidates` by default and include:
- `*_summary.json`
- `*_factor_value.pkl`
- `*_factor_result.json`
- `*_factor_result.pkl`
- `*_quality.json`
- `*_diagnostics.png` when plotting is enabled
