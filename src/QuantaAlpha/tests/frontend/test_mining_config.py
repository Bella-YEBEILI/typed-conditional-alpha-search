import importlib.util
import json
from pathlib import Path


def _load_backend_app():
    app_path = Path(__file__).resolve().parents[2] / "frontend-v2" / "backend" / "app.py"
    spec = importlib.util.spec_from_file_location("qa_frontend_backend_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frontend_mining_settings_are_applied_to_runtime_config():
    backend_app = _load_backend_app()
    req = backend_app.MiningStartRequest(
        direction="pv",
        numDirections=3,
        maxRounds=4,
        maxLoops=5,
        factorsPerHypothesis=6,
        parallelEnabled=True,
        qualityGateEnabled=False,
    )
    run_cfg = {}

    settings = backend_app._effective_mining_settings(req)
    backend_app._apply_mining_overrides(run_cfg, settings)

    assert run_cfg["planning"]["num_directions"] == 3
    assert run_cfg["evolution"]["max_rounds"] == 4
    assert run_cfg["execution"]["max_loops"] == 5
    assert run_cfg["factor"]["factors_per_hypothesis"] == 6
    assert run_cfg["evolution"]["parallel_enabled"] is True
    assert run_cfg["execution"]["parallel_execution"] is True
    assert run_cfg["quality_gate"]["complexity_enabled"] is False
    assert run_cfg["quality_gate"]["redundancy_enabled"] is False
    assert run_cfg["quality_gate"]["consistency_enabled"] is False


def test_frontend_mining_settings_are_normalized_before_logging():
    backend_app = _load_backend_app()
    req = backend_app.MiningStartRequest(
        direction="pv",
        numDirections=0,
        maxRounds=-3,
        maxLoops=None,
        factorsPerHypothesis=0,
    )

    settings = backend_app._effective_mining_settings(req)

    assert settings["numDirections"] == 1
    assert settings["maxRounds"] == 1
    assert settings["maxLoops"] == 2
    assert settings["factorsPerHypothesis"] == 1


def test_factor_library_list_hides_libraries_without_medium_or_high_factors(tmp_path):
    backend_app = _load_backend_app()
    base_library = tmp_path / "all_factors_library.json"
    mined_library = tmp_path / "all_factors_library_exp_20260522.json"
    medium_library = tmp_path / "all_factors_library_medium.json"
    low_quality_payload = {
        "factors": {
            "factor_a": {
                "name": "factor_a",
                "train_check_flags": {"check_passed": False},
                "test_check_flags": {"check_passed": False},
            }
        }
    }
    medium_quality_payload = {
        "factors": {
            "factor_b": {
                "name": "factor_b",
                "train_check_flags": {"check_passed": True},
                "test_check_flags": {"check_passed": False},
            }
        }
    }
    base_library.write_text(json.dumps(low_quality_payload), encoding="utf-8")
    mined_library.write_text(json.dumps(low_quality_payload), encoding="utf-8")
    medium_library.write_text(json.dumps(medium_quality_payload), encoding="utf-8")

    backend_app._find_factor_jsons = lambda: [str(base_library), str(mined_library), str(medium_library)]
    backend_app._ensure_quality_bucket_libraries = lambda: []

    displayable = [Path(path).name for path in backend_app._displayable_factor_jsons()]

    assert displayable == ["all_factors_library_medium.json"]


def test_quality_bucket_libraries_are_materialized_from_backend_libraries(tmp_path):
    backend_app = _load_backend_app()
    source_library = tmp_path / "all_factors_library_source.json"
    source_library.write_text(
        json.dumps(
            {
                "metadata": {"source": "test"},
                "factors": {
                    "high_factor": {
                        "factor_id": "high_factor",
                        "train_check_flags": {"check_passed": True},
                        "test_check_flags": {"check_passed": True},
                    },
                    "medium_factor": {
                        "factor_id": "medium_factor",
                        "train_check_flags": {"check_passed": True},
                        "test_check_flags": {"check_passed": False},
                    },
                    "low_factor": {
                        "factor_id": "low_factor",
                        "train_check_flags": {"check_passed": False},
                        "test_check_flags": {"check_passed": False},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    backend_app._find_factor_jsons = lambda: [str(source_library)]
    written = [Path(path).name for path in backend_app._ensure_quality_bucket_libraries(tmp_path)]

    assert written == ["high_quality_factors_library.json", "medium_quality_factors_library.json"]

    high = json.loads((tmp_path / "high_quality_factors_library.json").read_text(encoding="utf-8"))
    medium = json.loads((tmp_path / "medium_quality_factors_library.json").read_text(encoding="utf-8"))

    assert list(high["factors"]) == ["high_factor"]
    assert list(medium["factors"]) == ["medium_factor"]


def test_ml_backtest_walk_forward_settings_are_applied_to_cli_command(tmp_path):
    backend_app = _load_backend_app()
    req = backend_app.MLBacktestStartRequest(
        model="xgb",
        optuna_trials=7,
        label_type="direction_classification",
        split_mode="walk_forward",
        rolling_window=999,
        train_lookback=500,
        validation_size=60,
        gap_size=1,
        test_size=20,
        top_n=50,
        walk_test_start="2024-01-01",
        walk_test_end="2024-12-31",
        factor_library="high_quality_factors_library.json",
        quality_filter="high",
    )
    cmd = backend_app._build_ml_backtest_command(req, tmp_path / "high_quality_factors_library.json")

    expected_pairs = {
        "--model": "xgb",
        "--optuna-trials": "7",
        "--label-type": "direction_classification",
        "--rolling-window": "999",
        "--validation-size": "60",
        "--gap-size": "1",
        "--test-size": "20",
        "--top-n": "50",
        "--quality-filter": "high",
        "--split-mode": "walk_forward",
        "--train-lookback": "500",
        "--walk-test-start": "2024-01-01",
        "--walk-test-end": "2024-12-31",
    }
    for flag, value in expected_pairs.items():
        assert cmd[cmd.index(flag) + 1] == value


def test_ml_backtest_result_line_populates_task_metrics():
    backend_app = _load_backend_app()
    task = {"metrics": {}}
    line = (
        "__ML_RESULT__:"
        + json.dumps(
            {
                "IC": 0.12,
                "top_n": 5,
                "split_mode": "walk_forward",
                "daily_selected_stocks": [{"date": "2024-01-02", "stocks": ["000001.SZ"]}],
            }
        )
    )

    assert backend_app._ingest_ml_result_line(task, line) is True
    assert task["metrics"]["IC"] == 0.12
    assert task["metrics"]["top_n"] == 5
    assert task["metrics"]["daily_selected_stocks"][0]["stocks"] == ["000001.SZ"]
