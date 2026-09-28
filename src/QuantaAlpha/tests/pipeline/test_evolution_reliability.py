import json

from quantaalpha.factors.proposal import (
    _append_current_round_guidance,
    _build_factor_refill_feedback,
    _resolve_max_refill_attempts,
    _resolve_refill_request_count,
)
from quantaalpha.pipeline.factor_mining import (
    _build_domain_seed_direction,
    _has_trajectory_output,
    _record_empty_task_attempt,
    _run_evolution_task,
)
from quantaalpha.pipeline.evolution import EvolutionConfig, EvolutionController, RoundPhase, StrategyTrajectory
from quantaalpha.pipeline.evolution.crossover import CrossoverOperator


def test_current_round_guidance_is_appended_to_existing_history_context():
    context = _append_current_round_guidance(
        "Previous hypothesis and feedback.",
        "## Crossover Round Guidance\nUse parent role split explicitly.",
    )

    assert "Previous hypothesis and feedback." in context
    assert "Current Round Guidance" in context
    assert "Crossover Round Guidance" in context


def test_empty_loop_result_is_not_usable_trajectory_output():
    assert not _has_trajectory_output(
        {
            "hypothesis": object(),
            "experiment": None,
            "feedback": None,
        }
    )


def test_loop_result_with_factor_and_feedback_is_usable_trajectory_output():
    class Experiment:
        sub_tasks = [object()]

    assert _has_trajectory_output(
        {
            "hypothesis": object(),
            "experiment": Experiment(),
            "feedback": object(),
        }
    )


def test_empty_task_attempt_retries_before_counting_complete():
    attempts = {}
    task = {"phase": RoundPhase.CROSSOVER, "round_idx": 2, "direction_id": 4}

    assert _record_empty_task_attempt(attempts, task, max_retries=2)
    assert not _record_empty_task_attempt(attempts, task, max_retries=2)


def test_joint_reselect_seed_uses_joint_direction_pool_only(tmp_path, monkeypatch):
    def write_direction_file(filename, descriptions):
        payload = {
            "factor_portfolios": [
                {"description": description}
                for description in descriptions
            ]
        }
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")

    write_direction_file("original_direction_joint.json", ["joint one", "joint two"])
    write_direction_file("original_direction_daily.json", ["pv only"])
    write_direction_file("original_direction_minutes.json", ["minutes only"])
    monkeypatch.setattr("quantaalpha.pipeline.factor_mining.random.choice", lambda items: items[0])
    used_identities = {}

    first_seed = _build_domain_seed_direction(tmp_path, ("pv", "minutes"), used_identities)
    second_seed = _build_domain_seed_direction(tmp_path, ("pv", "minutes"), used_identities)

    assert first_seed == "joint one"
    assert second_seed == "joint two"
    assert used_identities == {"joint": {"joint one", "joint two"}}


def test_single_domain_reselect_seed_keeps_domain_specific_pool(tmp_path, monkeypatch):
    payload = {"factor_portfolios": [{"description": "pv only"}]}
    (tmp_path / "original_direction_daily.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("quantaalpha.pipeline.factor_mining.random.choice", lambda items: items[0])
    used_identities = {}

    seed = _build_domain_seed_direction(tmp_path, ("pv",), used_identities)

    assert seed == "pv only"
    assert used_identities == {"pv": {"pv only"}}


def test_factor_refill_feedback_requests_replacements_for_filtered_candidates():
    feedback = _build_factor_refill_feedback(
        target_factor_count=3,
        accepted_count=1,
        skipped_similar=["A: exact_duplicate_expression", "B: duplicated_subtree_size=8>5"],
    )

    assert "1/3" in feedback
    assert "2 new replacement" in feedback
    assert "distinct operator families" in feedback


def test_refill_attempt_budget_is_large_enough_to_fill_partial_batches():
    assert _resolve_max_refill_attempts(1) >= 6
    assert _resolve_max_refill_attempts(3) >= 9


def test_refill_request_count_matches_missing_slots_only():
    assert _resolve_refill_request_count(3, 0) == 3
    assert _resolve_refill_request_count(3, 1) == 2
    assert _resolve_refill_request_count(3, 2) == 1
    assert _resolve_refill_request_count(3, 3) == 0


def test_evolution_controller_stops_when_empty_crossover_reaches_max_rounds():
    controller = EvolutionController(
        EvolutionConfig(
            num_directions=1,
            max_rounds=3,
            mutation_enabled=True,
            crossover_enabled=True,
            crossover_size=2,
            crossover_n=1,
        )
    )

    original_task = controller.get_next_task()
    assert original_task is not None
    controller.report_task_complete(
        original_task,
        StrategyTrajectory(
            trajectory_id="orig",
            direction_id=0,
            round_idx=0,
            phase=RoundPhase.ORIGINAL,
            factors=[{"factor_name": "orig_factor", "factor_expression": "returns"}],
            backtest_metrics={"rankicir": 1.0},
        ),
    )

    mutation_task = controller.get_next_task()
    assert mutation_task is not None
    assert mutation_task["phase"] == RoundPhase.MUTATION

    assert controller.get_next_task() is None
    assert controller.is_complete()


def test_parent_trajectory_history_is_marked_as_backtested(monkeypatch):
    captured = {}

    class FakeAlphaAgentLoop:
        def __init__(self, *args, **kwargs):
            captured["initial_trace_history"] = kwargs["initial_trace_history"]

        def run(self, step_n, stop_event=None):
            return None

        def _get_trajectory_data(self):
            return {"hypothesis": None, "experiment": None, "feedback": None}

    monkeypatch.setattr(
        "quantaalpha.pipeline.factor_mining.AlphaAgentLoop",
        FakeAlphaAgentLoop,
    )

    class FakeFactorMiningExperiment:
        def __init__(self, sub_tasks):
            self.sub_tasks = sub_tasks
            self.sub_workspace_list = [None] * len(sub_tasks)
            self.based_experiments = []
            self.result = None

    monkeypatch.setattr(
        "quantaalpha.pipeline.factor_mining.FactorMiningExperiment",
        FakeFactorMiningExperiment,
    )

    parent = StrategyTrajectory(
        trajectory_id="parent",
        direction_id=0,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        hypothesis="parent hypothesis",
        hypothesis_details={"concise_observation": "obs"},
        factors=[
            {
                "factor_name": "parent_factor",
                "factor_description": "desc",
                "factor_formulation": "form",
                "factor_expression": "ts_mean(returns, 5)",
            }
        ],
        backtest_metrics={"rankicir": 1.2},
        feedback="parent feedback",
        feedback_details={"observations": "ok"},
    )

    _run_evolution_task(
        {
            "phase": RoundPhase.MUTATION,
            "direction_id": 0,
            "round_idx": 1,
            "parent_trajectories": [parent],
            "strategy_suffix": "mutate",
        },
        directions=["direction"],
        step_n=1,
        use_local=True,
        user_direction="direction",
        log_root=".",
        stop_event=None,
        quality_gate_cfg={},
    )

    history = captured["initial_trace_history"]
    assert len(history) == 1
    assert history[0][1].result == {"parent_factor": parent.backtest_metrics}


def test_failed_original_trajectory_still_enters_mutation():
    controller = EvolutionController(
        EvolutionConfig(
            num_directions=1,
            max_rounds=3,
            mutation_enabled=True,
            crossover_enabled=False,
        )
    )
    task = controller.get_next_task()
    assert task is not None

    parent = StrategyTrajectory(
        trajectory_id="failed-original",
        direction_id=0,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        hypothesis="failed hypothesis still has repair value",
        feedback="rankic and drawdown failed; repair these metrics",
        submission_checks={
            "factor_a": {
                "check_passed": False,
                "rankic_passed": False,
                "long_maxdd_passed": False,
                "failed_metrics": ["rankic", "long_maxdd"],
            }
        },
    )
    controller.report_task_complete(task, parent)

    mutation_task = controller.get_next_task()

    assert mutation_task is not None
    assert mutation_task["phase"] == RoundPhase.MUTATION
    assert mutation_task["parent_trajectories"][0].trajectory_id == "failed-original"
    assert "rankic" in mutation_task["strategy_suffix"]


def test_failed_original_trajectories_still_enter_crossover():
    controller = EvolutionController(
        EvolutionConfig(
            num_directions=2,
            max_rounds=2,
            mutation_enabled=False,
            crossover_enabled=True,
            crossover_size=2,
            crossover_n=1,
        )
    )
    for direction_id in range(2):
        task = controller.get_next_task()
        assert task is not None
        controller.report_task_complete(
            task,
            StrategyTrajectory(
                trajectory_id=f"failed-original-{direction_id}",
                direction_id=direction_id,
                round_idx=0,
                phase=RoundPhase.ORIGINAL,
                hypothesis=f"failed hypothesis {direction_id}",
                feedback="use failed checks as crossover material",
                submission_checks={
                    f"factor_{direction_id}": {
                        "check_passed": False,
                        "raw_passed": direction_id == 0,
                        "complete_passed": False,
                        "failed_metrics": ["rankic"] if direction_id == 0 else ["long_netir"],
                    }
                },
            ),
        )

    controller._prepare_crossover_groups()

    assert len(controller._crossover_groups) == 1
    assert {
        parent.trajectory_id
        for parent in controller._crossover_groups[0]["parents"]
    } == {"failed-original-0", "failed-original-1"}


def test_crossover_prompt_uses_metric_snapshot_without_collapsed_score(monkeypatch):
    op = CrossoverOperator()
    monkeypatch.setattr(
        op,
        "generate_crossover",
        lambda *args, **kwargs: {
            "hybrid_hypothesis": "combine detailed metric profiles",
            "combination_rationale": "use complementary failures",
        },
    )
    parents = [
        StrategyTrajectory(
            trajectory_id="p1",
            direction_id=0,
            round_idx=0,
            phase=RoundPhase.ORIGINAL,
            hypothesis="parent one",
            backtest_metrics={"rankic": -0.1, "long_netir": 0.2},
            submission_checks={"a": {"check_passed": False, "failed_metrics": ["rankic"]}},
        ),
        StrategyTrajectory(
            trajectory_id="p2",
            direction_id=1,
            round_idx=0,
            phase=RoundPhase.ORIGINAL,
            hypothesis="parent two",
            backtest_metrics={"rankic": 0.03, "long_netir": None},
            submission_checks={"b": {"check_passed": True, "rankic_passed": True}},
        ),
    ]

    suffix = op.generate_crossover_prompt_suffix(parents)

    assert "primary" + "_score" not in suffix
    assert "Metrics Snapshot" in suffix
    assert "rankic=-0.1000" in suffix
    assert "long_netir=0.2000" in suffix
