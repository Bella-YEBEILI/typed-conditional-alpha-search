from types import SimpleNamespace

from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem
from quantaalpha.core.evolving_agent import RAGEvoAgent
from quantaalpha.factors.coder.factor import FactorTask


class _OnePassStrategy:
    def evolve(self, *, evo, evolving_trace=None, queried_knowledge=None):
        evo.sub_workspace_list = [object(), object()]
        return evo


class _StaticEvaluator:
    def evaluate(self, evo, queried_knowledge=None):
        return [
            SimpleNamespace(
                final_decision=True,
                final_feedback="deterministic checks passed",
                execution_feedback="Execution succeeded without error.",
            ),
            SimpleNamespace(
                final_decision=False,
                final_feedback="hard failure",
                execution_feedback="RuntimeError: bad expression",
            ),
        ]


class _Agent(RAGEvoAgent):
    def filter_evolvable_subjects_by_feedback(self, evo, feedback):
        return evo


def test_multistep_evolve_marks_successful_subset_for_runner_handoff():
    task_ok = FactorTask("ok_factor", "demo", "demo", factor_expression="ts_mean(returns, 15)")
    task_bad = FactorTask("bad_factor", "demo", "demo", factor_expression="bad_expr")
    evo = EvolvingItem(sub_tasks=[task_ok, task_bad])
    evo.factor_generation_requested_count = 3
    evo.factor_generation_proposed_count = 3

    agent = _Agent(
        max_loop=1,
        evolving_strategy=_OnePassStrategy(),
        rag=None,
        with_knowledge=False,
        with_feedback=True,
        knowledge_self_gen=False,
    )

    final_evo = agent.multistep_evolve(evo, _StaticEvaluator(), filter_final_evo=True)

    assert [task.factor_name for task in final_evo.sub_tasks] == ["ok_factor"]
    assert final_evo.sub_workspace_list == [evo.sub_workspace_list[0]]
    assert final_evo.sub_tasks[0].factor_implementation is True
    assert "deterministic checks passed" in final_evo.sub_tasks[0].alpha_evaluation_feedback
    assert final_evo.factor_generation_requested_count == 3
    assert final_evo.factor_generation_proposed_count == 3
