from quantaalpha.coder.costeer import CoSTEER
from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem
from quantaalpha.core.experiment import Experiment
from quantaalpha.factors.coder.factor import FactorTask


class _FakeAgent:
    def __init__(self, *args, **kwargs):
        pass

    def multistep_evolve(self, evo, evaluator, filter_final_evo=False):
        result = EvolvingItem(sub_tasks=[evo.sub_tasks[0], evo.sub_tasks[2]])
        result.sub_workspace_list = [object(), object()]
        result.implementation_selection_applied = True
        return result


def test_costeer_develop_preserves_success_subset_even_when_task_flags_are_stale(monkeypatch):
    import quantaalpha.coder.costeer as costeer_module

    monkeypatch.setattr(costeer_module, "FilterFailedRAGEvoAgent", _FakeAgent)
    costeer = object.__new__(CoSTEER)
    costeer.max_loop = 1
    costeer.evolving_strategy = object()
    costeer.rag = None
    costeer.with_knowledge = False
    costeer.with_feedback = True
    costeer.knowledge_self_gen = False
    costeer.evaluator = object()
    costeer.filter_final_evo = True
    costeer.new_knowledge_base_path = None

    tasks = [
        FactorTask("kept_a", "demo", "demo", factor_expression="ts_mean(returns, 15)"),
        FactorTask("dropped", "demo", "demo", factor_expression="bad_expr"),
        FactorTask("kept_b", "demo", "demo", factor_expression="ts_mean(turnovers, 15)"),
    ]
    exp = Experiment(sub_tasks=tasks)

    result = costeer.develop(exp)

    assert [task.factor_name for task in result.sub_tasks] == ["kept_a", "kept_b"]
    assert [task.factor_implementation for task in result.sub_tasks] == [True, True]
    assert len(result.sub_workspace_list) == 2
    assert result.implementation_selection_applied is True
