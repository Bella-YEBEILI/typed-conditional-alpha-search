from abc import abstractmethod
from typing import List

from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem
from quantaalpha.core.conf import RD_AGENT_SETTINGS
from quantaalpha.core.evaluation import Evaluator, Feedback
from quantaalpha.core.evolving_framework import QueriedKnowledge
from quantaalpha.core.experiment import Workspace
from quantaalpha.core.scenario import Task
from quantaalpha.core.utils import multiprocessing_wrapper
from quantaalpha.log import logger


class CoSTEERSingleFeedback(Feedback):
    """This class is a base class for all code generator feedback to single implementation"""

    def __init__(
        self,
        execution_feedback: str = None,
        shape_feedback: str = None,
        code_feedback: str = None,
        value_feedback: str = None,
        final_decision: bool = None,
        final_feedback: str = None,
        value_generated_flag: bool = None,
        final_decision_based_on_gt: bool = None,
    ) -> None:
        self.execution_feedback = execution_feedback
        self.shape_feedback = shape_feedback
        self.code_feedback = code_feedback
        self.value_feedback = value_feedback
        self.final_decision = final_decision
        self.final_feedback = final_feedback
        self.value_generated_flag = value_generated_flag
        self.final_decision_based_on_gt = final_decision_based_on_gt

    def __str__(self) -> str:
        return f"""------------------Execution Feedback------------------
{self.execution_feedback if self.execution_feedback is not None else 'No execution feedback'}
------------------Shape Feedback------------------
{self.shape_feedback if self.shape_feedback is not None else 'No shape feedback'}
------------------Code Feedback------------------
{self.code_feedback if self.code_feedback is not None else 'No code feedback'}
------------------Value Feedback------------------
{self.value_feedback if self.value_feedback is not None else 'No value feedback'}
------------------Final Feedback------------------
{self.final_feedback if self.final_feedback is not None else 'No final feedback'}
------------------Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""


class CoSTEERMultiFeedback(
    Feedback,
    List[CoSTEERSingleFeedback],
):
    """Feedback contains a list, each element is the corresponding feedback for each factor implementation."""


class CoSTEEREvaluator(Evaluator):
    # TODO:
    # I think we should have unified interface for all evaluates, for examples.
    # So we should adjust the interface of other factors
    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        raise NotImplementedError("Please implement the `evaluator` method")


class CoSTEERMultiEvaluator(Evaluator):
    def __init__(self, single_evaluator: CoSTEEREvaluator, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.single_evaluator = single_evaluator

    def evaluate(
        self,
        evo: EvolvingItem,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERMultiFeedback:
        multi_implementation_feedback = multiprocessing_wrapper(
            [
                (
                    self.single_evaluator.evaluate,
                    (
                        evo.sub_tasks[index],
                        evo.sub_workspace_list[index],
                        evo.sub_gt_implementations[index] if evo.sub_gt_implementations is not None else None,
                        queried_knowledge,
                    ),
                )
                for index in range(len(evo.sub_tasks))
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        final_decision = [
            None if single_feedback is None else single_feedback.final_decision
            for single_feedback in multi_implementation_feedback
        ]
        true_count = final_decision.count(True)
        total = len(final_decision)
        if true_count < total:
            logger.info(f"Final decisions: {final_decision} ({true_count}/{total} passed)")
            for index, single_feedback in enumerate(multi_implementation_feedback):
                if single_feedback is None or final_decision[index] is True:
                    continue
                factor_name = getattr(evo.sub_tasks[index], "factor_name", getattr(evo.sub_tasks[index], "name", f"task_{index}"))
                failure_summary = _summarize_feedback_failure(single_feedback)
                try:
                    evo.sub_tasks[index].alpha_evaluation_feedback = (
                        (getattr(evo.sub_tasks[index], "alpha_evaluation_feedback", "") or "")
                        + "\n[coder-final-decision] "
                        + failure_summary
                    )
                except Exception:
                    pass
                logger.warning(f"Final decision failed for {factor_name}: {failure_summary}")

        for index in range(len(evo.sub_tasks)):
            if final_decision[index]:
                evo.sub_tasks[index].factor_implementation = True

        return multi_implementation_feedback


def _summarize_feedback_failure(single_feedback: CoSTEERSingleFeedback) -> str:
    actionable = _first_actionable_failure_line(getattr(single_feedback, "execution_feedback", None))
    if actionable:
        return actionable[:300]
    for attr_name in ("final_feedback", "execution_feedback", "value_feedback", "code_feedback", "shape_feedback"):
        value = getattr(single_feedback, attr_name, None)
        text = str(value or "").strip()
        if not text:
            continue
        first_line = _first_actionable_failure_line(text)
        if first_line:
            return first_line[:300]
    return "no failure summary available"


def _first_actionable_failure_line(text: str | None) -> str:
    skip_exact = {
        "execution succeeded without error.",
        "expected output file found.",
        "ast regularization check passed",
    }
    skip_prefixes = (
        "deterministic final decision failed because",
        "execution feedback summary: execution succeeded without error",
    )
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = " ".join(line.lower().split())
        if normalized in skip_exact or any(normalized.startswith(prefix) for prefix in skip_prefixes):
            continue
        return line
    return ""
