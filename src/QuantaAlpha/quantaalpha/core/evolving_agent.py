from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
import os
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

if TYPE_CHECKING:
    from quantaalpha.core.evaluation import Evaluator
    from quantaalpha.core.evolving_framework import EvolvableSubjects

from quantaalpha.core.evaluation import Feedback
from quantaalpha.core.evolving_framework import EvolvingStrategy, EvoStep
from quantaalpha.log import logger


def _enforce_domain_contract_on_feedback(
    evo: EvolvableSubjects,
    feedback: Feedback | None,
) -> None:
    """Apply domain-coverage and (when applicable) joint pv/minutes
    contracts to each sub-task's feedback in place.

    Runs for every active-domain mode (pv-only, min-only, joint).
    Module text is resolved via the same helper that the runner uses
    (multi-filename fallback) so the contract sees the actual emitted
    module, not just `factor.py`. Violations force final_decision=False
    and surface a concrete repair list in execution/final feedback so
    the next coder iteration can self-repair within max_loop.
    Errors inside the validators never propagate.
    """
    if not isinstance(feedback, list):
        return
    try:
        from quantaalpha.factors.tq_runner import (
            _extract_workspace_factor_module,
            _validate_module_domain_coverage,
        )
        from quantaalpha.factors.combined_domain_contract import (
            is_joint_pv_minutes_run,
            validate_joint_pv_minutes_contract,
        )
        from quantaalpha.factors.data_domains import resolve_factor_domains
    except Exception:
        return

    try:
        active_domains = resolve_factor_domains()
    except Exception:
        return
    if not active_domains:
        return

    is_joint = False
    try:
        is_joint = bool(is_joint_pv_minutes_run(active_domains))
    except Exception:
        is_joint = False

    workspaces = list(getattr(evo, 'sub_workspace_list', []) or [])
    for idx, item in enumerate(feedback):
        if item is None or idx >= len(workspaces):
            continue
        workspace = workspaces[idx]
        code_dict = getattr(workspace, 'code_dict', None) or {}
        if not isinstance(code_dict, dict):
            continue
        try:
            module_text, _module_level = _extract_workspace_factor_module(code_dict)
        except Exception:
            module_text = ''
        if not module_text:
            continue
        if 'STRUCTURED_RENDER_FAILURE_MESSAGE' in str(module_text):
            # Render-failure placeholder; proposal stage owns repair.
            continue

        errors: list[str] = []

        # Single-domain coverage check (applies to pv-only / min-only / joint).
        try:
            module_globals: dict[str, object] = {}
            try:
                exec(compile(module_text, '<contract-check>', 'exec'), module_globals)
            except Exception:
                module_globals = {}
            if module_globals:
                _validate_module_domain_coverage(
                    module_globals, active_domains, module_text=module_text
                )
        except ValueError as exc:
            errors.append(str(exc))
        except Exception:
            # Static load failures are not treated as contract violations;
            # the runner will surface real runtime errors later.
            pass

        # Joint pv/minutes structural contract (only when joint mode).
        if is_joint:
            try:
                joint_errors = validate_joint_pv_minutes_contract(
                    module_text, active_domains=active_domains
                )
                if joint_errors:
                    errors.extend(joint_errors)
            except Exception as exc:
                logger.warning(f'joint pv/minutes contract validator raised: {exc}')

        if not errors:
            continue

        violation_text = 'Domain contract violations:\n- ' + '\n- '.join(errors)
        try:
            if getattr(item, 'final_decision', None) is True or getattr(item, 'final_decision', None) is None:
                item.final_decision = False
            existing_final = str(getattr(item, 'final_feedback', '') or '').strip()
            item.final_feedback = (
                f'{existing_final}\n{violation_text}' if existing_final else violation_text
            ).strip()
            existing_exec = str(getattr(item, 'execution_feedback', '') or '').strip()
            item.execution_feedback = (
                f'{existing_exec}\n{violation_text}' if existing_exec else violation_text
            ).strip()
        except Exception as exc:
            logger.warning(f'failed to inject contract feedback for sub_task[{idx}]: {exc}')


# Backwards-compatible alias for any external callers / tests.
_enforce_joint_contract_on_feedback = _enforce_domain_contract_on_feedback


def _disable_tqdm() -> bool:
    raw = os.environ.get("QUANTAALPHA_DISABLE_TQDM")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    raw = os.environ.get("QUANTAALPHA_QUIET_CONSOLE")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _extract_feedback_final_decisions(feedback: Feedback | None) -> list[bool | None] | None:
    if not isinstance(feedback, list):
        return None
    decisions: list[bool | None] = []
    for item in feedback:
        if item is None:
            decisions.append(None)
            continue
        if not hasattr(item, "final_decision"):
            return None
        decisions.append(getattr(item, "final_decision"))
    return decisions


def _log_final_round_decisions(feedback: Feedback | None) -> None:
    decisions = _extract_feedback_final_decisions(feedback)
    if decisions is None:
        return
    true_count = decisions.count(True)
    total = len(decisions)
    logger.info(f"Final round decisions (last active subset): {decisions} ({true_count}/{total} passed)")


def _log_cumulative_decisions(successful_pairs: list[tuple[int, Any, Any]], original_total: int) -> None:
    decisions = [False] * original_total
    for original_idx, *_ in successful_pairs:
        if 0 <= original_idx < original_total:
            decisions[original_idx] = True
    true_count = decisions.count(True)
    logger.info(
        f"Cumulative coder decisions over original sub_tasks: {decisions} "
        f"({true_count}/{original_total} passed)"
    )


def _first_feedback_line(feedback_item: Any) -> str:
    for attr_name in ("final_feedback", "execution_feedback", "value_feedback", "code_feedback", "shape_feedback"):
        text = str(getattr(feedback_item, attr_name, "") or "").strip()
        if not text:
            continue
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if first_line:
            return first_line[:300]
    return "coder final_decision passed"


def _mark_task_successful_for_runner(task: Any, feedback_item: Any) -> None:
    try:
        task.factor_implementation = True
        summary = _first_feedback_line(feedback_item)
        existing = str(getattr(task, "alpha_evaluation_feedback", "") or "").strip()
        marker = f"[coder-final-decision] {summary}"
        task.alpha_evaluation_feedback = f"{existing}\n{marker}".strip() if existing else marker
    except Exception:
        pass


def _slice_evolvable_subjects(evo: EvolvableSubjects, indices: list[int]) -> EvolvableSubjects:
    sliced = copy(evo)
    sliced.sub_tasks = [evo.sub_tasks[idx] for idx in indices]
    sliced.sub_workspace_list = [evo.sub_workspace_list[idx] for idx in indices]
    if hasattr(evo, "sub_gt_implementations"):
        gt_impls = getattr(evo, "sub_gt_implementations")
        sliced.sub_gt_implementations = None if gt_impls is None else [gt_impls[idx] for idx in indices]
    return sliced


def _extract_passed_and_retry_indices(feedback: Feedback | None) -> tuple[list[int], list[int]] | None:
    decisions = _extract_feedback_final_decisions(feedback)
    if decisions is None:
        return None
    passed_indices = [idx for idx, decision in enumerate(decisions) if decision is True]
    retry_indices = [idx for idx, decision in enumerate(decisions) if decision is not True]
    return passed_indices, retry_indices


class EvoAgent(ABC):
    def __init__(self, max_loop: int, evolving_strategy: EvolvingStrategy) -> None:
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy

    @abstractmethod
    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        filter_final_evo: bool = False,
    ) -> EvolvableSubjects: ...

    @abstractmethod
    def filter_evolvable_subjects_by_feedback(
        self,
        evo: EvolvableSubjects,
        feedback: Feedback | None,
    ) -> EvolvableSubjects: ...


class RAGEvoAgent(EvoAgent):
    def __init__(
        self,
        max_loop: int,
        evolving_strategy: EvolvingStrategy,
        rag: Any,
        with_knowledge: bool = False,
        with_feedback: bool = True,
        knowledge_self_gen: bool = False,
    ) -> None:
        super().__init__(max_loop, evolving_strategy)
        self.rag = rag
        self.evolving_trace: list[EvoStep] = []
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen

    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        filter_final_evo: bool = False,
    ) -> EvolvableSubjects:
        """Multi-step evolution: knowledge self-evolve, RAG query, evolve, pack, evaluate, update trace."""
        active_evo = evo
        active_original_indices = list(range(len(getattr(evo, "sub_tasks", []) or [])))
        successful_pairs: list[tuple[int, Any, Any, Any]] = []
        decision_filtering_applied = False

        for loop_i in tqdm(range(self.max_loop), "Debugging", disable=_disable_tqdm()):
            if not active_original_indices:
                logger.info("All sub_tasks passed final_decision before max_loop; stop coder retries early.")
                break
            if loop_i > 0:
                logger.info(
                    f"Retrying only failed sub_tasks: {len(active_original_indices)} remaining "
                    f"(attempt {loop_i + 1}/{self.max_loop})."
                )

            if self.knowledge_self_gen and self.rag is not None:
                self.rag.generate_knowledge(self.evolving_trace)

            queried_knowledge = None
            if self.with_knowledge and self.rag is not None:
                queried_knowledge = self.rag.query(active_evo, self.evolving_trace)

            active_evo = self.evolving_strategy.evolve(
                evo=active_evo,
                evolving_trace=self.evolving_trace,
                queried_knowledge=queried_knowledge,
            )

            logger.log_object(active_evo.sub_workspace_list, tag="evolving code")  # type: ignore[attr-defined]
            if loop_i == 0:
                for sw in active_evo.sub_workspace_list:  # type: ignore[attr-defined]
                    logger.info(f"evolving code workspace: {sw}")

            es = EvoStep(active_evo, queried_knowledge)

            if self.with_feedback:
                es.feedback = (
                    eva
                    if isinstance(eva, Feedback)
                    else eva.evaluate(active_evo, queried_knowledge=queried_knowledge)  # type: ignore[arg-type, call-arg]
                )
                # Enforce the combined pv/minutes contract before final_decision
                # is consumed for retry/skip routing. Violations override
                # final_decision=False and are surfaced in execution/final
                # feedback so the next coder iteration can self-repair.
                _enforce_domain_contract_on_feedback(active_evo, es.feedback)
                logger.log_object(es.feedback, tag="evolving feedback")

            self.evolving_trace.append(es)

            if not self.with_feedback:
                continue

            index_groups = _extract_passed_and_retry_indices(es.feedback)
            if index_groups is None:
                continue
            decision_filtering_applied = True

            passed_indices, retry_indices = index_groups
            if passed_indices:
                for local_idx in passed_indices:
                    feedback_item = (
                        es.feedback[local_idx]
                        if isinstance(es.feedback, list) and local_idx < len(es.feedback)
                        else None
                    )
                    _mark_task_successful_for_runner(active_evo.sub_tasks[local_idx], feedback_item)
                    successful_pairs.append(
                        (
                            active_original_indices[local_idx],
                            active_evo.sub_tasks[local_idx],
                            active_evo.sub_workspace_list[local_idx],
                            feedback_item,
                        )
                    )

            if not retry_indices:
                logger.info("Current coder loop resolved all remaining sub_tasks; stop retries early.")
                active_original_indices = []
                break

            active_evo = self.filter_evolvable_subjects_by_feedback(active_evo, es.feedback)
            active_evo = _slice_evolvable_subjects(active_evo, retry_indices)
            active_original_indices = [active_original_indices[idx] for idx in retry_indices]

        if self.with_feedback and self.evolving_trace:
            _log_final_round_decisions(self.evolving_trace[-1].feedback)
            if decision_filtering_applied:
                _log_cumulative_decisions(successful_pairs, len(getattr(evo, "sub_tasks", []) or []))

        if not decision_filtering_applied:
            if self.with_feedback and filter_final_evo and self.evolving_trace:
                active_evo = self.filter_evolvable_subjects_by_feedback(active_evo, self.evolving_trace[-1].feedback)
            return active_evo

        if not successful_pairs:
            if self.with_feedback and filter_final_evo and self.evolving_trace:
                active_evo = self.filter_evolvable_subjects_by_feedback(active_evo, self.evolving_trace[-1].feedback)
            return _slice_evolvable_subjects(active_evo, [])

        successful_pairs.sort(key=lambda item: item[0])
        final_evo = _slice_evolvable_subjects(evo, [])
        final_evo.sub_tasks = [task for _, task, _, _ in successful_pairs]
        final_evo.sub_workspace_list = [workspace for _, _, workspace, _ in successful_pairs]
        final_evo.implementation_selection_applied = True
        if hasattr(evo, "sub_gt_implementations"):
            gt_impls = getattr(evo, "sub_gt_implementations")
            if gt_impls is None:
                final_evo.sub_gt_implementations = None
            else:
                final_evo.sub_gt_implementations = [gt_impls[idx] for idx, _, _, _ in successful_pairs]
        return final_evo

    def filter_evolvable_subjects_by_feedback(
        self,
        evo: EvolvableSubjects,
        feedback: Feedback | None,
    ) -> EvolvableSubjects:
        # Implementation of filter_evolvable_subjects_by_feedback method
        pass
