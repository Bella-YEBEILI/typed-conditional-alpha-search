import ast
import re
from pathlib import Path

from quantaalpha.coder.costeer.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from quantaalpha.factors.coder.eva_utils import (
    FactorCodeEvaluator,
    FactorFinalDecisionEvaluator,
    FactorValueEvaluator,
)
from quantaalpha.factors.coder.factor import FactorTask
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.combined_domain_contract import (
    is_joint_pv_minutes_run,
    validate_joint_pv_minutes_contract,
)
from quantaalpha.factors.data_domains import resolve_factor_domains
from quantaalpha.core.evolving_framework import QueriedKnowledge
from quantaalpha.core.experiment import Workspace
from quantaalpha.factors.regulator.factor_regulator import FactorRegulator
from quantaalpha.log import logger

FactorSingleFeedback = CoSTEERSingleFeedback
FactorMultiFeedback = CoSTEERMultiFeedback


def _normalize_feedback_text(text: str | None) -> str:
    return " ".join(str(text or "").lower().split())


def _execution_feedback_has_hard_failure(feedback: str | None) -> bool:
    normalized = _normalize_feedback_text(feedback)
    if "execution succeeded without error." not in normalized and "execution succeeded without error" not in normalized:
        return True
    if "recovered factor value via deterministic tq bridge fallback" in normalized:
        normalized = normalized.split("initial direct execution feedback:", 1)[0]
    failure_markers = (
        "ast regularization check failed",
        "expected output file not found",
        "execution timeout error",
        "failed to parse expression",
        "error found when reading hdf file",
        "traceback",
        "nameerror",
        "keyerror",
        "filenotfounderror",
    )
    return any(marker in normalized for marker in failure_markers)


def _value_feedback_has_blocking_signal(feedback: str | None) -> bool:
    normalized = _normalize_feedback_text(feedback)
    blocking_markers = (
        "not sufficiently high correlated",
        "different index",
        "different missing values",
        "source dataframe is none",
        "does not have a datetime index",
        "datetime index but it is not in the correct format",
    )
    if any(marker in normalized for marker in blocking_markers):
        return True
    if "infinite values" in normalized and "does not have any infinite values" not in normalized:
        return True
    return False


def _first_nonempty_line(text: str | None, limit: int = 500) -> str:
    for line in str(text or "").splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""


def _first_actionable_failure_line(text: str | None, limit: int = 500) -> str:
    skip_exact = {
        "execution succeeded without error.",
        "expected output file found.",
        "ast regularization check passed",
    }
    skip_prefixes = (
        "execution feedback summary: execution succeeded without error",
    )
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        normalized = " ".join(line.lower().split())
        if normalized in skip_exact or any(normalized.startswith(prefix) for prefix in skip_prefixes):
            continue
        return line[:limit]
    return _first_nonempty_line(text, limit=limit)


def _extract_module_string_assignment(module_text: str, variable_name: str) -> str | None:
    try:
        tree = ast.parse(module_text or "")
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value.strip()
    return None


class FactorEvaluatorForCoder(CoSTEEREvaluator):
    """This class is the v1 version of evaluator for a single factor implementation.
    It calls several evaluators in share modules to evaluate the factor implementation.
    Now includes AST-based regularization checks for factor quality.
    """

    def __init__(self, *args, factor_zoo_path: str = None, duplication_threshold: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value_evaluator = FactorValueEvaluator(self.scen)
        self.code_evaluator = FactorCodeEvaluator(self.scen)
        self.final_decision_evaluator = FactorFinalDecisionEvaluator(self.scen)
        # Initialize FactorRegulator for AST-based regularization checks
        # Use config settings if not explicitly provided
        factor_zoo_path = factor_zoo_path or FACTOR_COSTEER_SETTINGS.factor_zoo_path
        duplication_threshold = duplication_threshold if duplication_threshold is not None else FACTOR_COSTEER_SETTINGS.duplication_threshold
        symbol_length_threshold = getattr(FACTOR_COSTEER_SETTINGS, 'symbol_length_threshold', 400)
        base_features_threshold = getattr(FACTOR_COSTEER_SETTINGS, 'base_features_threshold', 6)
        self.factor_regulator = FactorRegulator(
            factor_zoo_path=factor_zoo_path,
            duplication_threshold=duplication_threshold,
            symbol_length_threshold=symbol_length_threshold,
            base_features_threshold=base_features_threshold
        )
    
    def extract_expr(self, code_str: str) -> str:
        """Extract expr from code string (expr = \"...\" or expr = '...')."""
        for variable_name in ("expr", "ORIGINAL_FACTOR_EXPRESSION", "FACTOR_EXPRESSION"):
            assigned_expr = _extract_module_string_assignment(code_str, variable_name)
            if assigned_expr:
                return assigned_expr

        pattern = r"expr\s*=\s*(?P<quote>'''|\"\"\"|'|\")(?P<expr>.*?)(?P=quote)"
        match = re.search(pattern, code_str or "", flags=re.DOTALL)
        if match:
            return match.group("expr").strip()
        else:
            return ""

    def check_combined_domain_contract(self, implementation: Workspace) -> tuple[bool, str]:
        """Pre-flight combined pv/minutes contract validation. Returns (ok, feedback).

        Runs the same validator the TQ runner uses, but inside the coder loop so
        violations turn into final_decision=False feedback and the existing
        multistep_evolve retry budget (max_loop=3) lets the LLM self-repair.
        """
        try:
            active_domains = resolve_factor_domains()
        except Exception as exc:
            logger.warning(f"Could not resolve active domains for contract check: {exc}")
            return True, ""
        if not is_joint_pv_minutes_run(active_domains):
            return True, ""
        try:
            code_dict = getattr(implementation, "code_dict", {}) or {}
            module_text = ""
            if isinstance(code_dict, dict):
                module_text = code_dict.get("factor.py") or code_dict.get("model.py") or ""
                if not module_text:
                    for name, content in code_dict.items():
                        if str(name).endswith(".py") and content:
                            module_text = content
                            break
            if not module_text and hasattr(implementation, "code"):
                module_text = getattr(implementation, "code") or ""
            if not module_text:
                return (
                    False,
                    "Combined pv/minutes Contract Validation Failed:\n"
                    "  1. Joint pv/minutes runs require generated factor.py module code, "
                    "but the implementation workspace did not contain any Python module text.\n"
                    "Regenerate the factor as a minute-level module with META['level']='minutes', "
                    "prepare_minute_datas(), calc_factor(data_ctx, minute_ctx), at least one raw "
                    "minute input, and at least one declared daily pv field.",
                )
            if "STRUCTURED_RENDER_FAILURE_MESSAGE" in module_text and "STRUCTURED_RENDER_FAILURE_MESSAGE = None" not in module_text:
                return (
                    False,
                    "Combined pv/minutes Contract Validation Failed:\n"
                    "  1. Structured minute module rendering failed and produced a placeholder "
                    "failure module instead of executable factor code.\n"
                    "Use this feedback to regenerate the module from the original expression rather "
                    "than passing the placeholder to TQ runner.",
                )
            errors = validate_joint_pv_minutes_contract(module_text, active_domains=active_domains)
            if not errors:
                return True, ""
            numbered = "\n".join(f"  {idx + 1}. {msg}" for idx, msg in enumerate(errors))
            feedback = (
                "Combined pv/minutes Contract Validation Failed:\n"
                f"{numbered}\n"
                "Fix exactly these points and resubmit. Refer to the joint contract checklist "
                "and the minimal template in the system prompt."
            )
            return False, feedback
        except Exception as exc:
            logger.warning(f"Combined pv/minutes contract check raised: {exc}")
            return True, ""
    
    def check_ast_regularization(self, implementation: Workspace) -> tuple[bool, str]:
        """Check if factor expression meets AST regularization. Returns (ok, feedback)."""
        try:
            if hasattr(implementation, 'code_dict') and 'factor.py' in implementation.code_dict:
                code = implementation.code_dict['factor.py']
            elif hasattr(implementation, 'code'):
                code = implementation.code
            else:
                return True, "AST Regularization Check Skipped: Cannot extract code from implementation"
            
            expr = self.extract_expr(code)
            
            if not expr:
                return True, ""

            daily_render_failure = _extract_module_string_assignment(code, "DAILY_RENDER_FAILURE_MESSAGE")
            if daily_render_failure and expr == "raise_expression_render_failure()":
                return False, f"Daily Expression Render Failed: {daily_render_failure}"
            
            if not self.factor_regulator.is_parsable(expr):
                return False, f"AST Regularization Check Failed: Expression cannot be parsed: {expr}"
            
            success, eval_dict = self.factor_regulator.evaluate(expr)
            if not success:
                return False, f"AST Regularization Check Failed: Failed to evaluate expression: {expr}"
            
            is_acceptable = self.factor_regulator.is_expression_acceptable(eval_dict)
            
            if not is_acceptable:
                feedback_parts = []
                unsupported_functions = eval_dict.get('unsupported_functions', [])
                unsupported_fields = eval_dict.get('unsupported_fields', [])
                operators_allowed = bool(eval_dict.get('operators_allowed', True))
                fields_allowed = bool(eval_dict.get('fields_allowed', True))

                if unsupported_functions:
                    normalized_functions = [
                        str(item).replace("<semantic:", "").rstrip(">")
                        for item in unsupported_functions
                    ]
                    feedback_parts.append(
                        "Operator Validation Failed: " + "; ".join(normalized_functions)
                    )

                if unsupported_fields:
                    feedback_parts.append(
                        "Field Validation Failed: " + ", ".join(str(item) for item in unsupported_fields)
                    )
                
                # Novelty (only when factor zoo exists)
                dup_size = eval_dict.get('duplicated_subtree_size', 0)
                dup_threshold = self.factor_regulator.duplication_threshold
                has_factor_zoo = self.factor_regulator.factor_zoo_path is not None and len(self.factor_regulator.alphazoo) > 0
                
                if has_factor_zoo and dup_size > dup_threshold:
                    matched_alpha = eval_dict.get('matched_alpha', 'Unknown')
                    duplicated_subtree = eval_dict.get('duplicated_subtree', '')
                    feedback_parts.append(
                        f"Novelty Check Failed: Duplicated subtree size ({dup_size}) exceeds threshold ({dup_threshold}). "
                        f"Matched with: {matched_alpha}. Duplicated subtree: {duplicated_subtree}"
                    )
                elif not has_factor_zoo:
                    feedback_parts.append(
                        f"Note: Novelty check skipped (no factor zoo provided). "
                        f"Duplicated subtree size: {dup_size}"
                    )
                
                # Free args ratio (soft complexity diagnostic)
                num_free_args = eval_dict.get('num_free_args', 0)
                num_all_nodes = eval_dict.get('num_all_nodes', 0)
                if num_all_nodes > 0:
                    free_args_ratio = num_free_args / num_all_nodes
                    if free_args_ratio >= 0.5:
                        feedback_parts.append(
                            f"Free Arguments Ratio Warning: Free arguments ratio ({free_args_ratio:.2%}) >= 50%. "
                            f"Number of free args: {num_free_args}, Total nodes: {num_all_nodes}. "
                            f"This may indicate over-parameterization; keep it if the mechanism is intentional and executable."
                        )
                
                # Unique vars ratio (soft complexity diagnostic)
                num_unique_vars = eval_dict.get('num_unique_vars', 0)
                if num_all_nodes > 0:
                    unique_vars_ratio = num_unique_vars / num_all_nodes
                    if unique_vars_ratio >= 0.5:
                        feedback_parts.append(
                            f"Unique Variables Ratio Warning: Unique variables ratio ({unique_vars_ratio:.2%}) >= 50%. "
                            f"Number of unique vars: {num_unique_vars}, Total nodes: {num_all_nodes}. "
                            f"This may indicate a narrow construction; keep it if the field choice is deliberate."
                        )
                
                # Symbol length (SL) (soft complexity diagnostic)
                symbol_length = eval_dict.get('symbol_length', 0)
                symbol_length_threshold = self.factor_regulator.symbol_length_threshold
                if symbol_length > symbol_length_threshold:
                    feedback_parts.append(
                        f"Symbol Length (SL) Warning: Symbol length ({symbol_length}) exceeds soft threshold ({symbol_length_threshold}). "
                        f"This is a complexity risk diagnostic only; do not simplify away valid structure solely to satisfy SL."
                    )
                
                # Base features (ER) (soft complexity diagnostic)
                num_base_features = eval_dict.get('num_base_features', 0)
                base_features_threshold = self.factor_regulator.base_features_threshold
                if num_base_features > base_features_threshold:
                    feedback_parts.append(
                        f"Base Features Count (ER) Warning: Number of base features ({num_base_features}) exceeds soft threshold ({base_features_threshold}). "
                        f"The factor uses too many raw features (e.g., $close, $open, $high, $low, $volume), "
                        f"which may indicate over-engineering; keep it if the cross-field mechanism is intentional."
                        )
                
                # Only return False when there are real failures
                has_failures = (
                    (not operators_allowed) or
                    (not fields_allowed) or
                    (has_factor_zoo and dup_size > dup_threshold)
                )
                
                if has_failures:
                    feedback = "AST Regularization Check Failed:\n" + "\n".join(feedback_parts)
                    return False, feedback
                else:
                    return True, "\n".join(feedback_parts)
            else:
                return True, "AST Regularization Check Passed"
                
        except Exception as e:
            logger.warning(f"AST regularization check failed with exception: {e}")
            return True, f"AST Regularization Check Skipped: {str(e)}"

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return None

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return FactorSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                value_generated_flag=False,
                code_feedback="This task has failed too many times, skip code evaluation.",
                value_feedback="This task has failed too many times, skip value evaluation.",
                final_decision=False,
                final_feedback="This task has failed too many times, skip final decision evaluation.",
                final_decision_based_on_gt=False,
            )
        else:
            factor_feedback = FactorSingleFeedback()

            # 0. AST Regularization Check (before execution)
            ast_check_passed, ast_feedback = self.check_ast_regularization(implementation)
            if not ast_check_passed:
                # If AST regularization check fails, mark as failed and return early
                factor_feedback.execution_feedback = f"AST Regularization Check Failed: {ast_feedback}"
                factor_feedback.value_generated_flag = False
                factor_feedback.value_feedback = "AST Regularization Check Failed, skip value evaluation."
                factor_feedback.code_feedback = ast_feedback
                factor_feedback.final_decision = False
                factor_feedback.final_feedback = f"Factor rejected due to AST regularization violations:\n{ast_feedback}"
                factor_feedback.final_decision_based_on_gt = False
                return factor_feedback

            # 0b. Combined pv/minutes contract pre-flight (joint runs only)
            contract_passed, contract_feedback = self.check_combined_domain_contract(implementation)
            if not contract_passed:
                factor_feedback.execution_feedback = contract_feedback
                factor_feedback.value_generated_flag = False
                factor_feedback.value_feedback = "Combined pv/minutes contract check failed, skip value evaluation."
                factor_feedback.code_feedback = contract_feedback
                factor_feedback.final_decision = False
                factor_feedback.final_feedback = (
                    "Factor rejected by combined pv/minutes contract pre-flight; "
                    "use the violation list above to repair the module on the next attempt."
                )
                factor_feedback.final_decision_based_on_gt = False
                return factor_feedback

            # 1. Get factor execution feedback to generated implementation and remove the long list of numbers in execution feedback
            (
                execution_feedback,
                gen_df,
            ) = implementation.execute()

            execution_feedback = re.sub(r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback)
            factor_feedback.execution_feedback = "\n".join(
                [line for line in execution_feedback.split("\n") if "warning" not in line.lower()]
            )
            
            # Add AST regularization check result to execution feedback if passed
            if ast_feedback and ast_feedback != "":
                factor_feedback.execution_feedback = f"{ast_feedback}\n\n{factor_feedback.execution_feedback}"

            # 2. Get factor value feedback
            if gen_df is None:
                factor_feedback.value_feedback = "No factor value generated, skip value evaluation."
                factor_feedback.value_generated_flag = False
                decision_from_value_check = None
            else:
                factor_feedback.value_generated_flag = True
                (
                    factor_feedback.value_feedback,
                    decision_from_value_check,
                ) = self.value_evaluator.evaluate(
                    implementation=implementation, gt_implementation=gt_implementation, version=target_task.version
                )

            factor_feedback.final_decision_based_on_gt = gt_implementation is not None
            # import pdb; pdb.set_trace()
            if decision_from_value_check is not None and decision_from_value_check is True:
                # To avoid confusion, when same_value_or_high_correlation is True, we do not need code feedback
                factor_feedback.code_feedback = "Final decision is True and there are no code critics."
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation passed, skip final decision evaluation."
            elif decision_from_value_check is not None and decision_from_value_check is False:
                factor_feedback.code_feedback = (
                    "Skipped model code review because deterministic value evaluation already failed."
                )
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation failed, skip final decision evaluation."
            else:
                factor_feedback.code_feedback = (
                    "Skipped model code review; using deterministic execution and value checks only."
                )
                if _execution_feedback_has_hard_failure(factor_feedback.execution_feedback):
                    factor_feedback.final_decision = False
                    execution_summary = _first_actionable_failure_line(factor_feedback.execution_feedback)
                    factor_feedback.final_feedback = (
                        "Deterministic final decision failed because execution feedback still contains a hard failure."
                        + (f"\nExecution feedback summary: {execution_summary}" if execution_summary else "")
                    )
                elif _value_feedback_has_blocking_signal(factor_feedback.value_feedback):
                    factor_feedback.final_decision = False
                    value_summary = _first_nonempty_line(factor_feedback.value_feedback)
                    factor_feedback.final_feedback = (
                        "Deterministic final decision failed because value feedback contains a blocking mismatch signal."
                        + (f"\nValue feedback summary: {value_summary}" if value_summary else "")
                    )
                else:
                    factor_feedback.final_decision = True
                    factor_feedback.final_feedback = (
                        "Deterministic checks passed: execution succeeded and no blocking value-check issue was found."
                    )
            return factor_feedback


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorFinalDecisionEvaluator.evaluate
    # - FactorCodeEvaluator.evaluate
