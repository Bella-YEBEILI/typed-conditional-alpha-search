import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, StrictUndefined

from quantaalpha.core.experiment import Experiment
from quantaalpha.core.proposal import (
    Hypothesis,
    HypothesisExperiment2Feedback,
    HypothesisFeedback,
    Trace,
)
from quantaalpha.factors.alignment.prompt_proxy import DomainPromptProxy
from quantaalpha.factors.data_domains import resolve_factor_domains
from quantaalpha.log import logger
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.utils import convert2bool

# Max retries for JSON parsing
MAX_JSON_PARSE_RETRIES = 3
FEEDBACK_STAGE = "selection"

_PROMPTS_DIR = Path(__file__).parent / "prompts"


base_feedback_prompts = DomainPromptProxy(base_file="prompts.yaml", prompt_dir=_PROMPTS_DIR)
DIRNAME = Path(__file__).absolute().resolve().parent


def _extract_stage_result(exp: Experiment | None, stage_name: str) -> dict | pd.Series | pd.DataFrame | None:
    if exp is None:
        return None

    stage_results = getattr(exp, "tq_stage_backtest_results", None)
    if isinstance(stage_results, dict) and stage_results:
        runtime_context = getattr(exp, "tq_runtime_context", None)
        fallback_stage_name = stage_name
        if isinstance(runtime_context, dict):
            primary_split = str(runtime_context.get("primary_split") or "").strip()
            if primary_split:
                fallback_stage_name = primary_split
        extracted: dict[str, dict] = {}
        for factor_name, factor_stage_payload in stage_results.items():
            if not isinstance(factor_stage_payload, dict):
                continue
            stage_payload = factor_stage_payload.get(stage_name)
            if not isinstance(stage_payload, dict) and fallback_stage_name != stage_name:
                stage_payload = factor_stage_payload.get(fallback_stage_name)
            if not isinstance(stage_payload, dict):
                continue
            metrics = stage_payload.get("metrics")
            if isinstance(metrics, dict) and metrics:
                extracted[str(factor_name)] = metrics
        if extracted:
            return extracted

    return getattr(exp, "result", None)


def process_results(current_result, sota_result):
    def _normalize_factor_result(factor_name, res):
        if isinstance(res, pd.DataFrame):
            if res.empty:
                return None
            df = res.copy()
            if len(df.columns) == 0:
                df[factor_name] = 0
            elif factor_name not in df.columns:
                first_col = df.columns[0]
                df = df.rename(columns={first_col: factor_name})
            df.index.name = "metric"
            return df[[factor_name]] if factor_name in df.columns else df

        if isinstance(res, pd.Series):
            df = res.to_frame(name=factor_name)
            df.index.name = "metric"
            return df

        if isinstance(res, dict):
            if not res:
                return None
            series = pd.Series(res, name=factor_name)
            df = series.to_frame()
            df.index.name = "metric"
            return df

        if res is None:
            return None

        df = pd.DataFrame({factor_name: [res]}, index=["value"])
        df.index.name = "metric"
        return df

    # Handle dictionary of results (multi-factor backtest)
    if isinstance(current_result, dict) and not isinstance(current_result, pd.DataFrame):
        dfs = []
        for factor_name, res in current_result.items():
            if isinstance(res, (pd.DataFrame, pd.Series, dict)) or res is not None:
                df = _normalize_factor_result(factor_name, res)
                if df is None:
                    continue
                dfs.append(df)
        
        if not dfs:
            current_df = pd.DataFrame()
        else:
            current_df = pd.concat(dfs, axis=1)
    else:
        # Convert the results to dataframes
        current_df = _normalize_factor_result("Current Result", current_result)
        if current_df is None:
            current_df = pd.DataFrame()
        current_df.index.name = "metric"
        # Rename the value column to reflect the result type
        if "0" in current_df.columns:
            current_df.rename(columns={"0": "Current Result"}, inplace=True)
        elif len(current_df.columns) > 0:
            first_col = current_df.columns[0]
            current_df.rename(columns={first_col: "Current Result"}, inplace=True)
        else:
            current_df["Current Result"] = 0
    
    # Handle case where sota_result might be None or empty
    if sota_result is None or (isinstance(sota_result, pd.DataFrame) and sota_result.empty):
        # If no SOTA result, return only current result
        
        # Select important metrics for comparison
        important_metrics = [
            "long_ret",
            "long_ir",
            "long_maxdd",
            "long_netret",
            "long_netir",
            "long_netmaxdd",
            "ls_ret",
            "ls_ir",
            "ls_maxdd",
            "ls_netret",
            "ls_netir",
            "ls_netmaxdd",
            "rankic",
            "rankicir",
            "long_turnover",
            "ls_turnover",
            "long_num",
            "coverage",
        ]
        
        # Filter the DataFrame to retain only the important metrics that exist
        available_metrics = [m for m in important_metrics if m in current_df.index]
        if available_metrics:
            filtered_df = current_df.loc[available_metrics]
        else:
            filtered_df = current_df
        
        return filtered_df.to_string()
    
    sota_df = pd.DataFrame(sota_result)
    sota_df.index.name = "metric"

    if "0" in sota_df.columns:
        sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)
    elif len(sota_df.columns) > 0:
        first_col = sota_df.columns[0]
        sota_df.rename(columns={first_col: "SOTA Result"}, inplace=True)
    else:
        sota_df["SOTA Result"] = 0

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Select important metrics for comparison
    important_metrics = [
        "long_ret",
        "long_ir",
        "long_maxdd",
        "long_netret",
        "long_netir",
        "long_netmaxdd",
        "ls_ret",
        "ls_ir",
        "ls_maxdd",
        "ls_netret",
        "ls_netir",
        "ls_netmaxdd",
        "rankic",
        "rankicir",
        "long_turnover",
        "ls_turnover",
        "long_num",
        "coverage",
    ]

    # Filter the combined DataFrame to retain only the important metrics that exist
    available_metrics = [m for m in important_metrics if m in combined_df.index]
    if available_metrics:
        filtered_combined_df = combined_df.loc[available_metrics]
    else:
        filtered_combined_df = combined_df

    # Check if both columns exist before comparing
    if "Current Result" in filtered_combined_df.columns and "SOTA Result" in filtered_combined_df.columns:
        filtered_combined_df[
            "Bigger columns name (Didn't consider the direction of the metric, you should judge it by yourself that bigger is better or smaller is better)"
        ] = filtered_combined_df.apply(
                lambda row: "Current Result" if pd.notna(row["Current Result"]) and pd.notna(row["SOTA Result"]) and row["Current Result"] > row["SOTA Result"] else "SOTA Result", axis=1
        )
    elif "Current Result" in filtered_combined_df.columns:
        # Only current result available
        filtered_combined_df[
            "Bigger columns name (Didn't consider the direction of the metric, you should judge it by yourself that bigger is better or smaller is better)"
        ] = "Current Result"
    elif "SOTA Result" in filtered_combined_df.columns:
        # Only SOTA result available
        filtered_combined_df[
            "Bigger columns name (Didn't consider the direction of the metric, you should judge it by yourself that bigger is better or smaller is better)"
        ] = "SOTA Result"

    return filtered_combined_df.to_string()


def _extract_parent_role_assignment(text: str | None) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    pattern = re.compile(r"Parent Role Assignment\*{0,2}\*{0,2}\s*:\s*(.+)")
    for line in raw.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        value = re.sub(r"[*`_]+", "", match.group(1)).strip(" -:")
        if value:
            return value
    return ""


def _build_feedback_runtime_context(exp: Experiment, task_details: list[dict[str, Any]]) -> str:
    runtime_context = getattr(exp, "tq_runtime_context", {}) or {}
    active_domains = [
        str(domain)
        for domain in (runtime_context.get("data_domains") or resolve_factor_domains())
        if str(domain).strip()
    ]
    parts: list[str] = []
    if active_domains:
        parts.append(f"Current run active domains: {', '.join(active_domains)}.")

    qa_feedback_context = getattr(exp, "qa_feedback_context", {}) or {}
    parent_role_assignment = _extract_parent_role_assignment(qa_feedback_context.get("strategy_suffix"))
    if parent_role_assignment:
        parts.append(f"Parent strategy role assignment from the current crossover guidance: {parent_role_assignment}.")

    runtime_fragments: list[str] = []
    for task in task_details:
        factor_name = str(task.get("factor_name") or "").strip()
        context_text = str(task.get("runtime_analysis_context") or "").strip()
        if not factor_name or not context_text:
            continue
        runtime_fragments.append(f"{factor_name}: {context_text}")
    if runtime_fragments:
        parts.append("Observed per-factor runtime evidence: " + " ".join(runtime_fragments))

    return " ".join(parts)


class FactorExperiment2Feedback(HypothesisExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (FactorMiningExperiment): The experiment to generate feedback for.
            hypothesis (FactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = _extract_stage_result(exp, FEEDBACK_STAGE)
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        runtime_context_summary = _build_feedback_runtime_context(exp, tasks_factors)
        # Safely get SOTA result, handle case where based_experiments might be empty or result is None
        sota_result = None
        if exp.based_experiments and len(exp.based_experiments) > 0:
            sota_result = _extract_stage_result(exp.based_experiments[-1], FEEDBACK_STAGE)

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(base_feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )

        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(base_feedback_prompts["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
                runtime_context_summary=runtime_context_summary,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback with retry
        response_json = None
        last_error = None
        
        for attempt in range(MAX_JSON_PARSE_RETRIES):
            try:
                response = APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=usr_prompt,
                    system_prompt=sys_prompt,
                    json_mode=True,
                )
                # Parse the JSON response using robust parser
                response_json = robust_json_parse(response)
                break
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"[QuantaAlpha] JSON parse failed (attempt {attempt + 1}/{MAX_JSON_PARSE_RETRIES}): {e}")
                if attempt < MAX_JSON_PARSE_RETRIES - 1:
                    logger.info("[QuantaAlpha] Re-requesting LLM...")
                continue
        
        if response_json is None:
            logger.error(f"[QuantaAlpha] JSON parse still failed after {MAX_JSON_PARSE_RETRIES} attempts")
            return HypothesisFeedback(
                observations="JSON parse failed; could not extract feedback",
                hypothesis_evaluation="Unable to evaluate",
                new_hypothesis="",
                reason=f"JSON parse error: {last_error}",
                decision=False,
            )

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )



qa_feedback_prompts = DomainPromptProxy(base_file="prompts.yaml", prompt_dir=_PROMPTS_DIR)
class AlphaAgentFactorExperiment2Feedback(HypothesisExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (FactorMiningExperiment): The experiment to generate feedback for.
            hypothesis (FactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = _extract_stage_result(exp, FEEDBACK_STAGE)
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        runtime_context_summary = _build_feedback_runtime_context(exp, tasks_factors)
        # Safely get SOTA result, handle case where based_experiments might be empty or result is None
        sota_result = None
        if exp.based_experiments and len(exp.based_experiments) > 0:
            sota_result = _extract_stage_result(exp.based_experiments[-1], FEEDBACK_STAGE)

        # Extract complexity information by directly calculating from factor expressions
        # Import complexity calculation functions
        try:
            from quantaalpha.factors.coder.factor_ast import (
                calculate_symbol_length, count_base_features
            )
            from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
            
            for idx, task_detail in enumerate(tasks_factors):
                if idx < len(exp.sub_tasks):
                    task = exp.sub_tasks[idx]
                    factor_expr = task_detail.get("factor_expression", "")
                    if factor_expr:
                        complexity_warnings = []
                        # Calculate symbol length
                        symbol_length = calculate_symbol_length(factor_expr)
                        symbol_length_threshold = getattr(FACTOR_COSTEER_SETTINGS, 'symbol_length_threshold', 400)
                        if symbol_length > symbol_length_threshold:
                            complexity_warnings.append(
                                f"Symbol Length (SL) Warning: Symbol length ({symbol_length}) exceeds soft threshold ({symbol_length_threshold}). "
                                f"Treat this as a complexity risk diagnostic, not an automatic simplification mandate."
                            )
                        
                        # Calculate base features count
                        num_base_features = count_base_features(factor_expr)
                        base_features_threshold = getattr(FACTOR_COSTEER_SETTINGS, 'base_features_threshold', 6)
                        if num_base_features > base_features_threshold:
                            complexity_warnings.append(
                                f"Base Features Count (ER) Warning: Number of base features ({num_base_features}) exceeds soft threshold ({base_features_threshold}). "
                                f"Keep the complexity if the cross-field mechanism is intentional and executable."
                            )
                        
                        if complexity_warnings:
                            task_detail["complexity_feedback"] = "\n".join(complexity_warnings)
        except Exception as e:
            logger.warning(f"Failed to calculate complexity info: {e}")

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )

        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_feedback_prompts["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
                runtime_context_summary=runtime_context_summary,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback with retry
        response_json = None
        last_error = None
        
        for attempt in range(MAX_JSON_PARSE_RETRIES):
            try:
                response = APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=usr_prompt,
                    system_prompt=sys_prompt,
                    json_mode=True,
                )
                # Parse the JSON response using robust parser
                response_json = robust_json_parse(response)
                break
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"[AlphaAgent] JSON parse failed (attempt {attempt + 1}/{MAX_JSON_PARSE_RETRIES}): {e}")
                if attempt < MAX_JSON_PARSE_RETRIES - 1:
                    logger.info("[AlphaAgent] Re-requesting LLM...")
                continue
        
        if response_json is None:
            logger.error(f"[AlphaAgent] JSON parse still failed after {MAX_JSON_PARSE_RETRIES} attempts")
            return HypothesisFeedback(
                observations="JSON parse failed; could not extract feedback",
                hypothesis_evaluation="Unable to evaluate",
                new_hypothesis="",
                reason=f"JSON parse error: {last_error}",
                decision=False,
            )

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class ModelHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    """Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks & their comparisons with previous performances"""

    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included, as well as the comparison between previous results (done by LLM).
        For example: experiment metrics and execution artifacts will be included.
        """

        logger.info("Generating feedback...")
        # Define the system prompt for hypothesis feedback
        system_prompt = feedback_prompts["model_feedback_generation"]["system"]

        # Define the user prompt for hypothesis feedback
        context = trace.scen
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(feedback_prompts["model_feedback_generation"]["user"])
            .render(
                context=context,
                last_hypothesis=SOTA_hypothesis,
                last_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
                last_code=(
                    (SOTA_experiment.sub_workspace_list[0].code_dict.get("factor.py")
                        or SOTA_experiment.sub_workspace_list[0].code_dict.get("model.py"))
                    if (
                        SOTA_hypothesis
                        and SOTA_experiment.sub_workspace_list
                        and SOTA_experiment.sub_workspace_list[0] is not None
                        and SOTA_experiment.sub_workspace_list[0].code_dict
                    )
                    else None
                ),
                last_result=SOTA_experiment.result if SOTA_hypothesis else None,
                hypothesis=hypothesis,
                exp=exp,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback with retry
        response_json_hypothesis = None
        last_error = None
        
        for attempt in range(MAX_JSON_PARSE_RETRIES):
            try:
                response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                )
                # Parse the JSON response using robust parser
                response_json_hypothesis = robust_json_parse(response_hypothesis)
                break
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"[Model] JSON parse failed (attempt {attempt + 1}/{MAX_JSON_PARSE_RETRIES}): {e}")
                if attempt < MAX_JSON_PARSE_RETRIES - 1:
                    logger.info("[Model] Re-requesting LLM...")
                continue
        
        if response_json_hypothesis is None:
            logger.error(f"[Model] JSON parse still failed after {MAX_JSON_PARSE_RETRIES} attempts")
            return HypothesisFeedback(
                observations="JSON parse failed; could not extract feedback",
                hypothesis_evaluation="Unable to evaluate",
                new_hypothesis="",
                reason=f"JSON parse error: {last_error}",
                decision=False,
            )
        
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )


FactorHypothesisExperiment2Feedback = FactorExperiment2Feedback
AlphaAgentFactorHypothesisExperiment2Feedback = AlphaAgentFactorExperiment2Feedback






