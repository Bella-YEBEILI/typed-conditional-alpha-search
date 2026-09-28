"""
Mutation operator for generating orthogonal strategies.

The mutation operator takes a parent trajectory and generates a new hypothesis
that explores an orthogonal/independent direction from the parent. This ensures
diversity in the exploration space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from quantaalpha.log import logger
from quantaalpha.llm.client import APIBackend, robust_json_parse
from .trajectory import StrategyTrajectory, RoundPhase

REPAIR_MUTATION_INTENT = "repair"
ORTHOGONAL_DIVERSE_MUTATION_INTENT = "orthogonal-diverse"
DEFAULT_MUTATION_INTENTS: tuple[str, str] = (
    REPAIR_MUTATION_INTENT,
    ORTHOGONAL_DIVERSE_MUTATION_INTENT,
)


# Default prompt path
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evolution_prompts.yaml"


class MutationOperator:
    """
    Generates orthogonal (mutated) strategies from parent trajectories.
    
    The mutation process:
    1. Takes a parent trajectory's hypothesis, factors, and feedback
    2. Generates a new hypothesis that explores an orthogonal direction
    3. The new hypothesis should be fundamentally different to ensure diversity
    
    Key principles:
    - Orthogonality: New strategy should be nearly independent from parent
    - Diversity: Avoid repeating exploration paths
    - Learning: Use feedback from parent to avoid known pitfalls
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        """
        Initialize mutation operator.
        
        Args:
            prompt_path: Path to YAML file containing prompts. 
                        If None, uses default prompt path.
        """
        self.prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> dict[str, str]:
        """Load prompts from YAML file."""
        if self.prompt_path and self.prompt_path.exists():
            try:
                all_prompts = yaml.safe_load(self.prompt_path.read_text(encoding="utf-8")) or {}
                mutation_prompts = all_prompts.get("mutation", {})
                if mutation_prompts:
                    return mutation_prompts
            except Exception as e:
                logger.warning(f"Failed to load mutation prompts from {self.prompt_path}: {e}")
        
        # Minimal fallback prompts (English)
        logger.warning("Using minimal fallback prompts for mutation operator")
        return {
            "system": "You are a quantitative finance strategy expert. Generate orthogonal strategies.",
            "user": "Generate an orthogonal strategy based on parent: {parent_hypothesis}",
            "simple_user": "Generate orthogonal hypothesis: {parent_hypothesis}",
            "fallback_templates": [
                "Explore mean reversion characteristics",
                "Study volume-price nonlinear relationships",
                "Analyze cross-cycle trend signals",
                "Mine market microstructure liquidity features",
            ]
        }
    
    def generate_mutation(
        self,
        parent: StrategyTrajectory,
        use_detailed_prompt: bool = True,
        mutation_intent: str = ORTHOGONAL_DIVERSE_MUTATION_INTENT,
    ) -> dict[str, str]:
        """
        Generate a mutated (orthogonal) strategy from parent.
        
        Args:
            parent: The parent trajectory to mutate from
            use_detailed_prompt: Whether to use detailed prompt (returns structured output)
                               or simple prompt (returns just hypothesis text)
        
        Returns:
            Dictionary containing mutation results:
            - "new_hypothesis": The new hypothesis text
            - "exploration_direction": Direction description (if detailed)
            - "orthogonality_reason": Why this is orthogonal (if detailed)
            - "expected_characteristics": Expected characteristics (if detailed)
        """
        # Format parent information
        parent_hypothesis = parent.hypothesis or "N/A"
        
        parent_factors = ""
        if parent.factors:
            for f in parent.factors[:5]:
                name = f.get("name", "unknown")
                expr = f.get("expression", "")
                desc = f.get("description", "")
                parent_factors += f"- {name}: {expr}\n  Description: {desc}\n"
        else:
            parent_factors = "N/A"
        
        parent_metrics = ""
        if parent.backtest_metrics:
            for k, v in parent.backtest_metrics.items():
                if v is not None:
                    parent_metrics += f"- {k}: {v:.4f}\n"
        if not parent_metrics:
            parent_metrics = "N/A"
        
        parent_feedback = parent.feedback or "N/A"
        parent_check_status = parent.get_submission_status_text() or "N/A"
        parent_failed_metrics = ", ".join(parent.get_failed_metrics()) or "none"
        parent_optimization_feedback = []
        for factor_name, payload in parent.submission_checks.items():
            feedback_text = str(payload.get("feedback_text") or "").strip()
            if not feedback_text:
                continue
            prefix = f"{factor_name}: " if len(parent.submission_checks) > 1 else ""
            parent_optimization_feedback.append(f"{prefix}{feedback_text}")
        parent_optimization_feedback_text = "\n".join(parent_optimization_feedback) if parent_optimization_feedback else "N/A"
        intent_payload = self._build_mutation_intent_payload(parent, mutation_intent)
        
        # Build prompt
        system_prompt = self.prompts.get("system", "")
        
        if use_detailed_prompt:
            user_prompt = self.prompts.get("user", "").format(
                parent_hypothesis=parent_hypothesis,
                parent_factors=parent_factors,
                parent_metrics=parent_metrics,
                parent_feedback=parent_feedback,
                parent_check_status=parent_check_status,
                parent_failed_metrics=parent_failed_metrics,
                parent_optimization_feedback=parent_optimization_feedback_text,
                mutation_intent=intent_payload["intent_name"],
                mutation_objective=intent_payload["objective"],
                mutation_requirements=intent_payload["requirements_text"],
            )
        else:
            user_prompt = self.prompts.get("simple_user", "").format(
                parent_hypothesis=parent_hypothesis,
                parent_factors=parent_factors,
                parent_check_status=parent_check_status,
                parent_failed_metrics=parent_failed_metrics,
                mutation_intent=intent_payload["intent_name"],
                mutation_objective=intent_payload["objective"],
            )
        
        # Call LLM
        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=use_detailed_prompt
            )
            
            if use_detailed_prompt:
                result = self._parse_detailed_response(response)
            else:
                result = {"new_hypothesis": response.strip()}
            result["mutation_intent"] = intent_payload["intent_name"]
             
            logger.info(f"Generated {intent_payload['intent_name']} mutation from parent {parent.trajectory_id}")
            return result
             
        except Exception as e:
            logger.error(f"Mutation generation failed: {e}")
            # Return fallback
            return {
                "new_hypothesis": self._generate_fallback_hypothesis(parent, mutation_intent=mutation_intent),
                "exploration_direction": intent_payload["objective"],
                "orthogonality_reason": "Using fallback strategy due to generation failure",
                "expected_characteristics": "Repair failed metrics or explore a new mechanism while preserving domain constraints",
                "mutation_intent": intent_payload["intent_name"],
            }
    
    def _parse_detailed_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from LLM."""
        try:
            data = robust_json_parse(response)
            return {
                "new_hypothesis": data.get("new_hypothesis", ""),
                "exploration_direction": data.get("exploration_direction", ""),
                "orthogonality_reason": data.get("orthogonality_reason", ""),
                "expected_characteristics": data.get("expected_characteristics", "")
            }
        except Exception:
            # If JSON parsing fails, treat entire response as hypothesis
            return {"new_hypothesis": response.strip()}
    
    def _generate_fallback_hypothesis(
        self,
        parent: StrategyTrajectory,
        mutation_intent: str = ORTHOGONAL_DIVERSE_MUTATION_INTENT,
    ) -> str:
        """Generate a fallback hypothesis when LLM fails."""
        parent_hypo = parent.hypothesis.lower() if parent.hypothesis else ""
        failed_metrics = ", ".join(parent.get_failed_metrics()) or "failed submission metrics"
        if mutation_intent == REPAIR_MUTATION_INTENT:
            return (
                f"Repair the parent strategy by directly targeting {failed_metrics} "
                "while preserving the same active data-domain constraints and retaining the strongest existing signal components"
            )
        
        # Get fallback templates from prompts
        fallback_templates = self.prompts.get("fallback_templates", [
            "Explore an orthogonal cross-sectional alpha mechanism under the same active domain constraints",
            "Study an alternative temporal mechanism under the same data-domain semantics",
            "Analyze regime-dependent behavior without changing the active data domains",
            "Build a robustness-oriented signal using a different construction logic in the same domain",
            "Explore participation-versus-efficiency mismatch under the current field constraints",
            "Search for a lower-correlation alpha within the same domain-specific research space",
        ])
        
        # Select based on parent content
        if "momentum" in parent_hypo:
            return "Explore a reversal-oriented alpha mechanism within the same active domain constraints"
        elif "mean reversion" in parent_hypo or "reversion" in parent_hypo:
            return "Explore a persistence-oriented alpha mechanism within the same active domain constraints"
        elif "volume" in parent_hypo:
            return "Explore an alternative construction path using the same active field family"
        elif "volatility" in parent_hypo:
            return "Explore a regime-sensitive variant that stays within the current domain semantics"
        else:
            import random
            return random.choice(fallback_templates)
    
    def _build_mutation_intent_payload(self, parent: StrategyTrajectory, mutation_intent: str) -> dict[str, str]:
        failed_metrics = ", ".join(parent.get_failed_metrics()) or "failed submission metrics"
        if mutation_intent == REPAIR_MUTATION_INTENT:
            requirements = [
                "Directly target the parent's failed submission metrics first.",
                "Preserve the strongest working parts of the parent instead of replacing everything.",
                "Prefer repairs that improve pass probability without leaving the same active domains.",
            ]
            return {
                "intent_name": REPAIR_MUTATION_INTENT,
                "objective": f"Repair the parent by improving {failed_metrics}.",
                "requirements_text": "\n".join(f"- {item}" for item in requirements),
                "notes": "This mutation is repair-oriented. Focus on the failed metrics before pursuing broader novelty.",
            }
        requirements = [
            "Explore a clearly different mechanism from the parent.",
            "Increase diversity while staying in the same active domains, and ensure at least two of the following three dimensions change materially: field usage, operator usage, structural composition.",
            "Do not just tune a parameter; shift the mechanism materially.",
        ]
        return {
            "intent_name": ORTHOGONAL_DIVERSE_MUTATION_INTENT,
            "objective": "Explore a new mechanism that is materially different from the parent while preserving domain constraints.",
            "requirements_text": "\n".join(f"- {item}" for item in requirements),
            "notes": "This mutation is diversity-oriented. Prefer a new mechanism over a small local tweak, and require at least two of field/operator/structure to change materially.",
        }

    def generate_mutation_prompt_suffix(
        self,
        parent: StrategyTrajectory,
        mutation_intent: str = ORTHOGONAL_DIVERSE_MUTATION_INTENT,
    ) -> str:
        """
        Generate a prompt suffix to be appended to the hypothesis generator.
        
        This suffix instructs the hypothesis generator to explore orthogonal directions.
        
        Args:
            parent: The parent trajectory
            
        Returns:
            Prompt suffix string
        """
        mutation_result = self.generate_mutation(parent, use_detailed_prompt=True, mutation_intent=mutation_intent)
        parent_check_status = parent.get_submission_status_text() or "N/A"
        parent_failed_metrics = ", ".join(parent.get_failed_metrics()) or "none"
        optimization_feedback = []
        for factor_name, payload in parent.submission_checks.items():
            feedback_text = str(payload.get("feedback_text") or "").strip()
            if not feedback_text:
                continue
            prefix = f"{factor_name}: " if len(parent.submission_checks) > 1 else ""
            optimization_feedback.append(f"{prefix}{feedback_text}")
        parent_optimization_feedback_text = "\n".join(optimization_feedback) if optimization_feedback else "N/A"
        intent_payload = self._build_mutation_intent_payload(parent, mutation_intent)
        
        # Use template from prompts if available
        suffix_template = self.prompts.get("suffix_template")
        if suffix_template:
            return suffix_template.format(
                parent_summary=parent.to_summary_text(),
                mutation_intent=intent_payload["intent_name"],
                mutation_objective=intent_payload["objective"],
                mutation_requirements=intent_payload["requirements_text"],
                mutation_intent_notes=intent_payload["notes"],
                new_hypothesis=mutation_result.get('new_hypothesis', 'Explore new direction'),
                exploration_direction=mutation_result.get('exploration_direction', ''),
                orthogonality_reason=mutation_result.get('orthogonality_reason', '')
            )
        
        # Default suffix (English)
        suffix = f"""

---

        ## Mutation Round Guidance

This is a mutation exploration round based on the parent strategy.

        ### Parent Strategy Summary
        {parent.to_summary_text()}

        ### Mutation Intent
        - Intent: {intent_payload["intent_name"]}
        - Objective: {intent_payload["objective"]}
        - Requirements:
{intent_payload["requirements_text"]}

        ### Submission Check Guidance
        - Current check status: {parent_check_status}
        - Failed metrics to optimize first: {parent_failed_metrics}
        - Optimization feedback: {parent_optimization_feedback_text}

        ### Mutation Direction Suggestions
        - **New Hypothesis Direction**: {mutation_result.get('new_hypothesis', 'Explore new direction')}
        - **Exploration Dimension**: {mutation_result.get('exploration_direction', '')}
        - **Orthogonality Reasoning**: {mutation_result.get('orthogonality_reason', '')}

### Important Notes
        1. Follow the mutation intent strictly: repair mutations fix failed metrics first; orthogonal-diverse mutations pursue a new mechanism
        2. If the parent failed specific submission metrics, directly target those weak points instead of giving generic improvements
        3. Preserve active data-domain boundaries from the parent run; do not introduce fields outside the same active domains
        4. Keep diversity high when the intent is orthogonal-diverse by exploring a different mechanism, not just a superficial parameter tweak
        5. For orthogonal-diverse mutations, require at least two of the following three dimensions to change materially: field usage, operator usage, structural composition

        Please propose your new hypothesis based on the above mutation guidance.
"""
        return suffix

