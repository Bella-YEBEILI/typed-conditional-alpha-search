"""
Crossover operator for combining multiple parent strategies.

The crossover operator takes multiple parent trajectories and generates a hybrid
strategy that combines their strengths while avoiding their weaknesses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import itertools
import math
import random

import yaml

from quantaalpha.log import logger
from quantaalpha.llm.client import APIBackend, robust_json_parse
from .trajectory import StrategyTrajectory, RoundPhase


# Default prompt path
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evolution_prompts.yaml"

REPAIR_COMPLEMENT_INTENT = "repair-complement"
STAGE_COMPLEMENT_INTENT = "stage-complement"
MECHANISM_SYNERGY_INTENT = "mechanism-synergy"
DIVERSE_ANGLE_INTENT = "diverse-angle"
DEFAULT_CROSSOVER_INTENTS: tuple[str, ...] = (
    REPAIR_COMPLEMENT_INTENT,
    STAGE_COMPLEMENT_INTENT,
    MECHANISM_SYNERGY_INTENT,
    DIVERSE_ANGLE_INTENT,
)
DEFAULT_CROSSOVER_INTENT_QUOTA: dict[str, float] = {
    REPAIR_COMPLEMENT_INTENT: 0.4,
    STAGE_COMPLEMENT_INTENT: 0.3,
    MECHANISM_SYNERGY_INTENT: 0.2,
    DIVERSE_ANGLE_INTENT: 0.1,
}
DEFAULT_CROSSOVER_CANDIDATE_FILTER: dict[str, Any] = {
    "max_per_direction": 4,
    "recency_rounds": 2,
}


class CrossoverOperator:
    """
    Combines multiple parent trajectories into hybrid strategies.
    
    The crossover process:
    1. Takes 2 or more parent trajectories
    2. Analyzes their strengths, weaknesses, and complementary aspects
    3. Generates a hybrid hypothesis that combines the best elements
    
    Key principles:
    - Synergy: Combine complementary aspects of parents
    - Improvement: Learn from both successes and failures
    - Innovation: Generate novel combinations not present in parents
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        """
        Initialize crossover operator.
        
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
                crossover_prompts = all_prompts.get("crossover", {})
                if crossover_prompts:
                    return crossover_prompts
            except Exception as e:
                logger.warning(f"Failed to load crossover prompts from {self.prompt_path}: {e}")
        
        # Minimal fallback prompts (English)
        logger.warning("Using minimal fallback prompts for crossover operator")
        return {
            "system": "You are a quantitative finance strategy fusion expert. Combine strategies effectively.",
            "user": "Combine parent strategies:\n{parent_summaries}",
            "simple_user": "Generate hybrid hypothesis from:\n{parent_summaries}",
            "parent_template": "Parent {idx}: {hypothesis}",
            "phase_names": {
                "original": "Original Round",
                "mutation": "Mutation Round",
                "crossover": "Crossover Round"
            }
        }
    
    def _format_parent_summary(self, parent: StrategyTrajectory, idx: int) -> str:
        """Format a single parent trajectory for the prompt."""
        phase_names = self.prompts.get("phase_names", {
            "original": "Original Round",
            "mutation": "Mutation Round",
            "crossover": "Crossover Round"
        })
        phase_name = phase_names.get(parent.phase.value, "Unknown")
        
        factors_str = ""
        if parent.factors:
            for f in parent.factors[:3]:
                name = f.get("name", "unknown")
                expr = f.get("expression", "")[:80]
                factors_str += f"  - {name}: {expr}\n"
        else:
            factors_str = "  N/A\n"
        
        metrics_str = ""
        if parent.backtest_metrics:
            for k, v in parent.backtest_metrics.items():
                if v is not None:
                    metrics_str += f"  - {k}: {v:.4f}\n"
        if not metrics_str:
            metrics_str = "  N/A\n"

        check_status = parent.get_submission_status_text() or "N/A"
        failed_metrics = ", ".join(parent.get_failed_metrics()) or "none"
        optimization_feedback = []
        for factor_name, payload in parent.submission_checks.items():
            feedback_text = str(payload.get("feedback_text") or "").strip()
            if not feedback_text:
                continue
            prefix = f"{factor_name}: " if len(parent.submission_checks) > 1 else ""
            optimization_feedback.append(f"{prefix}{feedback_text}")
        optimization_feedback_text = "\n".join(optimization_feedback) if optimization_feedback else "N/A"
        
        template = self.prompts.get("parent_template", "")
        if template:
            return template.format(
                idx=idx,
                phase_name=phase_name,
                direction_id=parent.direction_id,
                hypothesis=parent.hypothesis[:300] if parent.hypothesis else "N/A",
                factors=factors_str,
                metrics=metrics_str,
                feedback=parent.feedback[:200] if parent.feedback else "N/A",
                active_domains=", ".join(parent.active_domains) if parent.active_domains else "N/A",
                check_status=check_status,
                failed_metrics=failed_metrics,
                optimization_feedback=optimization_feedback_text,
            )
        
        # Default format
        return f"""### Parent {idx}: {phase_name}
**Direction ID**: {parent.direction_id}
**Hypothesis**: {parent.hypothesis[:300] if parent.hypothesis else 'N/A'}
**Factors**:
{factors_str}
**Metrics**:
{metrics_str}
**Active Domains**:
{", ".join(parent.active_domains) if parent.active_domains else "N/A"}
**Submission Checks**:
{check_status}
**Failed Metrics**:
{failed_metrics}
**Optimization Feedback**:
{optimization_feedback_text}
**Feedback**:
{parent.feedback[:200] if parent.feedback else 'N/A'}
        ---
"""

    @staticmethod
    def _normalize_domain_signature(trajectory: StrategyTrajectory) -> tuple[str, ...]:
        domains = [
            str(domain).strip()
            for domain in (trajectory.active_domains or [])
            if str(domain).strip()
        ]
        return tuple(sorted(set(domains)))

    @staticmethod
    def _normalize_intent_quota(intent_quota: Optional[dict[str, Any]]) -> dict[str, float]:
        normalized = dict(DEFAULT_CROSSOVER_INTENT_QUOTA)
        for intent_name, value in dict(intent_quota or {}).items():
            intent_key = str(intent_name).strip()
            if intent_key not in DEFAULT_CROSSOVER_INTENTS:
                continue
            try:
                normalized[intent_key] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        if sum(normalized.values()) <= 0:
            return dict(DEFAULT_CROSSOVER_INTENT_QUOTA)
        return normalized

    @staticmethod
    def _normalize_candidate_filter(candidate_filter: Optional[dict[str, Any]]) -> dict[str, Any]:
        normalized = dict(DEFAULT_CROSSOVER_CANDIDATE_FILTER)
        raw = dict(candidate_filter or {})
        for key in ("max_per_direction", "recency_rounds"):
            try:
                normalized[key] = max(0, int(raw.get(key, normalized[key])))
            except (TypeError, ValueError):
                continue
        return normalized

    @staticmethod
    def _non_null_metric_count(trajectory: StrategyTrajectory) -> int:
        return sum(1 for value in (trajectory.backtest_metrics or {}).values() if value is not None)

    @staticmethod
    def _trajectory_sort_key(trajectory: StrategyTrajectory) -> tuple[int, int, int, int, int, str]:
        return (
            int(trajectory.round_idx),
            int(trajectory.is_check_passed()),
            len(trajectory.get_pass_set()),
            -len(trajectory.get_fail_set()),
            CrossoverOperator._non_null_metric_count(trajectory),
            str(trajectory.created_at or ""),
        )

    @classmethod
    def _filter_candidates(
        cls,
        candidates: list[StrategyTrajectory],
        selection_strategy: str,
        candidate_filter: Optional[dict[str, Any]],
    ) -> list[StrategyTrajectory]:
        normalized_filter = cls._normalize_candidate_filter(candidate_filter)
        filtered = list(candidates)
        if not filtered:
            return []

        recency_rounds = int(normalized_filter["recency_rounds"])
        if recency_rounds > 0:
            recent_rounds = sorted({int(t.round_idx) for t in filtered}, reverse=True)[:recency_rounds]
            recent_round_set = set(recent_rounds)
            filtered = [t for t in filtered if int(t.round_idx) in recent_round_set]

        max_per_direction = int(normalized_filter["max_per_direction"])
        if max_per_direction > 0:
            by_direction: dict[int, list[StrategyTrajectory]] = {}
            for trajectory in filtered:
                by_direction.setdefault(int(trajectory.direction_id), []).append(trajectory)
            limited: list[StrategyTrajectory] = []
            for _, items in sorted(by_direction.items(), key=lambda item: item[0]):
                ranked = sorted(items, key=cls._trajectory_sort_key, reverse=True)
                limited.extend(ranked[:max_per_direction])
            filtered = limited

        filtered.sort(key=cls._trajectory_sort_key, reverse=True)
        return filtered

    @staticmethod
    def _match_repair_complement(left: StrategyTrajectory, right: StrategyTrajectory) -> bool:
        left_fail = left.get_fail_set()
        right_fail = right.get_fail_set()
        left_pass = left.get_pass_set()
        right_pass = right.get_pass_set()
        if not left_fail and not right_fail:
            return False
        if left_fail == right_fail:
            return False
        left_repaired_by_right = (left_fail & right_pass) - left_pass
        right_repaired_by_left = (right_fail & left_pass) - right_pass
        return bool(left_repaired_by_right or right_repaired_by_left)

    @staticmethod
    def _match_stage_complement(left: StrategyTrajectory, right: StrategyTrajectory) -> bool:
        # raw/zz1000s gate is treated as a quality metric, not a prerequisite:
        # any pair with differing stage signatures is eligible regardless of pass/fail.
        return left.get_stage_signature() != right.get_stage_signature()

    @staticmethod
    def _match_mechanism_synergy(left: StrategyTrajectory, right: StrategyTrajectory) -> bool:
        # raw/zz1000s gate is informational only; we no longer require both parents
        # to have passed it. Mechanism diversity remains the meaningful pairing signal.
        left_keys = left.get_mechanism_keys()
        right_keys = right.get_mechanism_keys()
        if not left_keys or not right_keys or left_keys == right_keys:
            return False
        overlap = len(left_keys & right_keys)
        return overlap <= max(1, min(len(left_keys), len(right_keys)) // 2)

    @staticmethod
    def _match_diverse_angle(left: StrategyTrajectory, right: StrategyTrajectory) -> bool:
        return any(
            (
                left.direction_id != right.direction_id,
                left.phase != right.phase,
                CrossoverOperator._normalize_domain_signature(left) != CrossoverOperator._normalize_domain_signature(right),
                left.get_stage_signature() != right.get_stage_signature(),
                left.get_mechanism_keys() != right.get_mechanism_keys(),
            )
        )

    @classmethod
    def _match_intent(cls, combo: tuple[StrategyTrajectory, ...], crossover_intent: str) -> bool:
        for left, right in itertools.combinations(combo, 2):
            if crossover_intent == REPAIR_COMPLEMENT_INTENT and cls._match_repair_complement(left, right):
                return True
            if crossover_intent == STAGE_COMPLEMENT_INTENT and cls._match_stage_complement(left, right):
                return True
            if crossover_intent == MECHANISM_SYNERGY_INTENT and cls._match_mechanism_synergy(left, right):
                return True
            if crossover_intent == DIVERSE_ANGLE_INTENT and cls._match_diverse_angle(left, right):
                return True
        return False

    @classmethod
    def _build_combo_evidence(cls, combo: tuple[StrategyTrajectory, ...], crossover_intent: str) -> str:
        lines: list[str] = []
        for idx, trajectory in enumerate(combo, start=1):
            lines.append(
                f"Parent {idx}: check_passed={trajectory.is_check_passed()}, "
                f"stage={trajectory.get_stage_signature()}, "
                f"fails={sorted(trajectory.get_fail_set()) or ['none']}, "
                f"passes={sorted(trajectory.get_pass_set()) or ['none']}, "
                f"mechanisms={sorted(trajectory.get_mechanism_keys()) or ['none']}"
            )
        if crossover_intent == REPAIR_COMPLEMENT_INTENT:
            for left, right in itertools.combinations(combo, 2):
                left_repaired = sorted(((left.get_fail_set() & right.get_pass_set()) - left.get_pass_set()))
                right_repaired = sorted(((right.get_fail_set() & left.get_pass_set()) - right.get_pass_set()))
                if left_repaired:
                    lines.append(
                        f"{right.trajectory_id} can cover {left.trajectory_id} on: {', '.join(left_repaired)}"
                    )
                if right_repaired:
                    lines.append(
                        f"{left.trajectory_id} can cover {right.trajectory_id} on: {', '.join(right_repaired)}"
                    )
        return "\n".join(f"- {line}" for line in lines) if lines else "- none"

    @classmethod
    def _build_crossover_intent_payload(
        cls,
        parents: list[StrategyTrajectory],
        crossover_intent: str,
    ) -> dict[str, str]:
        combo = tuple(parents)
        if crossover_intent == REPAIR_COMPLEMENT_INTENT:
            requirements = [
                "Identify which parent passes a metric that the other parent fails.",
                "Keep the passing mechanism intact instead of averaging everything together.",
                "Explicitly target a fail-to-pass repair path in the new fusion hypothesis.",
            ]
            objective = "Fuse the parents so one parent's passed checks repair the other's failed metrics."
        elif crossover_intent == STAGE_COMPLEMENT_INTENT:
            requirements = [
                "Use the stage mismatch between parents as the main source of complementarity.",
                "Explain whether the fusion is meant to preserve raw alpha, residual alpha, or both.",
                "Avoid combining parents if the result would simply inherit the same stage failure pattern.",
            ]
            objective = "Fuse parents with different raw/zz1000s/complete pass signatures into a more balanced strategy."
        elif crossover_intent == MECHANISM_SYNERGY_INTENT:
            requirements = [
                "Treat both parents as already-credible mechanisms and search for a stronger joint structure.",
                "Preserve the semantic role of each parent instead of blurring them into one vague idea.",
                "Prefer weighted, conditional, nested, or interaction-based fusion with an explicit rationale.",
            ]
            objective = "Combine two check-passed parents whose mechanisms are different enough to create synergy."
        else:
            requirements = [
                "Use a materially different angle from each parent to maintain diversity.",
                "Do not let one parent fully dominate the other unless the evidence clearly supports that role split.",
                "Keep the fusion hypothesis novel at the direction, phase, or mechanism level.",
            ]
            objective = "Use distinct parent angles to create a novel crossover direction."
        return {
            "intent_name": crossover_intent,
            "objective": objective,
            "requirements_text": "\n".join(f"- {item}" for item in requirements),
            "evidence": cls._build_combo_evidence(combo, crossover_intent),
        }

    @classmethod
    def _combo_signature_tokens(
        cls,
        combo: tuple[StrategyTrajectory, ...],
        crossover_intent: str,
    ) -> set[str]:
        tokens = {f"intent:{crossover_intent}"}
        for trajectory in combo:
            if trajectory.get_fail_set():
                tokens.update(f"fail:{item}" for item in sorted(trajectory.get_fail_set()))
            else:
                tokens.add("fail:none")
            tokens.update(f"pass:{item}" for item in sorted(trajectory.get_pass_set()))
            tokens.add(f"stage:{trajectory.get_stage_signature()}")
            for key in sorted(trajectory.get_mechanism_keys()):
                tokens.add(f"mechanism:{key}")
        return tokens

    @classmethod
    def _allocate_intent_targets(cls, crossover_n: int, intent_quota: dict[str, float]) -> dict[str, int]:
        if crossover_n <= 0:
            return {intent: 0 for intent in DEFAULT_CROSSOVER_INTENTS}
        total_weight = sum(intent_quota.values())
        if total_weight <= 0:
            intent_quota = dict(DEFAULT_CROSSOVER_INTENT_QUOTA)
            total_weight = sum(intent_quota.values())
        raw_targets = {
            intent: (crossover_n * float(intent_quota.get(intent, 0.0)) / total_weight)
            for intent in DEFAULT_CROSSOVER_INTENTS
        }
        allocated = {intent: int(math.floor(target)) for intent, target in raw_targets.items()}
        remaining = crossover_n - sum(allocated.values())
        remainders = sorted(
            DEFAULT_CROSSOVER_INTENTS,
            key=lambda intent: (raw_targets[intent] - allocated[intent], intent),
            reverse=True,
        )
        for intent in remainders[:remaining]:
            allocated[intent] += 1
        return allocated

    @classmethod
    def _diversified_pick(
        cls,
        combos: list[tuple[StrategyTrajectory, ...]],
        crossover_intent: str,
        target_n: int,
        selected_keys: set[tuple[str, ...]],
        parent_usage: dict[str, int],
        max_parent_reuse: int,
    ) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        covered_tokens: set[str] = set()
        remaining = list(combos)
        while remaining and len(picked) < target_n:
            best_idx = -1
            best_score: tuple[int, int, int] | None = None
            for idx, combo in enumerate(remaining):
                combo_key = tuple(sorted(t.trajectory_id for t in combo))
                if combo_key in selected_keys:
                    continue
                if max_parent_reuse > 0 and any(parent_usage.get(t.trajectory_id, 0) >= max_parent_reuse for t in combo):
                    continue
                tokens = cls._combo_signature_tokens(combo, crossover_intent)
                new_token_count = len(tokens - covered_tokens)
                novelty = len({t.direction_id for t in combo}) + len({t.phase for t in combo})
                metric_richness = sum(cls._non_null_metric_count(t) for t in combo)
                score = (new_token_count, novelty, metric_richness)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx < 0:
                break
            combo = remaining.pop(best_idx)
            combo_key = tuple(sorted(t.trajectory_id for t in combo))
            selected_keys.add(combo_key)
            for trajectory in combo:
                parent_usage[trajectory.trajectory_id] = parent_usage.get(trajectory.trajectory_id, 0) + 1
            covered_tokens.update(cls._combo_signature_tokens(combo, crossover_intent))
            picked.append(
                {
                    "parents": list(combo),
                    "crossover_intent": crossover_intent,
                }
            )
        return picked
    
    def generate_crossover(
        self,
        parents: list[StrategyTrajectory],
        use_detailed_prompt: bool = True,
        crossover_intent: str = REPAIR_COMPLEMENT_INTENT,
    ) -> dict[str, str]:
        """
        Generate a crossover (hybrid) strategy from multiple parents.
        
        Args:
            parents: List of parent trajectories to combine
            use_detailed_prompt: Whether to use detailed prompt with JSON output
            
        Returns:
            Dictionary containing crossover results:
            - "hybrid_hypothesis": The hybrid hypothesis text
            - "combination_rationale": Why the parents should be combined semantically
            - "parent_alpha_source_analysis": Alpha source and economic meaning of each parent
            - "fusion_logic": How parents were combined
            - "parent_role_assignment": Role split across parents or domains
            - "complementarity_reasoning": Why combination should outperform standalone parents
            - "combination_strategy": Weighted / conditional / nested / interaction strategy
            - "semantic_guidance": Downstream semantic guide for hypothesis generation
            - "innovation_points": Novel aspects of hybrid
            - "expected_benefits": Expected improvements
            - "parent_ids": List of parent trajectory IDs
        """
        if len(parents) < 2:
            logger.warning("Crossover requires at least 2 parents")
            return {"hybrid_hypothesis": parents[0].hypothesis if parents else ""}
        
        # Format parent summaries
        parent_summaries = "\n".join(
            self._format_parent_summary(p, i + 1) 
            for i, p in enumerate(parents)
        )
        
        # Build prompt
        system_prompt = self.prompts.get("system", "")
        
        intent_payload = self._build_crossover_intent_payload(parents, crossover_intent)
        if use_detailed_prompt:
            user_prompt = self.prompts.get("user", "").format(
                parent_summaries=parent_summaries,
                intent_name=intent_payload["intent_name"],
                intent_objective=intent_payload["objective"],
                intent_requirements=intent_payload["requirements_text"],
                intent_evidence=intent_payload["evidence"],
            )
        else:
            user_prompt = self.prompts.get("simple_user", "").format(
                parent_summaries=parent_summaries,
                intent_name=intent_payload["intent_name"],
                intent_objective=intent_payload["objective"],
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
                result = {"hybrid_hypothesis": response.strip()}
            
            result["parent_ids"] = [p.trajectory_id for p in parents]
            result["crossover_intent"] = intent_payload["intent_name"]
            
            logger.info(f"Generated crossover from {len(parents)} parents: "
                       f"{[p.trajectory_id for p in parents]}")
            return result
            
        except Exception as e:
            logger.error(f"Crossover generation failed: {e}")
            # Return fallback
            return self._generate_fallback_crossover(parents, crossover_intent=crossover_intent)
    
    def _parse_detailed_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from LLM."""
        try:
            data = robust_json_parse(response)
            return {
                "hybrid_hypothesis": data.get("hybrid_hypothesis", ""),
                "combination_rationale": data.get("combination_rationale", ""),
                "parent_alpha_source_analysis": data.get("parent_alpha_source_analysis", ""),
                "fusion_logic": data.get("fusion_logic", ""),
                "parent_role_assignment": data.get("parent_role_assignment", ""),
                "complementarity_reasoning": data.get("complementarity_reasoning", ""),
                "combination_strategy": data.get("combination_strategy", ""),
                "semantic_guidance": data.get("semantic_guidance", ""),
                "innovation_points": data.get("innovation_points", ""),
                "expected_benefits": data.get("expected_benefits", "")
            }
        except Exception:
            return {"hybrid_hypothesis": response.strip()}
    
    def _generate_fallback_crossover(
        self,
        parents: list[StrategyTrajectory],
        crossover_intent: str = REPAIR_COMPLEMENT_INTENT,
    ) -> dict[str, str]:
        """Generate a fallback crossover when LLM fails."""
        # Simple heuristic: combine keywords from parent hypotheses
        keywords = []
        for p in parents:
            if p.hypothesis:
                # Extract key concepts
                words = p.hypothesis[:100].split()
                keywords.extend(words[:5])
        
        hypothesis = f"Hybrid strategy: combining advantages of {len(parents)} parent strategies, " \
                    f"exploring synergistic effects in directions including {', '.join(set(keywords[:3]))}"
        
        return {
            "hybrid_hypothesis": hypothesis,
            "combination_rationale": "Combine parents only when they contribute complementary alpha mechanisms or a clear role split.",
            "parent_alpha_source_analysis": "Analyze each parent as a distinct alpha source before fusing them.",
            "fusion_logic": "Simple fusion of core concepts from each parent",
            "parent_role_assignment": "Assign each parent a distinct semantic role instead of blending formulas mechanically.",
            "complementarity_reasoning": "Combine parents when one contributes a stable anchor and the other contributes conditional timing or confirmation.",
            "combination_strategy": "weighted or conditional, depending on whether the parents should be blended continuously or switched by regime.",
            "semantic_guidance": "First keep the parent role split explicit, then generate a hybrid hypothesis that respects the chosen combination structure.",
            "innovation_points": "Multi-strategy combination may produce synergistic effects",
            "expected_benefits": "Reduce single strategy risk through combination",
            "parent_ids": [p.trajectory_id for p in parents],
            "crossover_intent": crossover_intent,
        }
    
    def generate_crossover_prompt_suffix(
        self, 
        parents: list[StrategyTrajectory],
        crossover_intent: str = REPAIR_COMPLEMENT_INTENT,
    ) -> str:
        """
        Generate a prompt suffix to be appended to the hypothesis generator.
        
        This suffix instructs the hypothesis generator to create a hybrid strategy.
        
        Args:
            parents: List of parent trajectories
            
        Returns:
            Prompt suffix string
        """
        crossover_result = self.generate_crossover(
            parents,
            use_detailed_prompt=True,
            crossover_intent=crossover_intent,
        )
        intent_payload = self._build_crossover_intent_payload(parents, crossover_intent)
        
        parent_summaries = []
        for i, p in enumerate(parents):
            metrics_snapshot = self._format_metrics_snapshot(p)
            summary = f"""**Parent {i+1}** (Direction {p.direction_id}, {p.phase.value}):
- Hypothesis: {p.hypothesis[:200] if p.hypothesis else 'N/A'}...
- Active Domains: {', '.join(p.active_domains) if p.active_domains else 'N/A'}
- Metrics Snapshot: {metrics_snapshot}
- Submission Checks: {p.get_submission_status_text() or 'N/A'}
- Failed Metrics: {', '.join(p.get_failed_metrics()) or 'none'}"""
            parent_summaries.append(summary)
        
        # Use template from prompts if available
        suffix_template = self.prompts.get("suffix_template")
        if suffix_template:
            return suffix_template.format(
                parent_summaries=chr(10).join(parent_summaries),
                intent_name=intent_payload["intent_name"],
                intent_objective=intent_payload["objective"],
                intent_requirements=intent_payload["requirements_text"],
                intent_evidence=intent_payload["evidence"],
                hybrid_hypothesis=crossover_result.get('hybrid_hypothesis', 'Combine parent advantages'),
                combination_rationale=crossover_result.get('combination_rationale', ''),
                parent_alpha_source_analysis=crossover_result.get('parent_alpha_source_analysis', ''),
                complementarity_reasoning=crossover_result.get('complementarity_reasoning', ''),
                combination_strategy=crossover_result.get('combination_strategy', ''),
                fusion_logic=crossover_result.get('fusion_logic', ''),
                parent_role_assignment=crossover_result.get('parent_role_assignment', ''),
                semantic_guidance=crossover_result.get('semantic_guidance', ''),
                innovation_points=crossover_result.get('innovation_points', '')
            )
        
        # Default suffix (English)
        suffix = f"""

---

## Crossover Round Guidance

This is a crossover fusion exploration round that requires generating a hybrid strategy by combining multiple parent strategies.

### Parent Strategy Summaries
{chr(10).join(parent_summaries)}

### Crossover Intent
- Intent: {intent_payload["intent_name"]}
- Objective: {intent_payload["objective"]}
- Requirements:
{intent_payload["requirements_text"]}

### Failed/Passed Metric Map
{intent_payload["evidence"]}

### Fusion Direction Suggestions
 - **Parent Alpha Source Analysis**: {crossover_result.get('parent_alpha_source_analysis', '')}
- **Hybrid Hypothesis Direction**: {crossover_result.get('hybrid_hypothesis', 'Combine parent advantages')}
- **Why This Combination Works**: {crossover_result.get('combination_rationale', '')}
 - **Complementarity Reasoning**: {crossover_result.get('complementarity_reasoning', '')}
 - **Combination Strategy**: {crossover_result.get('combination_strategy', '')}
- **Fusion Logic**: {crossover_result.get('fusion_logic', '')}
- **Parent Role Assignment**: {crossover_result.get('parent_role_assignment', '')}
 - **Semantic Guidance**: {crossover_result.get('semantic_guidance', '')}
- **Innovation Points**: {crossover_result.get('innovation_points', '')}

### Important Notes
1. First explain the semantic reason for combining the parents before attempting any new expression design
2. Explicitly analyze the alpha source and economic meaning of each parent
3. Decide whether the combination should be weighted, conditional, nested, or interaction-based
4. Your new hypothesis should fuse the advantages of all parent strategies
5. Avoid inheriting common weaknesses of the parents
6. Look for synergistic effects between parent strategies
7. Generated factors should capture the comprehensive characteristics of the combined strategies
8. Use each parent's failed metrics and passed metrics to decide whether the combination can repair weaknesses through complementarity
9. Do not mix inactive data domains; stay within the active domains already present in the current run
10. Maintain diversity by combining distinct mechanisms rather than blending nearly identical parents

Please propose your fusion hypothesis based on the above crossover guidance.
"""
        return suffix

    @staticmethod
    def _format_metrics_snapshot(parent: StrategyTrajectory) -> str:
        metrics = {
            str(key): value
            for key, value in (parent.backtest_metrics or {}).items()
            if value is not None
        }
        if not metrics:
            return "N/A"
        return ", ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
    
    def select_crossover_pairs(
        self,
        candidates: list[StrategyTrajectory],
        crossover_size: int = 2,
        crossover_n: int = 3,
        prefer_diverse: bool = True,
        selection_strategy: str = "best",
        top_percent_threshold: float = 0.3,
        intent_quota: Optional[dict[str, Any]] = None,
        candidate_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Select parent groups for crossover.
        
        Args:
            candidates: All available trajectories
            crossover_size: Number of parents per group
            crossover_n: Number of groups to create
            prefer_diverse: Whether to prefer combinations from different directions
            selection_strategy: Parent selection strategy:
                - "best": Prioritize complementary metric/check profiles
                - "random": Random selection
                - "weighted": Deprecated alias for semantic pairing
                - "weighted_inverse": Deprecated alias for semantic pairing
                - "top_percent_plus_random": Deprecated alias for semantic pairing
            top_percent_threshold: Deprecated config compatibility field
            
        Returns:
            List of parent groups
        """
        del top_percent_threshold  # deprecated: retained only for config compatibility
        if len(candidates) < crossover_size or crossover_n <= 0:
            return []

        selection_strategy = str(selection_strategy or "best").strip().lower()
        if selection_strategy in {"weighted", "weighted_inverse", "top_percent_plus_random"}:
            logger.warning(
                f"Crossover selection strategy '{selection_strategy}' is deprecated; "
                "using semantic intent selection instead."
            )
            selection_strategy = "best"

        selected_candidates = self._filter_candidates(
            candidates=candidates,
            selection_strategy=selection_strategy,
            candidate_filter=candidate_filter,
        )
        if len(selected_candidates) < crossover_size:
            return []

        all_combos = list(itertools.combinations(selected_candidates, crossover_size))
        all_combos = [
            combo
            for combo in all_combos
            if len({self._normalize_domain_signature(t) for t in combo}) <= 1
        ]
        if not all_combos:
            return []

        if selection_strategy == "random":
            random.shuffle(all_combos)
            return [
                {
                    "parents": list(combo),
                    "crossover_intent": DIVERSE_ANGLE_INTENT,
                }
                for combo in all_combos[:crossover_n]
            ]

        normalized_quota = self._normalize_intent_quota(intent_quota)
        intent_targets = self._allocate_intent_targets(crossover_n, normalized_quota)
        combos_by_intent: dict[str, list[tuple[StrategyTrajectory, ...]]] = {
            intent: [] for intent in DEFAULT_CROSSOVER_INTENTS
        }
        for combo in all_combos:
            for intent in DEFAULT_CROSSOVER_INTENTS:
                if self._match_intent(combo, intent):
                    combos_by_intent[intent].append(combo)
                    break

        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, ...]] = set()
        parent_usage: dict[str, int] = {}
        max_parent_reuse = math.ceil(crossover_n / 2) if prefer_diverse else 0
        for intent in DEFAULT_CROSSOVER_INTENTS:
            selected.extend(
                self._diversified_pick(
                    combos=combos_by_intent[intent],
                    crossover_intent=intent,
                    target_n=intent_targets.get(intent, 0),
                    selected_keys=selected_keys,
                    parent_usage=parent_usage,
                    max_parent_reuse=max_parent_reuse,
                )
            )

        if len(selected) < crossover_n:
            for intent in DEFAULT_CROSSOVER_INTENTS:
                remaining = [
                    combo
                    for combo in combos_by_intent[intent]
                    if tuple(sorted(t.trajectory_id for t in combo)) not in selected_keys
                ]
                needed = crossover_n - len(selected)
                if needed <= 0:
                    break
                selected.extend(
                    self._diversified_pick(
                        combos=remaining,
                        crossover_intent=intent,
                        target_n=needed,
                        selected_keys=selected_keys,
                        parent_usage=parent_usage,
                        max_parent_reuse=max_parent_reuse,
                    )
                )

        return selected
