from __future__ import annotations

from quantaalpha.factors.alignment.prompt_proxy import DomainPromptProxy, resolve_prompt_mode
from quantaalpha.factors.alignment.registry import (
    get_allowed_field_names,
    get_operator_arity_constraints,
    get_prompt_operator_names,
    get_prompt_operator_signature_hints,
    get_operator_semantic_notes,
    get_base_operator_names,
    get_minute_operator_names,
)
from quantaalpha.factors.combined_domain_contract import is_joint_pv_minutes_run
from quantaalpha.factors import data_domains as factor_data_domains

__all__ = [
    "DomainPromptProxy",
    "resolve_prompt_mode",
    "build_runtime_registry_constraints",
    "build_runtime_field_constraints",
]


def _joint_shared_fields() -> tuple[str, ...]:
    fields = getattr(factor_data_domains, "JOINT_PV_MINUTES_SHARED_FIELDS", None)
    if fields is not None:
        return tuple(fields)
    pv_fields = set(getattr(factor_data_domains, "PV_FIELDS", ()))
    minute_fields = set(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    return tuple(sorted(pv_fields & minute_fields))


def build_runtime_registry_constraints() -> str:
    domains = factor_data_domains.resolve_factor_domains()
    allowed_operators = sorted(get_prompt_operator_names(domains))
    allowed_fields = sorted(get_allowed_field_names(domains))
    arity_constraints = get_operator_arity_constraints(domains)
    signature_hints = get_prompt_operator_signature_hints(domains)
    operator_allowlist = ", ".join(allowed_operators) if allowed_operators else "none"
    allowlist = ", ".join(allowed_fields) if allowed_fields else "none"
    required_signature_ops = [
        signature_hints[name]
        for name in allowed_operators
        if name in signature_hints and arity_constraints.get(name, (0, None))[0] > 1
    ]
    signature_note = ""
    if required_signature_ops:
        signature_note = (
            "\n- Do not omit required operator arguments. Runtime signatures that require explicit extra args include: "
            + ", ".join(required_signature_ops)
        )
    semantic_notes = get_operator_semantic_notes(domains)
    semantic_note = ""
    if semantic_notes:
        semantic_note = "\n- Operator semantic notes:\n- " + "\n- ".join(semantic_notes)
    minute_runtime_note = ""
    if "minutes" in domains:
        minute_ops = ", ".join(sorted(get_minute_operator_names())) or "none"
        daily_ops = ", ".join(sorted(get_base_operator_names())) or "none"
        minute_runtime_note = (
            "\n- Minute runtime note: raw minute tensors inside `MinuteFactorEngine.run(...)` "
            "should use registered minute operators.\n"
            "- Use `ts_return(field, lag)` only for lagged same-field returns. Use `pct(a, b)` only for same-time tensor ratio `a / b - 1`; do not overload `pct(field, lag)`.\n"
            "- After minute data has been reduced to a daily 2D frame (`dates x stocks`), "
            "you may continue composing with the regular daily PV operators.\n"
            "- Once a minute reduce operator has produced a daily 2D frame, later `ts_rank/ts_zscore/ts_corr/...` calls follow regular daily semantics, not intraday minute semantics.\n"
            f"- Registered minute operators: {minute_ops}\n"
            f"- Daily 2D operators available after reduction: {daily_ops}"
        )
    joint_fields_note = ""
    if is_joint_pv_minutes_run(domains):
        daily_aliases = getattr(factor_data_domains, "JOINT_PV_DAILY_SHARED_FIELD_ALIASES", {})
        alias_text = ", ".join(f"{alias}->{target}" for alias, target in daily_aliases.items()) or "none"
        joint_fields_note = (
            f"\n- Joint pv/minutes explicit minute fields: {', '.join(factor_data_domains.MINUTE_FIELDS)}."
            f"\n- Joint pv/minutes explicit daily pv fields: {', '.join(factor_data_domains.PV_FIELDS)}."
            f"\n- Joint pv/minutes daily shared-field aliases: {alias_text}."
            f"\n- Joint pv/minutes shared names (ambiguous on their own): {', '.join(sorted(_joint_shared_fields()))}."
        )
    return (
        "Runtime registry constraints for this run:\n"
        f"- {factor_data_domains.describe_factor_domains(domains)}\n"
        "- This allowlist is the source of truth for this run; ignore any stale operator names or examples elsewhere in the prompt.\n"
        "- All operators in the allowlist are equally available; do not treat any listed operator as preferred unless the factor description or operator semantics require it.\n"
        f"- Only use operators from this allowlist: {operator_allowlist}\n"
        f"- Only use fields from this allowlist: {allowlist}"
        f"{signature_note}"
        f"{semantic_note}"
        f"{minute_runtime_note}"
        f"{joint_fields_note}"
    )


def build_runtime_field_constraints() -> str:
    return build_runtime_registry_constraints()
