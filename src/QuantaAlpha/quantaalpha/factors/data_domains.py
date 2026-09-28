from __future__ import annotations

import os
import re
from typing import Iterable

SUPPORTED_FACTOR_DOMAINS = ("pv", "fundamental", "minutes")

FACTOR_MODE_ALIASES = {
    "daily": "daily",
    "day": "daily",
    "pv": "daily",
    "price_volume": "daily",
    "price-volume": "daily",
    "pricevolume": "daily",
    "fundamental": "fundamental",
    "fundamentals": "fundamental",
    "fund": "fundamental",
    "funda": "fundamental",
    "minutes": "minutes",
    "minute": "minutes",
    "intraday": "minutes",
    "min": "minutes",
    "1m": "minutes",
}

AUTO_DOMAIN_TOKENS = frozenset({"", "auto", "default", "follow_mode", "follow-mode"})

PV_FIELDS = (
    "hfq_opens",
    "volumes",
    "hfq_closes",
    "hfq_highs",
    "hfq_lows",
    "vwaps",
    "turnovers",
)

FUNDAMENTAL_FIELDS = (
    "asset_growth",
    "asset_turnover",
    "bp",
    "cash_coverage",
    "consecutive_div_years",
    "cost_ratio_trend",
    "div_yield_stability",
    "div_yield_ttm",
    "div_yield_yoy",
    "earnings_growth",
    "ep",
    "ff_factors",
    "ff_factors_rolling",
    "leverage",
    "long_term_debt_ratio",
    "net_margin",
    "operating_cost_ratio",
    "payout_ratio",
    "profit_margin",
    "revenue_growth",
    "roe",
    "roe_stability",
    "roe_trend",
)

FUNDAMENTAL_FIELD_DESCRIPTIONS = {
    "asset_growth": "asset growth; expansion speed of the balance sheet, useful as a growth or capital-discipline signal",
    "asset_turnover": "asset turnover; revenue generated per unit of assets, a capital-efficiency signal",
    "bp": "book-to-price ratio; valuation signal where higher values generally mean cheaper book valuation",
    "cash_coverage": "cash-flow coverage; ability of cash generation to cover obligations or payouts",
    "consecutive_div_years": "consecutive dividend years; dividend continuity and shareholder-return discipline",
    "cost_ratio_trend": "cost-ratio trend; operating cost pressure or improvement over time",
    "div_yield_stability": "dividend-yield stability; persistence of dividend yield rather than one-off payout spikes",
    "div_yield_ttm": "trailing twelve-month dividend yield; direct shareholder yield signal",
    "div_yield_yoy": "year-over-year dividend-yield change; improving or deteriorating payout signal",
    "earnings_growth": "earnings growth; profitability growth persistence",
    "ep": "earnings-to-price ratio; valuation signal where higher values generally mean cheaper earnings yield",
    "ff_factors": "Fama-French style fundamental factor composite",
    "ff_factors_rolling": "rolling Fama-French style factor composite; smoother time-varying style exposure",
    "leverage": "financial leverage; balance-sheet risk and financing pressure",
    "long_term_debt_ratio": "long-term debt ratio; long-horizon debt burden",
    "net_margin": "net profit margin; bottom-line profitability quality",
    "operating_cost_ratio": "operating cost ratio; cost efficiency and expense burden",
    "payout_ratio": "payout ratio; dividend payout intensity relative to earnings",
    "profit_margin": "profit margin; broad profitability and pricing-power signal",
    "revenue_growth": "revenue growth; top-line demand and growth persistence",
    "roe": "return on equity; shareholder capital profitability",
    "roe_stability": "ROE stability; persistence and reliability of profitability",
    "roe_trend": "ROE trend; improving or deteriorating profitability trajectory",
}

MINUTE_FIELDS = (
    "opens",
    "highs",
    "lows",
    "closes",
    "volumes",
    "amounts",
    "ratios",
    "vwaps",
    "turnovers",
    "returns",
    "lreturns",
)

DOMAIN_FIELD_ALLOWLISTS = {
    "pv": PV_FIELDS,
    "fundamental": FUNDAMENTAL_FIELDS,
    "minutes": MINUTE_FIELDS,
}

JOINT_PV_MINUTES_SHARED_FIELDS = frozenset(set(PV_FIELDS) & set(MINUTE_FIELDS))
JOINT_PV_DAILY_SHARED_FIELD_ALIASES = {
    "volume": "volumes",
    "vwap": "vwaps",
    "turnover": "turnovers",
}
JOINT_PV_SHARED_TO_SINGULAR = {
    plural: singular
    for singular, plural in JOINT_PV_DAILY_SHARED_FIELD_ALIASES.items()
}
JOINT_PV_EXCLUSIVE_FIELDS = tuple(field for field in PV_FIELDS if field not in JOINT_PV_MINUTES_SHARED_FIELDS)
JOINT_MINUTE_EXCLUSIVE_FIELDS = tuple(field for field in MINUTE_FIELDS if field not in JOINT_PV_MINUTES_SHARED_FIELDS)
JOINT_RAW_SHARED_OHLC_FIELDS = frozenset({"opens", "highs", "lows", "closes"})

DOMAIN_ALIASES = {
    "pv": "pv",
    "price_volume": "pv",
    "price-volume": "pv",
    "pricevolume": "pv",
    "daily": "pv",
    "fundamental": "fundamental",
    "fundamentals": "fundamental",
    "fund": "fundamental",
    "funda": "fundamental",
    "minutes": "minutes",
    "minute": "minutes",
    "intraday": "minutes",
    "min": "minutes",
    "1m": "minutes",
    "combined": "combined",
    "all": "combined",
}


def parse_factor_domains(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ("pv",)

    if isinstance(raw, str):
        tokens = [tok for tok in re.split(r"[\/,+\s]+", raw.strip()) if tok]
    else:
        tokens = [str(tok).strip() for tok in raw if str(tok).strip()]

    if not tokens:
        return ("pv",)

    normalized: list[str] = []
    for token in tokens:
        mapped = DOMAIN_ALIASES.get(token.strip().lower())
        if mapped == "combined":
            normalized.extend(SUPPORTED_FACTOR_DOMAINS)
            continue
        if mapped in SUPPORTED_FACTOR_DOMAINS and mapped not in normalized:
            normalized.append(mapped)

    return tuple(normalized) if normalized else ("pv",)


def normalize_factor_mode(raw: str | None) -> str | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if not token:
        return None
    return FACTOR_MODE_ALIASES.get(token)


def default_domains_for_mode(mode: str | None) -> tuple[str, ...]:
    normalized = normalize_factor_mode(mode) or "daily"
    if normalized == "fundamental":
        return ("fundamental",)
    if normalized == "minutes":
        return ("minutes",)
    return ("pv",)


def resolve_effective_factor_domains(
    raw: str | Iterable[str] | None,
    data_mode: str | None = None,
) -> tuple[str, ...]:
    if raw is None:
        return default_domains_for_mode(data_mode)
    if isinstance(raw, str) and raw.strip().lower() in AUTO_DOMAIN_TOKENS:
        return default_domains_for_mode(data_mode)
    if not isinstance(raw, str):
        items = [str(item).strip() for item in raw if str(item).strip()]
        if not items:
            return default_domains_for_mode(data_mode)
        raw = items
    return parse_factor_domains(raw)


def resolve_factor_domains(factor_cfg: dict | None = None, current_env: dict | None = None) -> tuple[str, ...]:
    env = current_env or os.environ
    raw = env.get("FACTOR_DATA_DOMAINS")
    if raw is None and isinstance(factor_cfg, dict):
        raw = factor_cfg.get("data_domains")
    data_mode = (
        env.get("FACTOR_DATA_MODE")
        or env.get("FACTOR_PROMPT_MODE")
        or (factor_cfg or {}).get("data_mode")
        or (factor_cfg or {}).get("prompt_mode")
    )
    return resolve_effective_factor_domains(raw, data_mode=data_mode)


def get_domain_field_names(domains: Iterable[str] | None = None) -> tuple[str, ...]:
    resolved = parse_factor_domains(domains)
    fields: list[str] = []

    for domain in resolved:
        domain_fields = list(DOMAIN_FIELD_ALLOWLISTS.get(domain, ()))
        if domain == "pv" and "minutes" in resolved:
            domain_fields.extend(JOINT_PV_DAILY_SHARED_FIELD_ALIASES.keys())
        for item in domain_fields:
            if item not in fields:
                fields.append(item)
    return tuple(fields)


def get_joint_aware_field_names(domains: Iterable[str] | None = None) -> tuple[str, ...]:
    return get_domain_field_names(domains)


def canonicalize_joint_pv_daily_aliases(text: str | None) -> str:
    result = str(text or "")
    for singular, plural in sorted(JOINT_PV_DAILY_SHARED_FIELD_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        result = re.sub(rf"\b{re.escape(singular)}\b", plural, result)
    return result


def get_domain_field_map(domains: Iterable[str] | None = None) -> dict[str, tuple[str, ...]]:
    resolved = parse_factor_domains(domains)
    field_map: dict[str, tuple[str, ...]] = {}
    for domain in resolved:
        domain_fields = list(DOMAIN_FIELD_ALLOWLISTS.get(domain, ()))
        if domain == "pv" and "minutes" in resolved:
            domain_fields.extend(JOINT_PV_DAILY_SHARED_FIELD_ALIASES.keys())
        field_map[domain] = tuple(dict.fromkeys(domain_fields))
    return field_map


def validate_multi_domain_field_coverage(
    used_fields: Iterable[str] | None,
    domains: Iterable[str] | None = None,
) -> dict[str, object]:
    resolved = parse_factor_domains(domains)
    used = {str(field).strip() for field in (used_fields or []) if str(field).strip()}
    coverage: dict[str, list[str]] = {}
    missing_domains: list[str] = []

    for domain, domain_fields in get_domain_field_map(resolved).items():
        matched = [field for field in domain_fields if field in used]
        coverage[domain] = matched
        if len(resolved) > 1 and len(matched) == 0:
            missing_domains.append(domain)

    return {
        "ok": len(missing_domains) == 0,
        "domains": list(resolved),
        "used_fields": sorted(used),
        "coverage": coverage,
        "missing_domains": missing_domains,
    }


def describe_factor_domains(domains: Iterable[str] | None = None, field_limit: int = 24) -> str:
    resolved = parse_factor_domains(domains)
    fields = list(get_domain_field_names(resolved))
    preview = ", ".join(fields[:field_limit])
    if len(fields) > field_limit:
        preview = f"{preview}, ..."
    return f"domains={','.join(resolved)}; available_fields={preview or 'none'}"


def describe_joint_pv_minutes_semantics() -> str:
    return (
        "Joint pv/minutes semantics:\n"
        f"- Daily pv explicit fields: {', '.join(JOINT_PV_EXCLUSIVE_FIELDS)}.\n"
        f"- Daily pv shared-field aliases in joint mode: "
        f"{', '.join(f'{alias}->{target}' for alias, target in JOINT_PV_DAILY_SHARED_FIELD_ALIASES.items())}.\n"
        f"- Minute explicit fields: {', '.join(JOINT_MINUTE_EXCLUSIVE_FIELDS)}.\n"
        f"- Shared field names that are ambiguous by themselves: {', '.join(sorted(JOINT_PV_MINUTES_SHARED_FIELDS))}.\n"
        f"- Raw minute OHLC names allowed only on the minute side in joint expressions: {', '.join(sorted(JOINT_RAW_SHARED_OHLC_FIELDS))}.\n"
        "- In joint expressions, do not rely on shared fields alone to prove both domains are present.\n"
        "- Use explicit `hfq_*` fields for the daily pv side. Raw `opens/highs/lows/closes` belong only to the minute side.\n"
        "- Do not emit doubly adjusted names such as `hfq_hfq_closes`; use a single `hfq_*` prefix at most once.\n"
        "- Use `volume`, `vwap`, and `turnover` for daily pv shared-field anchors in joint expressions; runtime maps them back to `volumes`, `vwaps`, and `turnovers`.\n"
        "- `volumes`, `vwaps`, and `turnovers` remain raw minute field names inside minute reducers/subtrees.\n"
        "- Minute-side examples include `returns`, `lreturns`, `vwaps`, `turnovers`, `amounts`, `ratios`, and raw minute OHLC.\n"
        "- After minute data has been reduced to a daily 2D frame (`dates x stocks`), regular daily pv operators may be used on that reduced result."
    )


def validate_joint_pv_minutes_expression_fields(used_fields: Iterable[str] | None) -> dict[str, object]:
    used = {str(field).strip() for field in (used_fields or []) if str(field).strip()}
    joint_daily_aliases = set(JOINT_PV_DAILY_SHARED_FIELD_ALIASES)
    pv_exclusive_used = sorted(used & (set(JOINT_PV_EXCLUSIVE_FIELDS) | joint_daily_aliases))
    minute_exclusive_used = sorted(used & set(JOINT_MINUTE_EXCLUSIVE_FIELDS))
    shared_used = sorted(used & set(JOINT_PV_MINUTES_SHARED_FIELDS))
    errors: list[str] = []

    if not pv_exclusive_used:
        errors.append(
            "joint pv/minutes expressions must include at least one explicit daily pv field "
            f"such as {', '.join((*JOINT_PV_EXCLUSIVE_FIELDS, *sorted(joint_daily_aliases)))}"
        )
    if not minute_exclusive_used:
        errors.append(
            "joint pv/minutes expressions must include at least one explicit minute field "
            f"such as {', '.join(JOINT_MINUTE_EXCLUSIVE_FIELDS)}"
        )

    return {
        "ok": len(errors) == 0,
        "used_fields": sorted(used),
        "pv_exclusive_used": pv_exclusive_used,
        "minute_exclusive_used": minute_exclusive_used,
        "shared_used": shared_used,
        "raw_shared_ohlc_used": sorted(used & set(JOINT_RAW_SHARED_OHLC_FIELDS)),
        "errors": errors,
    }
