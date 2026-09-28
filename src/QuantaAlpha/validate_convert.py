#!/usr/bin/env python3
"""Validate all direction conversions work correctly."""
import json
import sys
sys.path.insert(0, "/home/workspace/users/liwei/tqstrategyserver/QuantaAlpha")
from convert_directions import convert_pv_description, convert_minutes_description

# Check PV
with open("/home/workspace/users/liwei/tqstrategyserver/QuantaAlpha/experiment/original_direction_daily.json") as f:
    pv = json.load(f)
pv_ok, pv_fail = 0, 0
for item in pv["factor_portfolios"]:
    old = item["description"]
    new = convert_pv_description(old)
    if new != old and "factors:" in new:
        pv_ok += 1
    else:
        pv_fail += 1
        print(f"PV UNCHANGED #{item['portfolio_id']}: {old[:120]}")
print(f"PV: {pv_ok} converted, {pv_fail} unchanged out of {len(pv['factor_portfolios'])}")

# Check Minutes
with open("/home/workspace/users/liwei/tqstrategyserver/QuantaAlpha/experiment/original_direction_minutes.json") as f:
    mins = json.load(f)
m_ok, m_fail = 0, 0
for item in mins["factor_portfolios"]:
    old = item["description"]
    new = convert_minutes_description(old)
    if new != old and "factors:" in new:
        m_ok += 1
    else:
        m_fail += 1
        print(f"MIN UNCHANGED #{item['portfolio_id']}: {old[:120]}")
print(f"Minutes: {m_ok} converted, {m_fail} unchanged out of {len(mins['factor_portfolios'])}")
