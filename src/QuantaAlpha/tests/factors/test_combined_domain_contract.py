from quantaalpha.factors.combined_domain_contract import validate_joint_pv_minutes_contract


def test_joint_contract_rejects_raw_daily_ohlc_via_data_ctx():
    module_text = """
TYPE = "regular"
META = {"level": "minutes"}
SETTING = {"data_needed": ["opens"]}

def prepare_minute_datas():
    return {"ret": 1}

def calc_factor(data_ctx, minute_ctx):
    return data_ctx["opens"]
"""
    errors = validate_joint_pv_minutes_contract(module_text, active_domains=("pv", "minutes"))
    assert any("raw daily OHLC" in error for error in errors)


def test_joint_contract_rejects_double_hfq_field_names():
    module_text = """
TYPE = "regular"
META = {"level": "minutes"}
SETTING = {"data_needed": ["hfq_hfq_closes"]}

def prepare_minute_datas():
    return {"ret": 1}

def calc_factor(data_ctx, minute_ctx):
    return data_ctx["hfq_hfq_closes"]
"""
    errors = validate_joint_pv_minutes_contract(module_text, active_domains=("pv", "minutes"))
    assert any("doubly adjusted" in error for error in errors)


def test_joint_contract_preserves_duplicate_minute_engine_inputs_for_arity():
    module_text = """
TYPE = "regular"
META = {"level": "minutes"}
SETTING = {"data_needed": ["hfq_closes"]}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    same_field_corr = mfe.run(
        inputs=["returns", "returns"],
        operator=("ts_corr", {"n": 60}),
    )
    return {"same_field_corr": same_field_corr}

def calc_factor(data_ctx, minute_ctx):
    return minute_ctx["same_field_corr"] * data_ctx["hfq_closes"]
"""
    errors = validate_joint_pv_minutes_contract(module_text, active_domains=("pv", "minutes"))
    assert not any("operator `ts_corr` expects 2 input field" in error for error in errors)
