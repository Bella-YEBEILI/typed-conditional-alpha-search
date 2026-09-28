from quant.quant_lib.quantEnum import UniverseType

DEFAULT_PROFILE_ID = "standards_o1_o2"

PROFILES = {
    "standards_o1_o2":{
        "id":"standards_o1_o2",
        "universe":"standards",
        "buypoint":"opens",
        "buylag":1,
        "sellpoint":"opens",
        "selllag":2,
        "qt":0.2,
    },
    "standards_c1_c2":{
        "id":"standards_c1_c2",
        "universe":"standards",
        "buypoint":"closes",
        "buylag":1,
        "sellpoint":"closes",
        "selllag":2,
        "qt":0.2,
    },
    "hl_o1_o2":{
        "id":"hl_o1_o2",
        "universe":UniverseType.dividend,
        "buypoint":"opens",
        "buylag":1,
        "sellpoint":"opens",
        "selllag":2,
        "qt":0.2,
    },
    "top1800s_o1_o2":{
        "id":"top1800s_o1_o2",
        "universe":"top1800s",
        "buypoint":"opens",
        "buylag":1,
        "sellpoint":"opens",
        "selllag":2,
        "qt":0.2,
    }
}