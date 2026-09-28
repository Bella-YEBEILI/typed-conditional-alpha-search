import json
from types import MethodType

from quantaalpha.llm.client import APIBackend


def test_auto_continue_skips_json_postprocess_for_fragments_and_ignores_empty_fences():
    client = APIBackend.__new__(APIBackend)
    calls = []
    responses = [
        ('{"a":', "length"),
        ("```", "stop"),
        ("1}", "stop"),
    ]

    def fake_inner(self, messages, **kwargs):
        calls.append(dict(kwargs))
        return responses.pop(0)

    client._create_chat_completion_inner_function = MethodType(fake_inner, client)

    result = APIBackend._create_chat_completion_auto_continue(
        client,
        messages=[{"role": "user", "content": "return json"}],
        json_mode=True,
    )

    assert json.loads(result) == {"a": 1}
    assert calls[1]["skip_json_postprocess"] is True
    assert calls[2]["skip_json_postprocess"] is True
