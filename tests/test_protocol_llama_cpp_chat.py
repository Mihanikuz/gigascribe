"""llama.cpp chat-completion usage and chat-template validation (item 5).

llama-cpp-python is an optional heavy dependency (requirements-protocol.txt)
that isn't installed in this test environment, so a minimal fake module is
injected into sys.modules -- exactly the boundary LlamaCppProvider.load()
lazily imports across, so this tests the real provider code against a fake
llama_cpp.Llama rather than re-implementing the provider's logic.
"""
import asyncio
import sys
import types

import pytest

from protocol.providers.base import extract_json
from protocol.providers.llama_cpp import LlamaCppProvider
from protocol.schemas import ProtocolValidationError


class _FakeLlama:
    def __init__(self, *, model_path, n_ctx, n_gpu_layers, verbose, metadata, **extra_launch_params):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.metadata = metadata
        self.extra_launch_params = extra_launch_params
        self.calls = []

    def create_chat_completion(self, *, messages, max_tokens, temperature):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


@pytest.fixture()
def fake_llama_cpp_module():
    """Install a fake `llama_cpp` module with the given metadata baked into
    every Llama instance it constructs, and clean it up afterwards so this
    never leaks into other test files."""
    state = {"metadata": {}, "instances": []}

    def _factory(**kwargs):
        instance = _FakeLlama(metadata=state["metadata"], **kwargs)
        state["instances"].append(instance)
        return instance

    module = types.ModuleType("llama_cpp")
    module.Llama = _factory
    sys.modules["llama_cpp"] = module
    try:
        yield state
    finally:
        sys.modules.pop("llama_cpp", None)


def test_llama_cpp_uses_chat_completion_with_system_and_user_roles(fake_llama_cpp_module):
    fake_llama_cpp_module["metadata"] = {"tokenizer.chat_template": "{% for m in messages %}{{ m }}{% endfor %}"}
    provider = LlamaCppProvider(model_path="/fake/model.gguf", context_length=4096, n_gpu_layers=10)

    async def run():
        await provider.load()
        return await provider.generate("расшифруй блок", temperature=0.2, max_tokens=100)

    text = asyncio.run(run())
    assert text == '{"ok": true}'
    instance = fake_llama_cpp_module["instances"][0]
    assert instance.n_gpu_layers == 10
    assert len(instance.calls) == 1
    messages = instance.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "расшифруй блок"


def test_llama_cpp_missing_chat_template_raises_clear_validation_error(fake_llama_cpp_module):
    fake_llama_cpp_module["metadata"] = {}  # no tokenizer.chat_template at all
    provider = LlamaCppProvider(model_path="/fake/model.gguf", context_length=4096)

    async def run():
        await provider.load()

    with pytest.raises(ProtocolValidationError, match="chat_template"):
        asyncio.run(run())


def test_no_think_flag_adds_a_system_instruction_against_reasoning_output(fake_llama_cpp_module):
    fake_llama_cpp_module["metadata"] = {"tokenizer.chat_template": "x"}
    provider = LlamaCppProvider(model_path="/fake/qwen3-8b.gguf", context_length=4096, no_think=True)

    async def run():
        await provider.load()
        await provider.generate("prompt", temperature=0.2, max_tokens=100)

    asyncio.run(run())
    instance = fake_llama_cpp_module["instances"][0]
    system_message = instance.calls[0]["messages"][0]["content"]
    assert "think" in system_message.lower()


def test_think_blocks_are_stripped_before_json_parsing_regardless_of_provider():
    """Defense in depth (item 5): even if a model ignores the no-think
    instruction, reasoning must never reach parsed JSON or the final
    protocol -- extract_json() strips it centrally for every provider."""
    raw = "<think>let me reason about this for a while...</think>\n{\"topic\": \"ok\", \"summary\": \"s\"}"
    result = extract_json(raw)
    assert result == {"topic": "ok", "summary": "s"}
    assert "think" not in str(result)
