"""LLMProvider base helpers: JSON extraction and retry-on-invalid-JSON (item 13)."""
import asyncio

import pytest

from protocol.providers.base import LLMProvider, extract_json
from protocol.schemas import ProtocolValidationError


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_markdown_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    assert extract_json('Вот твой JSON:\n```\n{"a": 1, "b": [1,2]}\n```\nСпасибо') == {"a": 1, "b": [1, 2]}


def test_extract_json_rejects_non_json():
    with pytest.raises(ProtocolValidationError):
        extract_json("not json at all")


class _RecordingProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__(model_path="fake", context_length=8192)
        self._responses = list(responses)
        self.calls = 0

    async def load(self):
        self.loaded = True

    async def generate(self, prompt, *, temperature, max_tokens, system_prompt=None):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]

    async def unload(self):
        self.loaded = False


def test_generate_json_recovers_after_invalid_response():
    provider = _RecordingProvider(["not json", '{"topic": "ok"}'])

    async def run():
        await provider.load()
        return await provider.generate_json("prompt", temperature=0.2, max_tokens=100, max_retries=2)

    result = asyncio.run(run())
    assert result == {"topic": "ok"}
    assert provider.calls == 2


def test_generate_json_gives_up_after_max_retries():
    provider = _RecordingProvider(["still bad"])

    async def run():
        await provider.load()
        await provider.generate_json("prompt", temperature=0.2, max_tokens=100, max_retries=2)

    with pytest.raises(ProtocolValidationError):
        asyncio.run(run())
    assert provider.calls == 3  # initial attempt + 2 retries
