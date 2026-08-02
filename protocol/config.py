"""Environment-driven configuration for the protocol module.

This is the only place in ``protocol/`` that reads process environment
variables, so the rest of the package can be tested with plain constructor
arguments. Nothing here imports GigaAM, pyannote, torch, or any part of the
main application -- ``server.py`` is the only integration point.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def protocol_enabled() -> bool:
    """Whether the protocol module should be loaded at all.

    Checked with a plain string compare against the env var so importing
    this module never has side effects -- callers decide what "disabled"
    means for them (skip import, hide UI, skip migration, ...).
    """
    return os.getenv("GIGASCRIBE_PROTOCOL_ENABLED", "0") == "1"


@dataclass(frozen=True)
class ProtocolConfig:
    """Runtime configuration, built once by the integration layer (server.py)."""

    data_dir: Path
    models_dir: Path
    db_path: Path
    default_chunk_minutes: float = 12.0
    min_chunk_minutes: float = 10.0
    max_chunk_minutes: float = 15.0
    chunk_overlap_seconds: float = 30.0
    topic_split_enabled: bool = True
    default_temperature: float = 0.2
    default_max_output_tokens: int = 2048
    max_retries: int = 2
    glossary_suggestions_enabled: bool = True
    default_glossary_scope: str = "global"
    # Item 13: full raw LLM responses can contain confidential meeting
    # content and must never land in protocol.log by default. Off unless
    # explicitly turned on for debugging.
    debug_raw_response: bool = False

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "protocols"

    @classmethod
    def from_env(cls, *, data_dir: Path, models_dir: Path) -> "ProtocolConfig":
        return cls(
            data_dir=data_dir,
            models_dir=models_dir / "protocol",
            db_path=data_dir / "jobs.sqlite3",
            default_chunk_minutes=float(os.getenv("GIGASCRIBE_PROTOCOL_CHUNK_MINUTES", "12")),
            chunk_overlap_seconds=float(os.getenv("GIGASCRIBE_PROTOCOL_CHUNK_OVERLAP_SECONDS", "30")),
            topic_split_enabled=os.getenv("GIGASCRIBE_PROTOCOL_TOPIC_SPLIT", "1") == "1",
            default_temperature=float(os.getenv("GIGASCRIBE_PROTOCOL_TEMPERATURE", "0.2")),
            default_max_output_tokens=int(os.getenv("GIGASCRIBE_PROTOCOL_MAX_OUTPUT_TOKENS", "2048")),
            max_retries=int(os.getenv("GIGASCRIBE_PROTOCOL_MAX_RETRIES", "2")),
            glossary_suggestions_enabled=os.getenv("GIGASCRIBE_PROTOCOL_GLOSSARY_SUGGESTIONS", "1") == "1",
            default_glossary_scope=os.getenv("GIGASCRIBE_PROTOCOL_DEFAULT_GLOSSARY_SCOPE", "global"),
            debug_raw_response=os.getenv("GIGASCRIBE_PROTOCOL_DEBUG_RAW_RESPONSE", "0") == "1",
        )
