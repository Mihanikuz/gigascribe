"""GigaScribe protocol (meeting-minutes) module.

Fully isolated from GigaAM/pyannote: this package is never imported unless
``GIGASCRIBE_PROTOCOL_ENABLED=1`` (see config.protocol_enabled()), and even
when it is, it only receives a transcript path, a job id, and a couple of
generic callables (a GPU lock, an "unload the other GPU workload" callback)
from the integration layer in server.py -- nothing here imports app.py,
model_manager.py, asr_backend.py, or diarization_backend.py.
"""
from __future__ import annotations

from .config import ProtocolConfig, protocol_enabled
from .schemas import ProtocolOptions, ProtocolResult
from .service import ProtocolService

__all__ = ["ProtocolConfig", "protocol_enabled", "ProtocolOptions", "ProtocolResult", "ProtocolService"]
