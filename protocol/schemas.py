"""Plain dataclasses shared across the protocol module.

No third-party dependency (no pydantic) so the module stays cheap to import
even when disabled. Validation of LLM-produced JSON lives in
``schemas.ChunkAnalysis.from_llm_json`` / ``ProtocolDocument`` below rather
than in the providers, so any engine can be swapped without touching the
validation rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

NOT_SPECIFIED = "не указан"

# --- transcript-side ---------------------------------------------------

@dataclass(frozen=True)
class TranscriptSegment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    start: float
    end: float
    segments: tuple[TranscriptSegment, ...]
    topic_hint: Optional[str] = None

    def to_prompt_text(self) -> str:
        lines = [f"{s.speaker} [{format_seconds(s.start)} - {format_seconds(s.end)}]: {s.text}" for s in self.segments]
        return "\n".join(lines)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 3600:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --- LLM chunk-analysis JSON (Stage 3) ----------------------------------

class ProtocolValidationError(ValueError):
    """Raised when an LLM response fails schema validation."""


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProtocolValidationError("expected a list of strings")
    return [str(v) for v in value]


@dataclass(frozen=True)
class DecisionItem:
    text: str
    speaker: str = NOT_SPECIFIED
    timestamp: str = NOT_SPECIFIED
    confidence: float = 0.0

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "DecisionItem":
        if "text" not in raw or not str(raw["text"]).strip():
            raise ProtocolValidationError("decision item missing 'text'")
        return cls(
            text=str(raw["text"]),
            speaker=str(raw.get("speaker") or NOT_SPECIFIED),
            timestamp=str(raw.get("timestamp") or NOT_SPECIFIED),
            confidence=_clamp_confidence(raw.get("confidence")),
        )


@dataclass(frozen=True)
class TaskItem:
    task: str
    owner: str = NOT_SPECIFIED
    deadline: str = NOT_SPECIFIED
    speaker: str = NOT_SPECIFIED
    timestamp: str = NOT_SPECIFIED
    confidence: float = 0.0

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "TaskItem":
        if "task" not in raw or not str(raw["task"]).strip():
            raise ProtocolValidationError("task item missing 'task'")
        return cls(
            task=str(raw["task"]),
            owner=str(raw.get("owner") or NOT_SPECIFIED),
            deadline=str(raw.get("deadline") or NOT_SPECIFIED),
            speaker=str(raw.get("speaker") or NOT_SPECIFIED),
            timestamp=str(raw.get("timestamp") or NOT_SPECIFIED),
            confidence=_clamp_confidence(raw.get("confidence")),
        )


def _clamp_confidence(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


@dataclass(frozen=True)
class ChunkAnalysis:
    chunk_index: int
    topic: str = ""
    summary: str = ""
    facts: tuple[str, ...] = ()
    decisions: tuple[DecisionItem, ...] = ()
    tasks: tuple[TaskItem, ...] = ()
    open_questions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    disagreements: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()

    @classmethod
    def from_llm_json(cls, chunk_index: int, raw: dict[str, Any]) -> "ChunkAnalysis":
        if not isinstance(raw, dict):
            raise ProtocolValidationError("chunk analysis must be a JSON object")
        return cls(
            chunk_index=chunk_index,
            topic=str(raw.get("topic") or ""),
            summary=str(raw.get("summary") or ""),
            facts=tuple(_as_str_list(raw.get("facts"))),
            decisions=tuple(DecisionItem.from_json(d) for d in (raw.get("decisions") or [])),
            tasks=tuple(TaskItem.from_json(t) for t in (raw.get("tasks") or [])),
            open_questions=tuple(_as_str_list(raw.get("open_questions"))),
            risks=tuple(_as_str_list(raw.get("risks"))),
            disagreements=tuple(_as_str_list(raw.get("disagreements"))),
            terms=tuple(_as_str_list(raw.get("terms"))),
        )


# --- final protocol document (Stage 4 output / section 6 structure) ----

@dataclass(frozen=True)
class TimestampRef:
    label: str
    timestamp: str
    speaker: str = NOT_SPECIFIED
    chunk_index: int = -1


@dataclass(frozen=True)
class ProtocolDocument:
    meeting_title: str
    processed_at: str
    source_filename: str
    duration_seconds: float
    participants: tuple[str, ...]
    summary: str
    topics: tuple[str, ...]
    decisions: tuple[DecisionItem, ...]
    tasks: tuple[TaskItem, ...]
    owners: tuple[str, ...]
    deadlines: tuple[str, ...]
    open_questions: tuple[str, ...]
    risks: tuple[str, ...]
    disagreements: tuple[str, ...]
    next_steps: tuple[str, ...]
    unverified_items: tuple[str, ...]
    timestamp_refs: tuple[TimestampRef, ...]
    model_id: str
    prompt_versions: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ProtocolOptions:
    model_id: Optional[str] = None
    chunk_minutes: Optional[float] = None
    topic_split: Optional[bool] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    glossary_project: Optional[str] = None
    glossary_scope: Optional[str] = None
    requested_by: Optional[str] = None


@dataclass
class ProtocolResult:
    protocol_job_id: str
    status: str
    document: Optional[ProtocolDocument]
    json_path: Optional[str]
    html_path: Optional[str]


# --- model / prompt registry --------------------------------------------

@dataclass
class ModelSpec:
    id: str
    label: str
    engine: str  # "llama_cpp" | "ollama"
    repo_id: Optional[str] = None
    filename: Optional[str] = None
    local_path: Optional[str] = None
    format: str = "gguf"
    quantization: str = "Q4_K_M"
    size_bytes: int = 0
    context_length: int = 8192
    temperature: float = 0.2
    max_output_tokens: int = 2048
    system_prompt_override: Optional[str] = None
    installed: bool = False
    last_check_status: Optional[str] = None
    last_check_at: Optional[float] = None
    updated_at: Optional[float] = None


PROMPT_KINDS = ("chunk_analysis", "topic_split", "merge", "fact_check", "html_template")


@dataclass
class PromptSpec:
    kind: str
    content: str
    version: int
    is_default: bool
    is_active: bool
    model_id: Optional[str] = None
    updated_by: Optional[str] = None
    updated_at: Optional[float] = None


# --- glossary --------------------------------------------------------

GLOSSARY_SCOPES = ("global", "department", "project", "job")
GLOSSARY_SCOPE_PRIORITY = ("job", "project", "department", "global")


@dataclass
class GlossaryTerm:
    id: Optional[int]
    canonical: str
    aliases: tuple[str, ...]
    category: str = ""
    context: str = ""
    scope: str = "global"
    project: Optional[str] = None
    status: str = "confirmed"
    usage_count: int = 0
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


@dataclass
class GlossarySuggestion:
    id: Optional[int]
    source: str  # "user_correction" | "llm_proposal" | "document_diff"
    wrong_text: str
    suggested_text: str
    context: str = ""
    timestamp: str = ""
    confidence: float = 0.0
    job_id: Optional[str] = None
    status: str = "proposed"
    created_at: Optional[float] = None


@dataclass
class GlossaryCorrection:
    id: Optional[int]
    job_id: Optional[str]
    original_text: str
    corrected_text: str
    rule: str
    source: str
    timestamp: str = ""
    confidence: float = 0.0
    created_at: Optional[float] = None
