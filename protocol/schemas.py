"""Plain dataclasses shared across the protocol module.

No third-party dependency (no pydantic) so the module stays cheap to import
even when disabled. Validation of LLM-produced JSON lives in
``schemas.ChunkAnalysis.from_llm_json`` / ``ProtocolDocument`` below rather
than in the providers, so any engine can be swapped without touching the
validation rules.
"""
from __future__ import annotations

import dataclasses
import re
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


# The model is not always consistent about which field name it uses for a
# bare piece of text inside a list item (a risk might come back as
# {"risk": "..."} or {"text": "..."} depending on the call). Tried in this
# order; a dict with none of these recognizable keys is dropped rather than
# stringified, since str(dict) would leak Python repr syntax like
# "{'text': '...'}" straight into the protocol.
_TEXT_FIELD_CANDIDATES = ("text", "question", "risk", "fact", "topic", "summary", "task", "content", "value", "description")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in _TEXT_FIELD_CANDIDATES:
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProtocolValidationError("expected a list of strings")
    return [t for t in (normalize_text(v) for v in value) if t]


def _dedup_key(text: str) -> str:
    """Normalize for duplicate comparison (item 9): collapse whitespace,
    lowercase, drop a trailing period, fold ё/е -- never mutates the actual
    displayed text, only the comparison key."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = t.rstrip(".")
    t = t.replace("ё", "е")  # ё -> е
    return t


def dedup_preserve_order(items: tuple[str, ...]) -> tuple[str, ...]:
    """Drop duplicates (by normalized text) while keeping first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _dedup_key(item)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(item)
    return tuple(out)


@dataclass(frozen=True)
class DecisionItem:
    text: str
    speaker: str = NOT_SPECIFIED
    timestamp: str = NOT_SPECIFIED
    confidence: float = 0.0
    anchor: Optional[str] = None
    verified: Optional[bool] = None
    verification_reason: str = ""
    # Item 7: the timecode a decision cites may be a single point or a
    # range ("15:12 - 16:00"); these hold the parsed/normalized bounds
    # (empty if `timestamp` couldn't be parsed at all) so JSON/HTML/XML
    # all show the same start/end rather than each re-parsing `timestamp`.
    timestamp_start: str = ""
    timestamp_end: str = ""

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
    anchor: Optional[str] = None
    verified: Optional[bool] = None
    verification_reason: str = ""
    timestamp_start: str = ""
    timestamp_end: str = ""

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


def dedup_items_preserve_order(items: tuple, key_fn) -> tuple:
    """Same as dedup_preserve_order but for DecisionItem/TaskItem-style
    objects, keyed by whatever text field the caller points at."""
    seen: set[str] = set()
    out: list = []
    for item in items:
        key = _dedup_key(key_fn(item))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(item)
    return tuple(out)


@dataclass(frozen=True)
class Topic:
    """Item 6: a topic is a title plus an optional short description, never
    a raw nested JSON blob dumped as text."""
    title: str
    summary: str = ""

    @classmethod
    def from_json(cls, raw: Any) -> Optional["Topic"]:
        if isinstance(raw, str):
            title = raw.strip()
            return cls(title=title) if title else None
        if isinstance(raw, dict):
            title = ""
            for key in ("title", "topic", "text", "name"):
                v = raw.get(key)
                if isinstance(v, str) and v.strip():
                    title = v.strip()
                    break
            if not title:
                return None
            summary = ""
            for key in ("summary", "description"):
                v = raw.get(key)
                if isinstance(v, str) and v.strip():
                    summary = v.strip()
                    break
            return cls(title=title, summary=summary)
        return None


@dataclass(frozen=True)
class TermSuggestion:
    """One glossary candidate proposed by the chunk-analysis LLM call.

    Never applied automatically -- service.py turns qualifying entries into
    `glossary_suggestions` rows with status "proposed" only.
    """
    detected: str
    suggested: str
    context: str = ""
    timestamp: str = ""
    confidence: float = 0.0

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Optional["TermSuggestion"]:
        if not isinstance(raw, dict):
            return None
        detected = str(raw.get("detected") or "").strip()
        suggested = str(raw.get("suggested") or "").strip()
        if not detected or not suggested:
            return None
        return cls(
            detected=detected, suggested=suggested,
            context=str(raw.get("context") or ""), timestamp=str(raw.get("timestamp") or ""),
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
    terms: tuple[TermSuggestion, ...] = ()

    @classmethod
    def from_llm_json(cls, chunk_index: int, raw: dict[str, Any]) -> "ChunkAnalysis":
        if not isinstance(raw, dict):
            raise ProtocolValidationError("chunk analysis must be a JSON object")
        raw_terms = raw.get("terms") or []
        if not isinstance(raw_terms, list):
            raw_terms = []
        terms = tuple(t for t in (TermSuggestion.from_json(r) for r in raw_terms) if t is not None)
        return cls(
            chunk_index=chunk_index,
            topic=str(raw.get("topic") or ""),
            summary=str(raw.get("summary") or ""),
            facts=tuple(as_str_list(raw.get("facts"))),
            decisions=tuple(DecisionItem.from_json(d) for d in (raw.get("decisions") or [])),
            tasks=tuple(TaskItem.from_json(t) for t in (raw.get("tasks") or [])),
            open_questions=tuple(as_str_list(raw.get("open_questions"))),
            risks=tuple(as_str_list(raw.get("risks"))),
            disagreements=tuple(as_str_list(raw.get("disagreements"))),
            terms=terms,
        )


# --- final protocol document (Stage 4 output / section 6 structure) ----

@dataclass(frozen=True)
class TimestampRef:
    label: str
    timestamp: str
    speaker: str = NOT_SPECIFIED
    chunk_index: int = -1
    anchor: str = ""
    source_text: str = ""
    element_type: str = "chunk"  # "chunk" | "decision" | "task"


@dataclass(frozen=True)
class ProtocolDocument:
    meeting_title: str
    processed_at: str
    source_filename: str
    duration_seconds: float
    participants: tuple[str, ...]
    summary: str
    topics: tuple[Topic, ...]
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
    # Reproducibility snapshot (item 11): everything needed to explain how
    # this specific protocol was produced. Never includes secrets/tokens.
    engine: str = ""
    model_path: str = ""
    context_length: int = 0
    temperature: float = 0.0
    max_output_tokens: int = 2048
    chunking_settings: dict[str, Any] = field(default_factory=dict)
    app_commit: str = ""
    generated_at: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# --- fact-check (Stage between merge and final document) ----------------

FACT_CHECK_TYPES = ("decision", "task")


@dataclass(frozen=True)
class FactCheckVerdict:
    type: str
    index: int
    confirmed: bool
    reason: str = ""
    source_timestamp: str = ""
    source_speaker: str = ""

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Optional["FactCheckVerdict"]:
        if not isinstance(raw, dict):
            return None
        t = str(raw.get("type") or "")
        if t not in FACT_CHECK_TYPES:
            return None
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None
        return cls(
            type=t, index=index, confirmed=bool(raw.get("confirmed")),
            reason=str(raw.get("reason") or ""), source_timestamp=str(raw.get("source_timestamp") or ""),
            source_speaker=str(raw.get("source_speaker") or ""),
        )


@dataclass
class ProtocolOptions:
    model_id: Optional[str] = None
    chunk_minutes: Optional[float] = None
    topic_split: Optional[bool] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_retries: Optional[int] = None
    glossary_project: Optional[str] = None
    glossary_scope: Optional[str] = None
    requested_by: Optional[str] = None


ENGINES = ("llama_cpp", "ollama")


@dataclass(frozen=True)
class SettingsSnapshot:
    """Everything a protocol job needs, frozen at creation time.

    Built once by ProtocolService.create_protocol() from (in priority
    order) the per-request options, the admin-tunable protocol_settings
    table, the selected model's own parameters, and finally ProtocolConfig
    defaults. Once written to protocol_jobs.settings_snapshot, later admin
    changes can never affect an already-created job -- _run() reads only
    this snapshot, never the live config/settings/model row again.
    """
    model_id: str
    engine: str
    model_path: str
    context_length: int
    temperature: float
    max_output_tokens: int
    chunk_minutes: float
    chunk_overlap_seconds: float
    topic_split_enabled: bool
    max_retries: int
    glossary_suggestions_enabled: bool
    default_glossary_scope: str
    glossary_project: Optional[str]
    prompt_versions: dict[str, int]
    created_at: float
    ollama_url: Optional[str] = None
    ollama_keep_alive: Optional[str] = None
    n_gpu_layers: Optional[int] = None
    extra_launch_params: dict[str, Any] = field(default_factory=dict)
    app_commit: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "SettingsSnapshot":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def validate_settings_snapshot(s: SettingsSnapshot) -> None:
    """Raise ProtocolValidationError listing every violated bound at once."""
    errors: list[str] = []
    if not (10 <= s.chunk_minutes <= 15):
        errors.append("chunk_minutes must be between 10 and 15")
    if s.chunk_overlap_seconds < 0:
        errors.append("chunk_overlap_seconds must be non-negative")
    if not (0 <= s.temperature <= 2):
        errors.append("temperature must be between 0 and 2")
    if not (0 <= s.max_retries <= 5):
        errors.append("max_retries must be between 0 and 5")
    if not (isinstance(s.context_length, int) and s.context_length > 0):
        errors.append("context_length must be a positive integer")
    if not (isinstance(s.max_output_tokens, int) and s.max_output_tokens > 0):
        errors.append("max_output_tokens must be a positive integer")
    elif isinstance(s.context_length, int) and s.context_length > 0:
        # item 4: context_length must leave real room for input beyond the
        # reserved output budget, not just be numerically larger than it.
        min_input_reserve = 512
        if s.context_length - s.max_output_tokens < min_input_reserve:
            errors.append(
                f"context_length ({s.context_length}) leaves too little room for input given "
                f"max_output_tokens ({s.max_output_tokens}); need at least {min_input_reserve} tokens free"
            )
    if s.default_glossary_scope not in GLOSSARY_SCOPES:
        errors.append(f"default_glossary_scope must be one of {GLOSSARY_SCOPES}")
    if s.engine not in ENGINES:
        errors.append(f"engine must be one of {ENGINES}")
    if not s.model_path:
        errors.append("model is not configured (missing path/tag)")
    if errors:
        raise ProtocolValidationError("; ".join(errors))


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
    # Engine config (item 4): llama_cpp uses n_gpu_layers/extra_launch_params,
    # ollama uses ollama_url/ollama_keep_alive. Both sets of fields exist on
    # every model row; only the ones matching `engine` are actually used.
    n_gpu_layers: int = -1
    extra_launch_params: dict[str, Any] = field(default_factory=dict)
    ollama_url: Optional[str] = None
    ollama_keep_alive: Optional[str] = None
    # Qwen3 GGUF builds emit <think>...</think> reasoning by default; item 5
    # requires the final JSON/protocol never contain it.
    no_think: bool = False


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
    chunk_index: Optional[int] = None
    model_id: Optional[str] = None


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
