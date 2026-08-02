"""ProtocolService: the one stable entry point the main app calls.

Owns the whole create-protocol pipeline (settings snapshot, GPU handover,
chunking, per-chunk analysis, merge, fact-check, render) and never touches
the main `jobs` table or raises in a way that could fail the caller's
transcription job -- every error is caught, logged to protocol.log, and
recorded on the protocol_jobs row only.
"""
from __future__ import annotations

import asyncio
import dataclasses
import gc
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from . import glossary as glossary_mod
from .chunking import (
    choose_chunks, enforce_context_budget, normalize_segments, parse_timestamp, parse_timestamp_range,
    parse_transcript, split_by_time_windows,
)
from .config import ProtocolConfig
from .models import SUPPORTED_PROTOCOL_MODELS
from .providers import create_provider, validate_engine_config
from .providers.base import LLMProvider
from .renderer import document_to_json_text, render_html, render_xml
from .schemas import (
    PROMPT_KINDS, ChunkAnalysis, DecisionItem, FactCheckVerdict, NOT_SPECIFIED, ProtocolDocument, ProtocolOptions,
    ProtocolResult, ProtocolValidationError, SettingsSnapshot, TaskItem, TimestampRef, Topic, TranscriptSegment,
    as_str_list, dedup_items_preserve_order, dedup_preserve_order, format_seconds, normalize_text,
    validate_settings_snapshot,
)
from .store import ProtocolStore

logger = logging.getLogger(__name__)

# unload_asr may be sync (ModelManager.unload_all) or async; both supported.
UnloadCallable = Callable[[], Any]

# Nearest-segment tolerance for linking a decision/task timestamp to a real
# transcript moment (item 8): beyond this, the timestamp is shown as plain
# text instead of a link and the item is flagged in unverified_items.
_TIMESTAMP_MATCH_TOLERANCE_SECONDS = 30.0


class ProtocolDisabledError(RuntimeError):
    pass


def _get_app_commit() -> str:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


class ProtocolService:
    def __init__(self, *, config: ProtocolConfig, store: ProtocolStore, gpu_lock: asyncio.Lock,
                 unload_asr: UnloadCallable):
        self.config = config
        self.store = store
        self.gpu_lock = gpu_lock
        self.unload_asr = unload_asr
        self._app_commit = _get_app_commit()
        for spec in SUPPORTED_PROTOCOL_MODELS.values():
            self.store.seed_model(spec)

    # ---- public API -----------------------------------------------------

    async def create_protocol(self, *, transcript_path: Path, job_id: str, username: str,
                               options: Optional[ProtocolOptions] = None) -> ProtocolResult:
        options = options or ProtocolOptions()
        existing = self.store.get_active_protocol_job_for_job(job_id)
        if existing:
            return ProtocolResult(protocol_job_id=existing["id"], status=existing["status"],
                                   document=None, json_path=existing.get("json_path"), html_path=existing.get("html_path"))
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        model_id = options.model_id or self.store.active_model_id()
        if not model_id:
            raise ValueError("No protocol model selected/installed")
        model_state = self.store.get_model_state(model_id)
        if not model_state or not model_state.get("installed"):
            raise ValueError(f"Protocol model is not installed: {model_id}")

        snapshot, prompt_contents = self._build_settings_snapshot(options=options, model_id=model_id, model_state=model_state)
        validate_settings_snapshot(snapshot)
        validate_engine_config(engine=snapshot.engine, model_path=snapshot.model_path, ollama_url=snapshot.ollama_url)

        protocol_job_id = uuid.uuid4().hex
        result_dir = self.config.results_dir / job_id / protocol_job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = result_dir / "protocol.log"
        self.store.create_protocol_job(id=protocol_job_id, job_id=job_id, username=username, model_id=model_id,
                                        settings_snapshot=snapshot.to_json_dict())
        self.store.update_protocol_job(protocol_job_id, log_path=str(log_path))
        asyncio.create_task(self._run(protocol_job_id, transcript_path, job_id, result_dir, snapshot, prompt_contents))
        return ProtocolResult(protocol_job_id=protocol_job_id, status="queued", document=None,
                               json_path=None, html_path=None)

    async def retry(self, protocol_job_id: str, transcript_path: Path, job_id: str) -> ProtocolResult:
        job = self.store.get_protocol_job(protocol_job_id)
        if not job:
            raise ValueError("Unknown protocol job")
        if job["status"] not in ("failed", "cancelled"):
            raise ValueError("Only a failed or cancelled protocol job can be retried")
        result_dir = Path(job["log_path"]).parent if job.get("log_path") else self.config.results_dir / job_id / protocol_job_id
        # A retry replays the *original* settings snapshot -- an admin
        # tweaking settings after this job was created must not change what
        # a retry of it does, same as it can't change the job's first run.
        snapshot = SettingsSnapshot.from_json_dict(job.get("settings_snapshot") or {})
        prompt_contents = {
            kind: (self.store.get_prompt_by_version(kind, snapshot.model_id, snapshot.prompt_versions.get(kind, 0))
                   or self.store.get_active_prompt(kind, snapshot.model_id)).content
            for kind in PROMPT_KINDS
        }
        self.store.update_protocol_job(protocol_job_id, status="queued", progress=0, message="В очереди",
                                        error=None, cancel_requested=0, attempts=job["attempts"] + 1)
        asyncio.create_task(self._run(protocol_job_id, transcript_path, job_id, result_dir, snapshot, prompt_contents))
        return ProtocolResult(protocol_job_id=protocol_job_id, status="queued", document=None, json_path=None, html_path=None)

    def request_cancel(self, protocol_job_id: str) -> None:
        self.store.update_protocol_job(protocol_job_id, cancel_requested=1)

    def resume_after_restart(self) -> int:
        """Recover interrupted protocol jobs after a server restart: never
        leave one in an in-progress state (that would imply a phantom GPU
        hold that no longer exists), and never re-launch it automatically
        -- the user retries explicitly, matching how the main job queue
        distinguishes 'was running' from 'is queued'.
        """
        resumed = 0
        for job in self.store.list_active_protocol_jobs():
            self.store.update_protocol_job(job["id"], status="failed",
                                            error="Interrupted by server restart",
                                            message="Прервано перезапуском сервера", finished_at=time.time())
            resumed += 1
        return resumed

    # ---- settings snapshot (item 1) ---------------------------------------

    def _build_settings_snapshot(self, *, options: ProtocolOptions, model_id: str,
                                  model_state: dict[str, Any]) -> tuple[SettingsSnapshot, dict[str, str]]:
        """Priority: request options > protocol_settings (admin) > selected
        model's own parameters > ProtocolConfig defaults. Frozen once here;
        _run() never re-reads live config/settings/model rows again."""
        stored = self.store.all_settings()
        cfg = self.config

        def _f(request_val, key, model_val, default_val) -> float:
            if request_val is not None:
                return float(request_val)
            if stored.get(key) not in (None, ""):
                try:
                    return float(stored[key])
                except (TypeError, ValueError):
                    pass
            if model_val is not None:
                return float(model_val)
            return float(default_val)

        def _i(request_val, key, model_val, default_val) -> int:
            return int(_f(request_val, key, model_val, default_val))

        def _b(request_val, key, default_val) -> bool:
            if request_val is not None:
                return bool(request_val)
            if key in stored and stored[key] not in (None, ""):
                return str(stored[key]).strip().lower() in ("1", "true", "yes", "on")
            return bool(default_val)

        def _s(request_val, key, default_val) -> str:
            if request_val:
                return str(request_val)
            if stored.get(key):
                return str(stored[key])
            return str(default_val)

        engine = _s(None, "engine", model_state.get("engine") or "llama_cpp")
        prompt_versions: dict[str, int] = {}
        prompt_contents: dict[str, str] = {}
        for kind in PROMPT_KINDS:
            spec = self.store.get_active_prompt(kind, model_id)
            prompt_versions[kind] = spec.version
            prompt_contents[kind] = spec.content

        snapshot = SettingsSnapshot(
            model_id=model_id, engine=engine, model_path=model_state.get("local_path") or "",
            context_length=_i(None, "context_length", model_state.get("context_length"), 8192),
            temperature=_f(options.temperature, "temperature", model_state.get("temperature"), cfg.default_temperature),
            max_output_tokens=_i(options.max_output_tokens, "max_output_tokens", model_state.get("max_output_tokens"),
                                  cfg.default_max_output_tokens),
            chunk_minutes=_f(options.chunk_minutes, "chunk_minutes", None, cfg.default_chunk_minutes),
            chunk_overlap_seconds=_f(None, "chunk_overlap_seconds", None, cfg.chunk_overlap_seconds),
            topic_split_enabled=_b(options.topic_split, "topic_split_enabled", cfg.topic_split_enabled),
            max_retries=_i(options.max_retries, "max_retries", None, cfg.max_retries),
            glossary_suggestions_enabled=_b(None, "glossary_suggestions_enabled", cfg.glossary_suggestions_enabled),
            default_glossary_scope=_s(options.glossary_scope, "default_glossary_scope", cfg.default_glossary_scope),
            glossary_project=options.glossary_project,
            prompt_versions=prompt_versions, created_at=time.time(),
            ollama_url=model_state.get("ollama_url"), ollama_keep_alive=model_state.get("ollama_keep_alive"),
            n_gpu_layers=model_state.get("n_gpu_layers"), extra_launch_params=model_state.get("extra_launch_params") or {},
            app_commit=self._app_commit,
        )
        return snapshot, prompt_contents

    # ---- pipeline ---------------------------------------------------------

    async def _run(self, protocol_job_id: str, transcript_path: Path, job_id: str,
                    result_dir: Path, snapshot: SettingsSnapshot, prompt_contents: dict[str, str]) -> None:
        log_path = result_dir / "protocol.log"

        def log(line: str) -> None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

        def set_status(status: str, message: str, *, progress: Optional[float] = None, **extra: Any) -> None:
            changes: dict[str, Any] = {"status": status, "message": message, **extra}
            if progress is not None:
                changes["progress"] = round(max(0.0, min(1.0, progress)), 3)
            self.store.update_protocol_job(protocol_job_id, **changes)
            log(f"[{status}] {message}")

        def cancelled() -> bool:
            job = self.store.get_protocol_job(protocol_job_id)
            return bool(job and job["cancel_requested"])

        def on_raw(raw: str) -> None:
            # Item 13: opt-in only. A meeting's raw LLM output can contain
            # confidential content and must never land in protocol.log
            # (which is retained on disk) unless explicitly requested.
            log(f"RAW RESPONSE: {raw}")

        raw_response_hook = on_raw if self.config.debug_raw_response else None

        model_id = snapshot.model_id
        temperature, max_tokens = snapshot.temperature, snapshot.max_output_tokens
        no_think = bool(SUPPORTED_PROTOCOL_MODELS.get(model_id) and SUPPORTED_PROTOCOL_MODELS[model_id].no_think)

        provider: Optional[LLMProvider] = None
        try:
            set_status("queued", "В очереди", progress=0.0)

            raw_transcript = transcript_path.read_text(encoding="utf-8")
            segments = parse_transcript(raw_transcript)
            segments = normalize_segments(segments)
            if not segments:
                raise ProtocolValidationError("Transcript is empty after parsing")

            glossary_terms = self.store.resolved_terms(project=snapshot.glossary_project)
            corrected_text, corrections = glossary_mod.apply_safe_replacements(
                raw_transcript, glossary_terms, job_id=job_id, store=self.store,
            )
            if corrections:
                log(f"Applied {len(corrections)} glossary correction(s)")
                segments = normalize_segments(parse_transcript(corrected_text))

            if cancelled():
                set_status("cancelled", "Отменено"); return

            set_status("waiting_for_gpu", "Ожидание освобождения GPU", progress=0.02)
            async with self.gpu_lock:
                if cancelled():
                    set_status("cancelled", "Отменено"); return

                set_status("unloading_asr", "Выгрузка ASR/диаризации", progress=0.05)
                await self._unload_asr()

                set_status("loading_llm", f"Загрузка модели {model_id}", progress=0.1)
                provider = create_provider(
                    engine=snapshot.engine, model_path=snapshot.model_path, context_length=snapshot.context_length,
                    n_gpu_layers=snapshot.n_gpu_layers, no_think=no_think, extra_launch_params=snapshot.extra_launch_params,
                    ollama_url=snapshot.ollama_url, ollama_keep_alive=snapshot.ollama_keep_alive,
                )
                await provider.load()
                log(f"Loaded provider engine={snapshot.engine} model={model_id}")

                if cancelled():
                    set_status("cancelled", "Отменено"); return

                set_status("splitting", "Разбиение на блоки", progress=0.15)
                topic_boundaries, topic_hints = (None, {})
                if snapshot.topic_split_enabled:
                    topic_boundaries, topic_hints = await self._two_stage_topic_split(
                        provider, segments, prompt_content=prompt_contents["topic_split"],
                        chunk_minutes=snapshot.chunk_minutes, temperature=temperature, max_tokens=max_tokens,
                        context_length=snapshot.context_length, log=log, raw_response_hook=raw_response_hook,
                    )
                chunks = choose_chunks(
                    segments, topic_boundaries=topic_boundaries,
                    window_seconds=snapshot.chunk_minutes * 60, overlap_seconds=snapshot.chunk_overlap_seconds,
                    context_length=snapshot.context_length,
                )
                if topic_boundaries and topic_hints:
                    chunks = [dataclasses.replace(c, topic_hint=_nearest_hint(c.start, topic_hints)) for c in chunks]
                if not chunks:
                    raise ProtocolValidationError("Chunking produced no chunks")
                self.store.update_protocol_job(protocol_job_id, chunk_total=len(chunks), chunk_current=0)
                log(f"Split into {len(chunks)} chunk(s)")

                analyses: list[ChunkAnalysis] = []
                set_status("processing_chunks", f"Обработка блока 1/{len(chunks)}", progress=0.2)
                for chunk in chunks:
                    if cancelled():
                        set_status("cancelled", "Отменено"); return
                    prompt = prompt_contents["chunk_analysis"].replace("{transcript_chunk}", chunk.to_prompt_text())
                    raw = await provider.generate_json(prompt, temperature=temperature, max_tokens=max_tokens,
                                                         max_retries=snapshot.max_retries, on_raw_response=raw_response_hook)
                    analysis = ChunkAnalysis.from_llm_json(chunk.index, raw)
                    analyses.append(analysis)
                    if snapshot.glossary_suggestions_enabled and analysis.terms:
                        created = glossary_mod.propose_terms_from_chunk_analysis(
                            self.store, terms=analysis.terms, job_id=job_id, chunk_index=chunk.index, model_id=model_id,
                        )
                        if created:
                            log(f"Chunk {chunk.index}: {len(created)} glossary suggestion(s) proposed")
                    self.store.update_protocol_job(protocol_job_id, chunk_current=chunk.index + 1)
                    progress = 0.2 + 0.45 * ((chunk.index + 1) / len(chunks))
                    set_status("processing_chunks", f"Обработка блока {chunk.index + 1}/{len(chunks)}", progress=progress)

                if cancelled():
                    set_status("cancelled", "Отменено"); return

                set_status("merging", "Сведение итогового протокола", progress=0.68)
                merge_prompt = prompt_contents["merge"].replace(
                    "{chunk_analyses}", json.dumps([_analysis_to_dict(a) for a in analyses], ensure_ascii=False),
                )
                merged_raw = await provider.generate_json(merge_prompt, temperature=temperature, max_tokens=max_tokens,
                                                            max_retries=snapshot.max_retries, on_raw_response=raw_response_hook)
                if not isinstance(merged_raw, dict):
                    raise ProtocolValidationError("merge response must be a JSON object")
                raw_decisions = [DecisionItem.from_json(d) for d in (merged_raw.get("decisions") or []) if isinstance(d, dict)]
                raw_tasks = [TaskItem.from_json(t) for t in (merged_raw.get("tasks") or []) if isinstance(t, dict)]

                if cancelled():
                    set_status("cancelled", "Отменено"); return

                set_status("fact_checking", "Проверка фактов", progress=0.78)
                fact_check_verdicts, fact_check_notes = await self._fact_check(
                    provider, decisions=raw_decisions, tasks=raw_tasks, segments=segments,
                    summary=str(merged_raw.get("summary") or ""), topics=merged_raw.get("topics") or [],
                    prompt_content=prompt_contents["fact_check"], temperature=temperature, max_tokens=max_tokens,
                    max_retries=snapshot.max_retries, context_length=snapshot.context_length, log=log,
                    raw_response_hook=raw_response_hook,
                )
                # Any failure above (invalid JSON after every retry, provider
                # error) propagates out of this try block: per item 2, a
                # technical fact-check failure must fail the whole job rather
                # than emit an unverified protocol as "ready".

                document = _build_document(
                    merged_raw, segments=segments, chunks=chunks, source_filename=transcript_path.stem,
                    model_id=model_id, snapshot=snapshot, raw_decisions=raw_decisions, raw_tasks=raw_tasks,
                    fact_check_verdicts=fact_check_verdicts, fact_check_notes=fact_check_notes,
                )

                set_status("rendering", "Формирование HTML", progress=0.9)
                # HTML, JSON, and XML are all rendered from this one
                # `document` -- never independently -- so the three always
                # agree (item 11).
                html_text = render_html(document, template=prompt_contents["html_template"])
                json_text = document_to_json_text(document)
                xml_text = render_xml(document)
                json_path = result_dir / "protocol.json"
                html_path = result_dir / "protocol.html"
                xml_path = result_dir / "protocol.xml"
                json_path.write_text(json_text, encoding="utf-8")
                html_path.write_text(html_text, encoding="utf-8")
                xml_path.write_text(xml_text, encoding="utf-8")
                self.store.save_result(protocol_job_id, document.to_json_dict(), model_id=model_id,
                                        prompt_versions=document.prompt_versions)

                set_status("completed", "Готово", progress=1.0, json_path=str(json_path),
                           html_path=str(html_path), xml_path=str(xml_path), finished_at=time.time())
        except asyncio.CancelledError:
            set_status("cancelled", "Отменено", finished_at=time.time())
            raise
        except Exception as exc:
            logger.exception("protocol job failed protocol_job_id=%s job_id=%s", protocol_job_id, job_id)
            log(f"ERROR: {type(exc).__name__}: {exc}")
            self.store.update_protocol_job(protocol_job_id, status="failed", error=_short_error(exc),
                                            message="Ошибка обработки", finished_at=time.time())
        finally:
            if provider is not None:
                try:
                    await provider.unload()
                    log("LLM unloaded")
                except Exception:
                    logger.exception("failed to unload protocol LLM provider")
            gc.collect()
            _empty_cuda_cache()

    async def _unload_asr(self) -> None:
        result = self.unload_asr()
        if asyncio.iscoroutine(result):
            await result
        gc.collect()
        _empty_cuda_cache()

    # ---- topic split (item 7): two stages so a long transcript is never
    # sent to the model in one piece, yet the whole meeting is still
    # considered for boundaries ----------------------------------------

    async def _two_stage_topic_split(self, provider: LLMProvider, segments: list[TranscriptSegment], *,
                                      prompt_content: str, chunk_minutes: float, temperature: float, max_tokens: int,
                                      context_length: int, log: Callable[[str], None],
                                      raw_response_hook: Optional[Callable[[str], None]] = None,
                                      ) -> tuple[Optional[list[float]], dict[float, str]]:
        if not segments:
            return None, {}
        try:
            pre_blocks = split_by_time_windows(segments, window_seconds=chunk_minutes * 60, overlap_seconds=0)
            pre_blocks = enforce_context_budget(pre_blocks, context_length=context_length)
            hints: dict[float, str] = {}
            for block in pre_blocks:
                hint = ""
                try:
                    raw = await provider.generate_json(
                        prompt_content.replace("{transcript}", block.to_prompt_text()),
                        temperature=temperature, max_tokens=min(max_tokens, 512), max_retries=1,
                        on_raw_response=raw_response_hook,
                    )
                    topics = raw.get("topics") or []
                    if topics and isinstance(topics[0], dict):
                        hint = str(topics[0].get("title") or "")
                except Exception:
                    pass
                hints[block.start] = hint

            condensed = "\n".join(
                f"{(b.segments[0].speaker if b.segments else '')} [{format_seconds(b.start)} - {format_seconds(b.end)}]: "
                f"{hints.get(b.start) or '(тема не определена)'}"
                for b in pre_blocks
            )
            raw = await provider.generate_json(prompt_content.replace("{transcript}", condensed),
                                                temperature=temperature, max_tokens=max_tokens, max_retries=1,
                                                on_raw_response=raw_response_hook)
            topics = raw.get("topics") or []
            boundaries: list[float] = []
            for t in topics:
                if not isinstance(t, dict):
                    continue
                ts = t.get("start_timestamp")
                if not ts:
                    continue
                try:
                    v = parse_timestamp(str(ts))
                except ValueError:
                    continue
                if v < segments[0].start or v > segments[-1].end:
                    continue  # boundary must fall within the transcript's own range
                boundaries.append(v)
            boundaries = sorted(set(boundaries))
            min_gap = max(60.0, chunk_minutes * 60 * 0.25)
            deduped: list[float] = []
            for b in boundaries:
                if not deduped or b - deduped[-1] >= min_gap:
                    deduped.append(b)
            if not deduped:
                log("Topic split: unreliable/empty, falling back to time windows")
                return None, {}
            log(f"Topic split: {len(deduped)} boundary(ies) found across {len(pre_blocks)} pre-block(s)")
            return deduped, hints
        except Exception as exc:
            log(f"Topic split failed ({exc}), falling back to time windows")
            return None, {}

    # ---- fact-check (item 2) ------------------------------------------

    async def _fact_check(self, provider: LLMProvider, *, decisions: list[DecisionItem], tasks: list[TaskItem],
                           segments: list[TranscriptSegment], summary: str, topics: list[Any], prompt_content: str,
                           temperature: float, max_tokens: int, max_retries: int, context_length: int,
                           log: Callable[[str], None], raw_response_hook: Optional[Callable[[str], None]] = None,
                           ) -> tuple[dict[tuple[str, int], FactCheckVerdict], list[str]]:
        if not decisions and not tasks:
            return {}, []
        fact_check_doc = {
            "summary": summary, "topics": topics,
            "decisions": [{"index": i, **dataclasses.asdict(d)} for i, d in enumerate(decisions)],
            "tasks": [{"index": i, **dataclasses.asdict(t)} for i, t in enumerate(tasks)],
        }
        transcript_text = _fit_transcript_for_fact_check(segments, decisions, tasks, context_length=context_length)
        prompt = (prompt_content
                  .replace("{document}", json.dumps(fact_check_doc, ensure_ascii=False))
                  .replace("{transcript}", transcript_text))
        # Deliberately NOT caught here: after max_retries exhausted,
        # generate_json raises ProtocolValidationError, which must fail the
        # whole protocol job rather than emit an unverified protocol (item 2).
        raw = await provider.generate_json(prompt, temperature=temperature, max_tokens=max_tokens, max_retries=max_retries,
                                            on_raw_response=raw_response_hook)
        verified_raw = raw.get("verified") or []
        if not isinstance(verified_raw, list):
            verified_raw = []
        verdicts: dict[tuple[str, int], FactCheckVerdict] = {}
        for entry in verified_raw:
            v = FactCheckVerdict.from_json(entry if isinstance(entry, dict) else {})
            if v is None:
                continue  # unknown/malformed entries are ignored, never crash the job
            bound = len(decisions) if v.type == "decision" else len(tasks)
            if v.index >= bound:
                continue  # out-of-range index: ignore rather than trust a hallucinated reference
            verdicts[(v.type, v.index)] = v
        raw_notes = raw.get("unverified_items")
        notes = [t for t in (normalize_text(x) for x in raw_notes) if t] if isinstance(raw_notes, list) else []
        log(f"Fact-check: {len(verdicts)}/{len(decisions) + len(tasks)} item(s) verified")
        return verdicts, notes


def _nearest_hint(chunk_start: float, hints: dict[float, str]) -> Optional[str]:
    if not hints:
        return None
    nearest_start = min(hints, key=lambda s: abs(s - chunk_start))
    return hints.get(nearest_start) or None


def _fit_transcript_for_fact_check(segments: list[TranscriptSegment], decisions: list[DecisionItem],
                                    tasks: list[TaskItem], *, context_length: int, chars_per_token: float = 3.2,
                                    reserved_tokens: int = 1500) -> str:
    """Bound the transcript text sent for fact-checking to the model's own
    context window: the full text if it fits, otherwise the segments around
    each cited timecode -- using the *whole* range when one was given (item
    7/8), not just its start, plus a margin so adjacent segments (a number
    restated in the next line, a correction) are visible to the check."""
    full_text = "\n".join(f"{s.speaker} [{format_seconds(s.start)} - {format_seconds(s.end)}]: {s.text}" for s in segments)
    budget_chars = max(2000, int(max(0, context_length - reserved_tokens) * chars_per_token))
    if len(full_text) <= budget_chars:
        return full_text
    windows: list[tuple[float, float]] = []
    margin = 60.0
    for item in (*decisions, *tasks):
        try:
            lo, hi = parse_timestamp_range(item.timestamp)
        except ValueError:
            continue
        windows.append((lo - margin, hi + margin))
    if not windows:
        return full_text[:budget_chars]
    windows.sort()
    kept = [s for s in segments if any(lo <= s.start <= hi or lo <= s.end <= hi for lo, hi in windows)]
    if not kept:
        return full_text[:budget_chars]
    text = "\n".join(f"{s.speaker} [{format_seconds(s.start)} - {format_seconds(s.end)}]: {s.text}" for s in kept)
    return text[:budget_chars]


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:2000]


def _analysis_to_dict(a: ChunkAnalysis) -> dict[str, Any]:
    return {
        "chunk_index": a.chunk_index, "topic": a.topic, "summary": a.summary, "facts": list(a.facts),
        "decisions": [d.__dict__ for d in a.decisions], "tasks": [t.__dict__ for t in a.tasks],
        "open_questions": list(a.open_questions), "risks": list(a.risks),
        "disagreements": list(a.disagreements),
    }


def _resolve_anchor(timestamp: str, segments: list[TranscriptSegment], counters: dict[int, int]
                     ) -> tuple[Optional[str], Optional[TranscriptSegment], str, str]:
    """Item 7/8: link a decision/task timecode -- a single point or a range
    -- to a real transcript segment and mint a unique, stable anchor for
    it, never the raw timestamp text itself (not guaranteed unique, not
    guaranteed to correspond to any real moment). The range's *start* is
    used for the link per spec; the parsed (start, end) is returned
    separately so callers can record it regardless of whether a link could
    be resolved.

    A match must be a genuine transcript segment, not just whatever is
    numerically nearest: prefer a segment whose own [start, end] actually
    overlaps the cited range, and only fall back to nearest-by-start (still
    tolerance-checked) when nothing overlaps.
    """
    if not timestamp or timestamp == NOT_SPECIFIED or not segments:
        return None, None, "", ""
    try:
        t_start, t_end = parse_timestamp_range(timestamp)
    except ValueError:
        return None, None, "", ""
    start_str, end_str = format_seconds(t_start), format_seconds(t_end)

    overlapping = [s for s in segments if s.start <= t_end and s.end >= t_start]
    if overlapping:
        best = min(overlapping, key=lambda s: abs(s.start - t_start))
    else:
        best = min(segments, key=lambda s: abs(s.start - t_start))
        if abs(best.start - t_start) > _TIMESTAMP_MATCH_TOLERANCE_SECONDS:
            return None, None, start_str, end_str

    second = int(best.start)
    n = counters.get(second, 0) + 1
    counters[second] = n
    return f"timestamp-{second}-{n}", best, start_str, end_str


def _build_document(merged_raw: dict[str, Any], *, segments: list[TranscriptSegment], chunks, source_filename: str,
                     model_id: str, snapshot: SettingsSnapshot, raw_decisions: list[DecisionItem],
                     raw_tasks: list[TaskItem], fact_check_verdicts: dict[tuple[str, int], FactCheckVerdict],
                     fact_check_notes: list[str]) -> ProtocolDocument:
    participants = tuple(sorted({s.speaker for s in segments}))
    duration = max((s.end for s in segments), default=0.0)

    seg_to_chunk: dict[TranscriptSegment, int] = {}
    for c in chunks:
        for seg in c.segments:
            seg_to_chunk.setdefault(seg, c.index)

    anchor_counters: dict[int, int] = {}
    timestamp_refs: list[TimestampRef] = []
    unverified_notes: list[str] = []

    def _link(item, element_type: str, index: int):
        anchor, seg, ts_start, ts_end = _resolve_anchor(item.timestamp, segments, anchor_counters)
        verdict = fact_check_verdicts.get((element_type, index))
        confirmed = verdict.confirmed if verdict is not None else None
        reason = verdict.reason if verdict is not None else ""
        label = item.text if element_type == "decision" else item.task
        if anchor and seg is not None:
            timestamp_refs.append(TimestampRef(
                label=label, timestamp=format_seconds(seg.start), speaker=seg.speaker,
                chunk_index=seg_to_chunk.get(seg, -1), anchor=anchor, source_text=seg.text,
                element_type=element_type,
            ))
        elif item.timestamp and item.timestamp != NOT_SPECIFIED:
            unverified_notes.append(f"Не удалось сопоставить таймкод '{item.timestamp}' для: {label[:120]}")
        return dataclasses.replace(item, anchor=anchor, verified=confirmed, verification_reason=reason,
                                    timestamp_start=ts_start, timestamp_end=ts_end)

    decisions = tuple(_link(d, "decision", i) for i, d in enumerate(raw_decisions))
    tasks = tuple(_link(t, "task", i) for i, t in enumerate(raw_tasks))
    # Item 9: dedup decisions/tasks by their own text, order preserved.
    decisions = dedup_items_preserve_order(decisions, lambda d: d.text)
    tasks = dedup_items_preserve_order(tasks, lambda t: t.task)

    # Item 10: a merge-stage note ("Ответственный не указан", ...) must not
    # coexist with an item the fact-check stage went on to actually
    # confirm -- drop any merge note that's clearly about a now-confirmed
    # decision/task before merging in the fact-check/timestamp notes.
    confirmed_texts = [normalize_text(d.text) for d in decisions if d.verified is True]
    confirmed_texts += [normalize_text(t.task) for t in tasks if t.verified is True]

    def _conflicts_with_confirmed(note: str) -> bool:
        n = normalize_text(note).lower()
        return bool(n) and any(c and (c.lower() in n or n in c.lower()) for c in confirmed_texts)

    merge_notes = [t for t in (normalize_text(x) for x in (merged_raw.get("unverified_items") or [])) if t]
    merge_notes = [n for n in merge_notes if not _conflicts_with_confirmed(n)]

    unverified_items = dedup_preserve_order(tuple(merge_notes) + tuple(unverified_notes) + tuple(fact_check_notes))

    topics = dedup_items_preserve_order(
        tuple(t for t in (Topic.from_json(raw) for raw in (merged_raw.get("topics") or [])) if t is not None),
        lambda t: t.title,
    )

    return ProtocolDocument(
        meeting_title=str(merged_raw.get("meeting_title") or source_filename),
        processed_at=time.strftime("%Y-%m-%d %H:%M"),
        source_filename=source_filename,
        duration_seconds=duration,
        participants=participants,
        summary=str(merged_raw.get("summary") or ""),
        topics=topics,
        decisions=decisions,
        tasks=tasks,
        owners=tuple(sorted({t.owner for t in tasks if t.owner and t.owner != NOT_SPECIFIED})),
        deadlines=tuple(sorted({t.deadline for t in tasks if t.deadline and t.deadline != NOT_SPECIFIED})),
        open_questions=dedup_preserve_order(tuple(as_str_list(merged_raw.get("open_questions")))),
        risks=dedup_preserve_order(tuple(as_str_list(merged_raw.get("risks")))),
        disagreements=dedup_preserve_order(tuple(as_str_list(merged_raw.get("disagreements")))),
        next_steps=dedup_preserve_order(tuple(as_str_list(merged_raw.get("next_steps")))),
        unverified_items=unverified_items,
        timestamp_refs=tuple(timestamp_refs),
        model_id=model_id,
        prompt_versions=snapshot.prompt_versions,
        engine=snapshot.engine, model_path=snapshot.model_path, context_length=snapshot.context_length,
        temperature=snapshot.temperature, max_output_tokens=snapshot.max_output_tokens,
        chunking_settings={
            "chunk_minutes": snapshot.chunk_minutes, "chunk_overlap_seconds": snapshot.chunk_overlap_seconds,
            "topic_split_enabled": snapshot.topic_split_enabled,
        },
        app_commit=snapshot.app_commit, generated_at=time.time(),
    )
