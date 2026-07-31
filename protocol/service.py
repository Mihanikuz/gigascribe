"""ProtocolService: the one stable entry point the main app calls.

Owns the whole create-protocol pipeline (GPU handover, chunking, per-chunk
analysis, merge, render) and never touches the main `jobs` table or raises
in a way that could fail the caller's transcription job -- every error is
caught, logged to protocol.log, and recorded on the protocol_jobs row only.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from . import glossary as glossary_mod
from .chunking import choose_chunks, normalize_segments, parse_timestamp, parse_transcript
from .config import ProtocolConfig
from .models import SUPPORTED_PROTOCOL_MODELS
from .providers import create_provider
from .providers.base import LLMProvider
from .renderer import document_to_json_text, render_html
from .schemas import (
    ChunkAnalysis, DecisionItem, ProtocolDocument, ProtocolOptions, ProtocolResult,
    ProtocolValidationError, TaskItem, TimestampRef, format_seconds,
)
from .store import ProtocolStore

logger = logging.getLogger(__name__)

# unload_asr may be sync (ModelManager.unload_all) or async; both supported.
UnloadCallable = Callable[[], Any]


class ProtocolDisabledError(RuntimeError):
    pass


class ProtocolService:
    def __init__(self, *, config: ProtocolConfig, store: ProtocolStore, gpu_lock: asyncio.Lock,
                 unload_asr: UnloadCallable):
        self.config = config
        self.store = store
        self.gpu_lock = gpu_lock
        self.unload_asr = unload_asr
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

        protocol_job_id = uuid.uuid4().hex
        result_dir = self.config.results_dir / job_id / protocol_job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = result_dir / "protocol.log"
        self.store.create_protocol_job(id=protocol_job_id, job_id=job_id, username=username, model_id=model_id)
        self.store.update_protocol_job(protocol_job_id, log_path=str(log_path))
        asyncio.create_task(self._run(protocol_job_id, transcript_path, job_id, result_dir, options))
        return ProtocolResult(protocol_job_id=protocol_job_id, status="queued", document=None,
                               json_path=None, html_path=None)

    async def retry(self, protocol_job_id: str, transcript_path: Path, job_id: str) -> ProtocolResult:
        job = self.store.get_protocol_job(protocol_job_id)
        if not job:
            raise ValueError("Unknown protocol job")
        if job["status"] not in ("failed", "cancelled"):
            raise ValueError("Only a failed or cancelled protocol job can be retried")
        result_dir = Path(job["log_path"]).parent if job.get("log_path") else self.config.results_dir / job_id / protocol_job_id
        self.store.update_protocol_job(protocol_job_id, status="queued", progress=0, message="В очереди",
                                        error=None, cancel_requested=0, attempts=job["attempts"] + 1)
        asyncio.create_task(self._run(protocol_job_id, transcript_path, job_id, result_dir,
                                       ProtocolOptions(model_id=job.get("model_id"))))
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

    # ---- pipeline ---------------------------------------------------------

    async def _run(self, protocol_job_id: str, transcript_path: Path, job_id: str,
                    result_dir: Path, options: ProtocolOptions) -> None:
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

        provider: Optional[LLMProvider] = None
        try:
            set_status("queued", "В очереди", progress=0.0)
            model_id = options.model_id or self.store.active_model_id()
            model_state = self.store.get_model_state(model_id)
            if not model_state or not model_state.get("installed"):
                raise ProtocolValidationError(f"Model not installed: {model_id}")

            raw_transcript = transcript_path.read_text(encoding="utf-8")
            segments = parse_transcript(raw_transcript)
            segments = normalize_segments(segments)
            if not segments:
                raise ProtocolValidationError("Transcript is empty after parsing")

            glossary_terms = self.store.resolved_terms(project=options.glossary_project)
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
                provider = create_provider(engine=model_state["engine"], model_path=model_state["local_path"],
                                            context_length=model_state["context_length"])
                await provider.load()
                log(f"Loaded provider engine={model_state['engine']} model={model_id}")

                temperature = options.temperature if options.temperature is not None else model_state["temperature"]
                max_tokens = options.max_output_tokens or model_state["max_output_tokens"]

                if cancelled():
                    set_status("cancelled", "Отменено"); return

                set_status("splitting", "Разбиение на блоки", progress=0.15)
                chunk_minutes = options.chunk_minutes or self.config.default_chunk_minutes
                topic_split = self.config.topic_split_enabled if options.topic_split is None else options.topic_split
                topic_boundaries = None
                if topic_split:
                    topic_boundaries = await self._try_topic_split(provider, segments, temperature=temperature,
                                                                    max_tokens=max_tokens, log=log)
                chunks = choose_chunks(
                    segments, topic_boundaries=topic_boundaries,
                    window_seconds=chunk_minutes * 60, overlap_seconds=self.config.chunk_overlap_seconds,
                    context_length=model_state["context_length"],
                )
                if not chunks:
                    raise ProtocolValidationError("Chunking produced no chunks")
                self.store.update_protocol_job(protocol_job_id, chunk_total=len(chunks), chunk_current=0)
                log(f"Split into {len(chunks)} chunk(s)")

                analyses: list[ChunkAnalysis] = []
                set_status("processing_chunks", f"Обработка блока 1/{len(chunks)}", progress=0.2)
                for chunk in chunks:
                    if cancelled():
                        set_status("cancelled", "Отменено"); return
                    prompt_spec = self.store.get_active_prompt("chunk_analysis", model_id)
                    prompt = prompt_spec.content.replace("{transcript_chunk}", chunk.to_prompt_text())
                    raw = await provider.generate_json(prompt, temperature=temperature, max_tokens=max_tokens,
                                                         max_retries=self.config.max_retries)
                    analysis = ChunkAnalysis.from_llm_json(chunk.index, raw)
                    analyses.append(analysis)
                    self.store.update_protocol_job(protocol_job_id, chunk_current=chunk.index + 1)
                    progress = 0.2 + 0.5 * ((chunk.index + 1) / len(chunks))
                    set_status("processing_chunks", f"Обработка блока {chunk.index + 1}/{len(chunks)}", progress=progress)

                if cancelled():
                    set_status("cancelled", "Отменено"); return

                set_status("merging", "Сведение итогового протокола", progress=0.75)
                merge_prompt_spec = self.store.get_active_prompt("merge", model_id)
                merge_prompt = merge_prompt_spec.content.replace(
                    "{chunk_analyses}", json.dumps([_analysis_to_dict(a) for a in analyses], ensure_ascii=False),
                )
                merged_raw = await provider.generate_json(merge_prompt, temperature=temperature, max_tokens=max_tokens,
                                                            max_retries=self.config.max_retries)

                document = _build_document(
                    merged_raw, segments=segments, chunks=chunks, source_filename=transcript_path.stem,
                    model_id=model_id, prompt_versions={
                        "chunk_analysis": self.store.get_active_prompt("chunk_analysis", model_id).version,
                        "merge": merge_prompt_spec.version,
                    },
                )

                set_status("rendering", "Формирование HTML", progress=0.9)
                template_spec = self.store.get_active_prompt("html_template", model_id)
                html_text = render_html(document, template=template_spec.content)
                json_text = document_to_json_text(document)
                json_path = result_dir / "protocol.json"
                html_path = result_dir / "protocol.html"
                json_path.write_text(json_text, encoding="utf-8")
                html_path.write_text(html_text, encoding="utf-8")
                self.store.save_result(protocol_job_id, document.to_json_dict(), model_id=model_id,
                                        prompt_versions=document.prompt_versions)

                set_status("completed", "Готово", progress=1.0, json_path=str(json_path),
                           html_path=str(html_path), finished_at=time.time())
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

    async def _try_topic_split(self, provider: LLMProvider, segments, *, temperature: float, max_tokens: int,
                                log: Callable[[str], None]) -> Optional[list[float]]:
        try:
            prompt_spec = self.store.get_active_prompt("topic_split")
            full_text = "\n".join(f"{s.speaker} [{format_seconds(s.start)}]: {s.text}" for s in segments)
            prompt = prompt_spec.content.replace("{transcript}", full_text[:20000])
            raw = await provider.generate_json(prompt, temperature=temperature, max_tokens=max_tokens, max_retries=1)
            topics = raw.get("topics") or []
            boundaries = []
            for t in topics:
                ts = t.get("start_timestamp")
                if ts:
                    boundaries.append(parse_timestamp(ts))
            if len(boundaries) < 1:
                log("Topic split: unreliable/empty, falling back to time windows")
                return None
            log(f"Topic split: {len(boundaries)} boundary(ies) found")
            return boundaries
        except Exception as exc:
            log(f"Topic split failed ({exc}), falling back to time windows")
            return None


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
        "disagreements": list(a.disagreements), "terms": list(a.terms),
    }


def _build_document(merged_raw: dict[str, Any], *, segments, chunks, source_filename: str, model_id: str,
                     prompt_versions: dict[str, int]) -> ProtocolDocument:
    if not isinstance(merged_raw, dict):
        raise ProtocolValidationError("merge response must be a JSON object")
    participants = tuple(sorted({s.speaker for s in segments}))
    duration = max((s.end for s in segments), default=0.0)
    timestamp_refs = tuple(
        TimestampRef(label=f"блок {c.index + 1}", timestamp=format_seconds(c.start), speaker=c.segments[0].speaker if c.segments else "",
                      chunk_index=c.index)
        for c in chunks
    )
    decisions = tuple(DecisionItem.from_json(d) if isinstance(d, dict) else d for d in (merged_raw.get("decisions") or []))
    tasks = tuple(TaskItem.from_json(t) if isinstance(t, dict) else t for t in (merged_raw.get("tasks") or []))
    return ProtocolDocument(
        meeting_title=str(merged_raw.get("meeting_title") or source_filename),
        processed_at=time.strftime("%Y-%m-%d %H:%M"),
        source_filename=source_filename,
        duration_seconds=duration,
        participants=participants,
        summary=str(merged_raw.get("summary") or ""),
        topics=tuple(str(t) for t in (merged_raw.get("topics") or [])),
        decisions=decisions,
        tasks=tasks,
        owners=tuple(sorted({t.owner for t in tasks if t.owner and t.owner != "не указан"})),
        deadlines=tuple(sorted({t.deadline for t in tasks if t.deadline and t.deadline != "не указан"})),
        open_questions=tuple(str(x) for x in (merged_raw.get("open_questions") or [])),
        risks=tuple(str(x) for x in (merged_raw.get("risks") or [])),
        disagreements=tuple(str(x) for x in (merged_raw.get("disagreements") or [])),
        next_steps=tuple(str(x) for x in (merged_raw.get("next_steps") or [])),
        unverified_items=tuple(str(x) for x in (merged_raw.get("unverified_items") or [])),
        timestamp_refs=timestamp_refs,
        model_id=model_id,
        prompt_versions=prompt_versions,
    )
