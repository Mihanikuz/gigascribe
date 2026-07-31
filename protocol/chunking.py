"""Transcript parsing, normalization, and chunking.

Pure, synchronous, and dependency-free (stdlib only) so it can be unit
tested without a GPU or an LLM. Chunk boundaries are always chosen between
segments, never inside one, and adjacent chunks get a small overlap so a
reply spanning a boundary is never cut mid-sentence for the model's context.
"""
from __future__ import annotations

import re
from typing import Optional

from .schemas import TranscriptChunk, TranscriptSegment

_LINE_RE = re.compile(r"^(?P<speaker>.+?) \[(?P<start>[\d:]+) - (?P<end>[\d:]+)\]: (?P<text>.*)$")

# Rough, deliberately conservative chars-per-token estimate for a safety
# margin across model families; used only to keep a chunk within the
# selected model's context window, not for exact token accounting.
_CHARS_PER_TOKEN = 3.2


def parse_timestamp(value: str) -> float:
    parts = [int(p) for p in value.split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    raise ValueError(f"Unrecognized timestamp: {value!r}")


def parse_transcript(text: str) -> list[TranscriptSegment]:
    """Parse GigaScribe's plain-text transcript format back into segments.

    Lines that don't match "Speaker [start - end]: text" are treated as a
    continuation of the previous segment's text (defensive: ASR text is not
    expected to contain literal newlines, but nothing guarantees it).
    """
    segments: list[TranscriptSegment] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if m:
            segments.append(TranscriptSegment(
                speaker=m.group("speaker"),
                start=parse_timestamp(m.group("start")),
                end=parse_timestamp(m.group("end")),
                text=m.group("text"),
            ))
        elif segments:
            last = segments[-1]
            segments[-1] = TranscriptSegment(last.speaker, last.start, last.end, last.text + "\n" + line)
    return segments


def normalize_segments(segments: list[TranscriptSegment], *, max_gap_seconds: float = 2.0,
                        max_merged_chars: int = 400) -> list[TranscriptSegment]:
    """Drop exact consecutive repeats and merge very short same-speaker replies.

    Never touches text content otherwise, and never merges across a gap
    bigger than max_gap_seconds (a real pause/topic change shouldn't be
    silently glued together).
    """
    deduped: list[TranscriptSegment] = []
    for seg in segments:
        if deduped and deduped[-1].speaker == seg.speaker and deduped[-1].text.strip() == seg.text.strip():
            deduped[-1] = TranscriptSegment(deduped[-1].speaker, deduped[-1].start, seg.end, deduped[-1].text)
            continue
        deduped.append(seg)

    merged: list[TranscriptSegment] = []
    for seg in deduped:
        if (merged and merged[-1].speaker == seg.speaker
                and seg.start - merged[-1].end <= max_gap_seconds
                and len(merged[-1].text) + len(seg.text) <= max_merged_chars):
            prev = merged[-1]
            merged[-1] = TranscriptSegment(prev.speaker, prev.start, seg.end, f"{prev.text} {seg.text}".strip())
        else:
            merged.append(seg)
    return merged


def split_by_time_windows(segments: list[TranscriptSegment], *, window_seconds: float,
                           overlap_seconds: float) -> list[TranscriptChunk]:
    """Split into fixed-size time windows, cutting only between segments."""
    if not segments:
        return []
    chunks: list[TranscriptChunk] = []
    window_start = segments[0].start
    i = 0
    index = 0
    n = len(segments)
    while i < n:
        window_end = window_start + window_seconds
        window_segments: list[TranscriptSegment] = []
        j = i
        while j < n and segments[j].start < window_end:
            window_segments.append(segments[j])
            j += 1
        if not window_segments:
            # No segment starts before window_end (a long gap): jump to the
            # next segment's own window instead of emitting an empty chunk.
            window_start = segments[i].start
            continue
        # Small trailing overlap: also include the next segment(s) that start
        # within overlap_seconds of this window's end, without advancing i
        # past them (they still open the following window too).
        k = j
        while k < n and segments[k].start < window_end + overlap_seconds:
            window_segments.append(segments[k])
            k += 1
        chunk_start = window_segments[0].start
        chunk_end = window_segments[-1].end
        chunks.append(TranscriptChunk(index=index, start=chunk_start, end=chunk_end, segments=tuple(window_segments)))
        index += 1
        i = j
        window_start = window_end
    return chunks


def split_by_topic_boundaries(segments: list[TranscriptSegment], boundary_timestamps: list[float], *,
                               overlap_seconds: float) -> list[TranscriptChunk]:
    """Split at the given topic-start timestamps (already validated by the caller)."""
    if not segments:
        return []
    boundaries = sorted(set(t for t in boundary_timestamps if t > segments[0].start))
    cut_points = [segments[0].start, *boundaries, segments[-1].end + 1]
    chunks: list[TranscriptChunk] = []
    index = 0
    for start, end in zip(cut_points, cut_points[1:]):
        window_segments = [s for s in segments if start <= s.start < end]
        if not window_segments:
            continue
        overlap_extra = [s for s in segments if end <= s.start < end + overlap_seconds]
        all_segments = window_segments + overlap_extra
        chunks.append(TranscriptChunk(
            index=index, start=window_segments[0].start, end=window_segments[-1].end,
            segments=tuple(all_segments),
        ))
        index += 1
    return chunks


def enforce_context_budget(chunks: list[TranscriptChunk], *, context_length: int,
                            reserved_tokens: int = 1200) -> list[TranscriptChunk]:
    """Recursively halve any chunk whose text would overflow the model's
    context window (after leaving room for the prompt template and output).
    """
    max_chars = max(500, int((context_length - reserved_tokens) * _CHARS_PER_TOKEN))
    result: list[TranscriptChunk] = []

    def _process(chunk: TranscriptChunk) -> list[TranscriptChunk]:
        if len(chunk.to_prompt_text()) <= max_chars or len(chunk.segments) <= 1:
            return [chunk]
        mid = len(chunk.segments) // 2
        left = chunk.segments[:mid]
        right = chunk.segments[mid:]
        parts: list[TranscriptChunk] = []
        parts.extend(_process(TranscriptChunk(index=chunk.index, start=left[0].start, end=left[-1].end, segments=left)))
        parts.extend(_process(TranscriptChunk(index=chunk.index, start=right[0].start, end=right[-1].end, segments=right)))
        return parts

    for c in chunks:
        result.extend(_process(c))
    for new_index, c in enumerate(result):
        result[new_index] = TranscriptChunk(index=new_index, start=c.start, end=c.end, segments=c.segments, topic_hint=c.topic_hint)
    return result


def choose_chunks(segments: list[TranscriptSegment], *, topic_boundaries: Optional[list[float]],
                   window_seconds: float, overlap_seconds: float, context_length: int) -> list[TranscriptChunk]:
    """Prefer topic boundaries when given and non-trivial, else time windows."""
    if topic_boundaries:
        chunks = split_by_topic_boundaries(segments, topic_boundaries, overlap_seconds=overlap_seconds)
    else:
        chunks = []
    if not chunks:
        chunks = split_by_time_windows(segments, window_seconds=window_seconds, overlap_seconds=overlap_seconds)
    return enforce_context_budget(chunks, context_length=context_length)
