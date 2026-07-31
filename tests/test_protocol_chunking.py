"""Transcript parsing, normalization, and chunking (test list items 10-12)."""
from protocol.chunking import (
    choose_chunks, enforce_context_budget, normalize_segments, parse_transcript,
    split_by_time_windows, split_by_topic_boundaries,
)


def _synthetic_transcript(n_segments: int, *, gap: int = 2, dur: int = 4) -> str:
    lines = []
    t = 0
    for i in range(n_segments):
        start, end = t, t + dur
        speaker = f"Спикер {1 + i % 2}"
        lines.append(f"{speaker} [{start // 60:02d}:{start % 60:02d} - {end // 60:02d}:{end % 60:02d}]: Реплика {i}.")
        t = end + gap
    return "\n".join(lines)


def test_parse_transcript_roundtrip():
    text = _synthetic_transcript(10)
    segments = parse_transcript(text)
    assert len(segments) == 10
    assert segments[0].speaker == "Спикер 1"
    assert segments[0].start == 0
    assert segments[-1].end > segments[0].end


def test_long_transcript_splits_into_multiple_chunks():
    # ~20 minutes of dialogue at 12-minute windows -> at least 2 chunks
    segments = parse_transcript(_synthetic_transcript(200))
    chunks = split_by_time_windows(segments, window_seconds=600, overlap_seconds=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert all(c.start <= s.start <= c.end for s in c.segments), "a reply was split across a chunk boundary"


def test_chunk_overlap_and_timecodes_preserved():
    segments = parse_transcript(_synthetic_transcript(200))
    chunks = split_by_time_windows(segments, window_seconds=600, overlap_seconds=30)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end + 30 + 1
        assert a.start is not None and a.end is not None


def test_topic_split_preferred_over_time_when_boundaries_given():
    segments = parse_transcript(_synthetic_transcript(60))
    chunks = choose_chunks(segments, topic_boundaries=[100.0, 250.0], window_seconds=600,
                            overlap_seconds=10, context_length=8192)
    assert len(chunks) == 3


def test_falls_back_to_time_windows_when_topic_split_unreliable():
    segments = parse_transcript(_synthetic_transcript(200))
    chunks_empty_boundaries = choose_chunks(segments, topic_boundaries=None, window_seconds=600,
                                             overlap_seconds=30, context_length=8192)
    chunks_time_only = split_by_time_windows(segments, window_seconds=600, overlap_seconds=30)
    assert len(chunks_empty_boundaries) == len(enforce_context_budget(chunks_time_only, context_length=8192))


def test_context_budget_splits_oversized_chunk():
    segments = parse_transcript(_synthetic_transcript(500))
    big_chunks = split_by_time_windows(segments, window_seconds=3600, overlap_seconds=30)
    assert len(big_chunks) == 1
    bounded = enforce_context_budget(big_chunks, context_length=1024)
    assert len(bounded) > 1
    for c in bounded:
        assert len(c.to_prompt_text()) <= int((1024 - 1200) * 3.2) or len(c.to_prompt_text()) < 5000


def test_normalize_merges_short_same_speaker_replies_and_drops_repeats():
    text = (
        "Спикер 1 [00:00 - 00:02]: Привет.\n"
        "Спикер 1 [00:03 - 00:05]: Привет.\n"  # exact duplicate, should collapse
        "Спикер 1 [00:06 - 00:08]: как дела\n"  # short, same speaker, small gap -> merge
        "Спикер 2 [00:09 - 00:12]: Хорошо, спасибо."
    )
    segments = parse_transcript(text)
    normalized = normalize_segments(segments)
    assert len(normalized) < len(segments)
    assert normalized[-1].speaker == "Спикер 2"
