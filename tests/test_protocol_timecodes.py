"""Item 7/8: timecode ranges, and linking a decision/task timestamp to a
genuine transcript segment rather than merely the nearest one.
"""
import pytest

from protocol.chunking import parse_timestamp_range
from protocol.schemas import TranscriptSegment
from protocol.service import _resolve_anchor


def test_parse_timestamp_range_single_point():
    assert parse_timestamp_range("15:12") == (912.0, 912.0)


def test_parse_timestamp_range_mm_ss_range():
    start, end = parse_timestamp_range("15:12 - 16:00")
    assert start == 912.0
    assert end == 960.0


def test_parse_timestamp_range_hh_mm_ss_range():
    start, end = parse_timestamp_range("01:02:15 - 01:03:20")
    assert start == 3735.0
    assert end == 3800.0


def test_parse_timestamp_range_sorts_reversed_bounds():
    start, end = parse_timestamp_range("16:00 - 15:12")
    assert start == 912.0
    assert end == 960.0


def test_parse_timestamp_range_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp_range("not a timestamp")


def _segments():
    return [
        TranscriptSegment(speaker="Спикер 1", start=0.0, end=10.0, text="Начало обсуждения."),
        TranscriptSegment(speaker="Спикер 2", start=910.0, end=915.0, text="Ставим 80 процентов."),
        TranscriptSegment(speaker="Спикер 1", start=915.0, end=958.0, text="Согласен, фиксируем диапазон."),
        TranscriptSegment(speaker="Спикер 2", start=2000.0, end=2010.0, text="Отдельная поздняя реплика."),
    ]


def test_resolve_anchor_links_single_timestamp_to_overlapping_segment():
    anchor, seg, start, end = _resolve_anchor("15:12", _segments(), {})
    assert anchor is not None
    assert seg.text == "Ставим 80 процентов."
    assert start == "15:12"
    assert end == "15:12"


def test_resolve_anchor_range_uses_start_for_the_anchor_but_keeps_both_bounds():
    anchor, seg, start, end = _resolve_anchor("15:12 - 16:00", _segments(), {})
    assert anchor is not None
    assert start == "15:12"
    assert end == "16:00"


def test_resolve_anchor_prefers_genuine_overlap_over_bare_nearest_distance():
    """A timestamp that falls inside a real segment's own [start, end] must
    resolve to *that* segment, not to whichever segment merely starts
    closest in absolute time (item 7's 'not just the nearest segment')."""
    segments = [
        TranscriptSegment(speaker="A", start=0.0, end=5.0, text="рядом по старту, но не тот сегмент"),
        TranscriptSegment(speaker="B", start=6.0, end=40.0, text="настоящий сегмент, длинный"),
    ]
    # 20s falls inside segment B's [6, 40] range, even though segment A's
    # start (0) is numerically no closer than B's start (6) by much --
    # verifies overlap wins the match over blind nearest-start.
    anchor, seg, _, _ = _resolve_anchor("00:20", segments, {})
    assert seg.text == "настоящий сегмент, длинный"


def test_resolve_anchor_returns_none_for_unmatchable_timestamp():
    anchor, seg, start, end = _resolve_anchor("99:59:59", _segments(), {})
    assert anchor is None
    assert seg is None
    assert start == "99:59:59"  # bounds are still reported even without a link
    assert end == "99:59:59"


def test_resolve_anchor_ids_are_unique_per_call_even_at_the_same_second():
    counters: dict[int, int] = {}
    a1, _, _, _ = _resolve_anchor("15:12", _segments(), counters)
    a2, _, _, _ = _resolve_anchor("15:12", _segments(), counters)
    assert a1 != a2
    assert a1.startswith("timestamp-910-")  # anchored to the matched segment's own start
    assert a2.startswith("timestamp-910-")
