"""HTML rendering: escaping (item 16), timestamp links, JSON export."""
import json

from protocol.prompts import DEFAULT_PROMPTS, validate_prompt_content
from protocol.renderer import build_body, document_to_json_text, render_html
from protocol.schemas import DecisionItem, ProtocolDocument, TaskItem, TimestampRef

TEMPLATE = DEFAULT_PROMPTS["html_template"]


def _doc(**overrides) -> ProtocolDocument:
    base = dict(
        meeting_title="Синхрон", processed_at="2026-07-31", source_filename="m.mp3",
        duration_seconds=100.0, participants=("Спикер 1",), summary="s", topics=(),
        decisions=(), tasks=(), owners=(), deadlines=(), open_questions=(), risks=(),
        disagreements=(), next_steps=(), unverified_items=(), timestamp_refs=(),
        model_id="qwen3-8b", prompt_versions={},
    )
    base.update(overrides)
    return ProtocolDocument(**base)


def test_html_escapes_script_tags_in_title():
    doc = _doc(meeting_title="<script>alert(1)</script> Синхрон")
    html_out = render_html(doc, template=TEMPLATE)
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_html_escapes_live_tags_in_participants_and_decisions():
    doc = _doc(
        participants=("Спикер 2 <img src=x onerror=alert(2)>",),
        decisions=(DecisionItem(text='Решение" onmouseover="alert(4)', speaker="И", timestamp="00:01", confidence=0.5),),
    )
    html_out = render_html(doc, template=TEMPLATE)
    assert "<img " not in html_out
    assert 'onmouseover="alert' not in html_out
    assert "&lt;img" in html_out


def test_unspecified_fields_render_as_not_specified_marker():
    doc = _doc(tasks=(TaskItem(task="Сделать X"),))
    html_out = render_html(doc, template=TEMPLATE)
    assert "не указан" in html_out


def test_timestamp_appendix_links_match_decision_anchors():
    """Item 8: a decision/task only links when it carries a real, resolved
    anchor (produced by service.py's nearest-segment matching) -- the
    renderer itself just has to make sure that link and that appendix row
    always agree on the same anchor id."""
    doc = _doc(
        decisions=(DecisionItem(text="X", speaker="И", timestamp="00:05:12", confidence=0.9, anchor="timestamp-312-1"),),
        timestamp_refs=(TimestampRef(label="X", timestamp="00:05:12", speaker="И", chunk_index=0,
                                      anchor="timestamp-312-1", source_text="реплика", element_type="decision"),),
    )
    html_out = render_html(doc, template=TEMPLATE)
    assert "id='timestamp-312-1'" in html_out
    assert "href='#timestamp-312-1'" in html_out


def test_unresolved_timestamp_renders_as_plain_text_not_a_dead_link():
    """Item 8/17: no anchor -> no link, just the raw timestamp text."""
    doc = _doc(decisions=(DecisionItem(text="X", speaker="И", timestamp="99:59:59", confidence=0.9, anchor=None),))
    html_out = render_html(doc, template=TEMPLATE)
    assert "<a class='timestamp-link'" not in html_out
    assert "99:59:59" in html_out


def test_render_html_requires_body_placeholder():
    try:
        validate_prompt_content("html_template", "<p>no body var</p>")
        assert False, "should have rejected a template without $body"
    except ValueError:
        pass


def test_document_json_text_is_valid_and_round_trips_unescaped():
    doc = _doc(meeting_title="<b>Заголовок</b>")
    text = document_to_json_text(doc)
    parsed = json.loads(text)
    assert parsed["meeting_title"] == "<b>Заголовок</b>"  # JSON keeps raw text; only HTML rendering escapes
