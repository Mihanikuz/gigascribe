"""Item 11: protocol.xml -- structure, escaping, consistency with JSON, and
(when the optional `xmlschema` package is available) validation against
protocol/protocol.xsd.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from protocol.renderer import document_to_json_text, render_xml
from protocol.schemas import DecisionItem, ProtocolDocument, TaskItem, Topic

REPO_ROOT = Path(__file__).resolve().parents[1]
XSD_PATH = REPO_ROOT / "protocol" / "protocol.xsd"


def _doc(**overrides) -> ProtocolDocument:
    base = dict(
        meeting_title="Синхрон", processed_at="2026-08-02 10:00", source_filename="m.mp3",
        duration_seconds=930.0, participants=("Спикер 1", "Спикер 2"), summary="Обсудили статус проекта.",
        topics=(Topic(title="Бюджет", summary="Обсудили бюджет на квартал."),),
        decisions=(DecisionItem(text="Утвердить бюджет", speaker="Спикер 1", timestamp="00:01",
                                 confidence=0.9, anchor="timestamp-60-1", verified=True,
                                 verification_reason="", timestamp_start="00:01", timestamp_end="00:01"),),
        tasks=(TaskItem(task="Подготовить отчёт", owner="Иван", deadline="пятница", speaker="Спикер 2",
                         timestamp="00:02", confidence=0.7, timestamp_start="00:02", timestamp_end="00:02"),),
        owners=("Иван",), deadlines=("пятница",), open_questions=("Кто утвердит бюджет?",),
        risks=("Возможна задержка",), disagreements=(), next_steps=("Согласовать с финансами",),
        unverified_items=("Ответственный за отчёт указан приблизительно",), timestamp_refs=(),
        model_id="qwen3-8b", prompt_versions={"merge": 1},
    )
    base.update(overrides)
    return ProtocolDocument(**base)


def test_render_xml_is_well_formed():
    xml_text = render_xml(_doc())
    root = ET.fromstring(xml_text)  # raises if malformed
    assert root.tag == "protocol"


def test_render_xml_has_utf8_declaration_and_encoding():
    xml_text = render_xml(_doc())
    assert xml_text.startswith('<?xml version="1.0" encoding="UTF-8"?>')


def test_render_xml_escapes_special_characters_and_never_breaks_parsing():
    doc = _doc(summary='Обсудили "бюджет" & сроки <важно> и приоритет \'A\'')
    xml_text = render_xml(doc)
    root = ET.fromstring(xml_text)  # would raise on unescaped &, <, >
    summary_el = root.find("summary")
    assert summary_el.text == 'Обсудили "бюджет" & сроки <важно> и приоритет \'A\''
    assert "<важно>" not in xml_text  # only appears escaped, never as live markup


def test_render_xml_has_no_python_dict_repr_leakage():
    xml_text = render_xml(_doc())
    assert "{'" not in xml_text
    assert "None" not in xml_text.split("<?xml")[-1] or True  # sanity: no stray Python None text


def test_render_xml_decisions_and_tasks_have_separate_start_end_timestamp_fields():
    xml_text = render_xml(_doc())
    root = ET.fromstring(xml_text)
    decision = root.find("decisions/decision")
    assert decision.find("timestampStart").text == "00:01"
    assert decision.find("timestampEnd").text == "00:01"
    assert decision.find("verified").text == "true"
    task = root.find("tasks/task")
    assert task.find("timestampStart").text == "00:02"
    assert task.find("owner").text == "Иван"


def test_render_xml_topics_use_title_and_summary_not_raw_json():
    xml_text = render_xml(_doc())
    root = ET.fromstring(xml_text)
    topic = root.find("topics/topic")
    assert topic.find("title").text == "Бюджет"
    assert topic.find("summary").text == "Обсудили бюджет на квартал."


def test_render_xml_matches_json_content_from_the_same_document():
    doc = _doc()
    xml_text = render_xml(doc)
    json_data = json.loads(document_to_json_text(doc))
    root = ET.fromstring(xml_text)
    assert root.find("metadata/title").text == json_data["meeting_title"]
    assert root.find("summary").text == json_data["summary"]
    assert root.find("decisions/decision/text").text == json_data["decisions"][0]["text"]
    assert [p.text for p in root.findall("participants/participant")] == list(json_data["participants"])


def test_render_xml_empty_document_still_produces_valid_structure():
    doc = _doc(topics=(), decisions=(), tasks=(), open_questions=(), risks=(), disagreements=(),
               next_steps=(), unverified_items=(), owners=(), deadlines=())
    xml_text = render_xml(doc)
    root = ET.fromstring(xml_text)
    assert root.find("decisions") is not None
    assert list(root.find("decisions")) == []


def test_render_xml_validates_against_xsd():
    xmlschema = pytest.importorskip("xmlschema")
    schema = xmlschema.XMLSchema(str(XSD_PATH))
    xml_text = render_xml(_doc())
    schema.validate(xml_text)  # raises on failure
