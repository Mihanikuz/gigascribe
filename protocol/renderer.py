"""Render a ProtocolDocument to a self-contained HTML page.

The HTML is always derived from the structured JSON document -- never the
other way around -- and every piece of LLM- or user-provided text is
escaped before being placed in markup. The admin-editable "html_template"
prompt only controls the outer page shell (title/body placeholders); the
body itself is always assembled from the validated document fields, so an
edited template can't accidentally swallow or corrupt document data.
"""
from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from string import Template
from typing import Any

from .schemas import ProtocolDocument

_PAGE_CSS = """
:root{color-scheme:light dark}
body{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.5}
h1{font-size:1.4rem}
h2{font-size:1.15rem;border-bottom:1px solid #8884;padding-bottom:4px;margin-top:28px}
.meta{color:#888;font-size:.9em;margin-bottom:16px}
.toc{border:1px solid #8884;border-radius:8px;padding:12px 18px;margin:16px 0}
.toc a{display:block;padding:2px 0}
ul{padding-left:20px}
table{border-collapse:collapse;width:100%;margin:8px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #8883;vertical-align:top}
.badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:.78em;background:#eee;color:#333}
.badge.unverified-badge{background:#fbe8cc;color:#7a4a00}
.unverified{color:#a15c00}
.timestamp-link{text-decoration:none;color:inherit;font-variant-numeric:tabular-nums;border-bottom:1px dotted}
.actions{margin:18px 0;display:flex;gap:8px}
.actions button,.actions a{padding:6px 12px;border-radius:6px;border:1px solid #8884;background:transparent;cursor:pointer;color:inherit;text-decoration:none}
@media print{.actions,.toc{display:none}}
"""


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _list_section(title: str, anchor: str, items: tuple, *, unverified: bool = False) -> str:
    if not items:
        return ""
    cls = " class='unverified'" if unverified else ""
    lis = "".join(f"<li{cls}>{_e(item)}</li>" for item in items)
    return f"<h2 id='{anchor}'>{_e(title)}</h2><ul>{lis}</ul>"


def _topics_section(topics) -> str:
    """Item 6: a title plus an optional short description -- never a raw
    nested JSON object dumped as text."""
    if not topics:
        return ""
    items = "".join(
        f"<li><strong>{_e(t.title)}</strong>{f'<br>{_e(t.summary)}' if t.summary else ''}</li>"
        for t in topics
    )
    return f"<h2 id='topics'>Обсуждавшиеся темы</h2><ul>{items}</ul>"


def _ts_cell(item) -> str:
    """Item 8: link only when a real anchor was resolved (nearest transcript
    segment within tolerance) -- otherwise show the raw timestamp as inert
    text so we never emit a link to an id that doesn't exist in the page."""
    if getattr(item, "anchor", None):
        return f"<a class='timestamp-link' href='#{_e(item.anchor)}'>{_e(item.timestamp)}</a>"
    return _e(item.timestamp)


def _verified_cell(item) -> str:
    verified = getattr(item, "verified", None)
    if verified is True:
        return "<span class='badge completed' title='Подтверждено проверкой фактов'>подтверждено</span>"
    if verified is False:
        reason = getattr(item, "verification_reason", "") or ""
        title = f" title='{_e(reason)}'" if reason else ""
        return f"<span class='badge unverified-badge'{title}>не подтверждено</span>"
    return "<span class='muted'>—</span>"


def _decisions_table(decisions) -> str:
    if not decisions:
        return ""
    rows = "".join(
        f"<tr><td>{_e(d.text)}</td><td>{_e(d.speaker)}</td>"
        f"<td>{_ts_cell(d)}</td><td>{d.confidence:.2f}</td><td>{_verified_cell(d)}</td></tr>"
        for d in decisions
    )
    return (
        "<h2 id='decisions'>Принятые решения</h2>"
        "<table><thead><tr><th>Решение</th><th>Говорящий</th><th>Таймкод</th><th>Уверенность</th><th>Проверено</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _tasks_table(tasks) -> str:
    if not tasks:
        return ""
    rows = "".join(
        f"<tr><td>{_e(t.task)}</td><td>{_e(t.owner)}</td><td>{_e(t.deadline)}</td>"
        f"<td>{_ts_cell(t)}</td><td>{_verified_cell(t)}</td></tr>"
        for t in tasks
    )
    return (
        "<h2 id='tasks'>Поручения</h2>"
        "<table><thead><tr><th>Поручение</th><th>Ответственный</th><th>Срок</th><th>Таймкод</th><th>Проверено</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


_ELEMENT_TYPE_LABELS = {"decision": "решение", "task": "поручение", "chunk": "блок"}


def _appendix(timestamp_refs) -> str:
    if not timestamp_refs:
        return ""
    rows = "".join(
        f"<tr id='{_e(r.anchor) or (_e(r.timestamp))}'><td>{_e(r.timestamp)}</td><td>{_e(r.speaker)}</td>"
        f"<td>{_e(r.source_text or r.label)}</td><td>{r.chunk_index}</td>"
        f"<td>{_e(_ELEMENT_TYPE_LABELS.get(r.element_type, r.element_type))}</td></tr>"
        for r in timestamp_refs
    )
    return (
        "<h2 id='appendix'>Приложение: таймкоды</h2>"
        "<table><thead><tr><th>Таймкод</th><th>Говорящий</th><th>Реплика</th><th>Блок</th><th>Тип</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def build_body(doc: ProtocolDocument) -> str:
    meta = (
        f"<div class='meta'>Файл: {_e(doc.source_filename)} &middot; "
        f"Обработано: {_e(doc.processed_at)} &middot; "
        f"Длительность: {_e(doc.duration_seconds)} с &middot; "
        f"Участники: {_e(', '.join(doc.participants) or 'не указаны')} &middot; "
        f"Модель: {_e(doc.model_id)}</div>"
    )

    # (anchor, TOC label, rendered section HTML) -- a section and its TOC
    # entry are always hidden together (item 12: hide empty sections).
    parts = [
        ("summary", "Резюме", f"<h2 id='summary'>Краткое резюме</h2><p>{_e(doc.summary)}</p>" if doc.summary else ""),
        ("topics", "Темы", _topics_section(doc.topics)),
        ("decisions", "Решения", _decisions_table(doc.decisions)),
        ("tasks", "Поручения", _tasks_table(doc.tasks)),
        ("open_questions", "Открытые вопросы", _list_section("Открытые вопросы", "open_questions", doc.open_questions)),
        ("risks", "Риски", _list_section("Риски", "risks", doc.risks)),
        ("disagreements", "Разногласия", _list_section("Разногласия", "disagreements", doc.disagreements)),
        ("next_steps", "Следующие шаги", _list_section("Следующие шаги", "next_steps", doc.next_steps)),
        ("unverified", "Неподтверждённые пункты",
         _list_section("Неподтверждённые пункты", "unverified", doc.unverified_items, unverified=True)),
        ("appendix", "Таймкоды", _appendix(doc.timestamp_refs)),
    ]
    present = [(anchor, label, section) for anchor, label, section in parts if section]
    toc = "<nav class='toc'>" + "".join(f"<a href='#{a}'>{_e(t)}</a>" for a, t, _ in present) + "</nav>"
    sections = "".join(section for _, _, section in present)

    actions = (
        "<div class='actions'>"
        "<button onclick='window.print()'>🖨 Печать</button>"
        "<a href='protocol.html' download>Скачать HTML</a>"
        "<a href='protocol.json' download>Скачать JSON</a>"
        "<a href='protocol.xml' download>Скачать XML</a>"
        "</div>"
    )
    return f"<h1>{_e(doc.meeting_title)}</h1>{meta}{actions}{toc}{sections}"


def render_html(doc: ProtocolDocument, *, template: str) -> str:
    """Render doc into the (admin-editable) page shell. `template` must
    contain a $body placeholder (validated by prompts.py before it's ever
    saved) and may use $title; substituted with string.Template, which only
    ever does plain text substitution -- no attribute access, no code
    execution, unlike str.format() on an untrusted format string.
    """
    body = build_body(doc)
    page = Template(template).safe_substitute(title=_e(doc.meeting_title), body=body)
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{_e(doc.meeting_title)}</title><style>{_PAGE_CSS}</style>"
        f"{page}"
    )


# --- XML (item 11) -------------------------------------------------------
#
# xml.etree.ElementTree escapes &, <, >, ", ' automatically for both text
# and attribute content -- built from the same validated ProtocolDocument
# as the HTML/JSON, never independently, so all three formats always agree.

def _xml_text(parent: ET.Element, tag: str, value: Any) -> ET.Element:
    el = ET.SubElement(parent, tag)
    el.text = "" if value is None else str(value)
    return el


def _xml_verified(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _xml_item_list(parent: ET.Element, wrapper_tag: str, items: tuple) -> None:
    wrapper = ET.SubElement(parent, wrapper_tag)
    for item in items:
        _xml_text(wrapper, "item", item)


def build_xml_tree(doc: ProtocolDocument) -> ET.Element:
    root = ET.Element("protocol")

    meta = ET.SubElement(root, "metadata")
    _xml_text(meta, "title", doc.meeting_title)
    _xml_text(meta, "sourceFile", doc.source_filename)
    _xml_text(meta, "processedAt", doc.processed_at)
    _xml_text(meta, "durationSeconds", doc.duration_seconds)
    _xml_text(meta, "model", doc.model_id)

    participants_el = ET.SubElement(root, "participants")
    for p in doc.participants:
        _xml_text(participants_el, "participant", p)

    _xml_text(root, "summary", doc.summary)

    topics_el = ET.SubElement(root, "topics")
    for t in doc.topics:
        topic_el = ET.SubElement(topics_el, "topic")
        _xml_text(topic_el, "title", t.title)
        _xml_text(topic_el, "summary", t.summary)

    decisions_el = ET.SubElement(root, "decisions")
    for d in doc.decisions:
        d_el = ET.SubElement(decisions_el, "decision")
        _xml_text(d_el, "text", d.text)
        _xml_text(d_el, "speaker", d.speaker)
        _xml_text(d_el, "timestampStart", d.timestamp_start)
        _xml_text(d_el, "timestampEnd", d.timestamp_end)
        _xml_text(d_el, "confidence", f"{d.confidence:.2f}")
        _xml_text(d_el, "verified", _xml_verified(d.verified))
        _xml_text(d_el, "verificationReason", d.verification_reason)

    tasks_el = ET.SubElement(root, "tasks")
    for t in doc.tasks:
        t_el = ET.SubElement(tasks_el, "task")
        _xml_text(t_el, "text", t.task)
        _xml_text(t_el, "owner", t.owner)
        _xml_text(t_el, "deadline", t.deadline)
        _xml_text(t_el, "speaker", t.speaker)
        _xml_text(t_el, "timestampStart", t.timestamp_start)
        _xml_text(t_el, "timestampEnd", t.timestamp_end)
        _xml_text(t_el, "confidence", f"{t.confidence:.2f}")
        _xml_text(t_el, "verified", _xml_verified(t.verified))
        _xml_text(t_el, "verificationReason", t.verification_reason)

    _xml_item_list(root, "openQuestions", doc.open_questions)
    _xml_item_list(root, "risks", doc.risks)
    _xml_item_list(root, "disagreements", doc.disagreements)
    _xml_item_list(root, "nextSteps", doc.next_steps)
    _xml_item_list(root, "unverifiedItems", doc.unverified_items)
    return root


def render_xml(doc: ProtocolDocument) -> str:
    root = build_xml_tree(doc)
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def document_to_json_text(doc: ProtocolDocument) -> str:
    return json.dumps(doc.to_json_dict(), ensure_ascii=False, indent=2)
