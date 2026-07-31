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


def _decisions_table(decisions) -> str:
    if not decisions:
        return ""
    rows = "".join(
        f"<tr><td>{_e(d.text)}</td><td>{_e(d.speaker)}</td>"
        f"<td><a class='timestamp-link' href='#t-{_e(d.timestamp)}'>{_e(d.timestamp)}</a></td>"
        f"<td>{d.confidence:.2f}</td></tr>"
        for d in decisions
    )
    return (
        "<h2 id='decisions'>Принятые решения</h2>"
        "<table><thead><tr><th>Решение</th><th>Говорящий</th><th>Таймкод</th><th>Уверенность</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _tasks_table(tasks) -> str:
    if not tasks:
        return ""
    rows = "".join(
        f"<tr><td>{_e(t.task)}</td><td>{_e(t.owner)}</td><td>{_e(t.deadline)}</td>"
        f"<td><a class='timestamp-link' href='#t-{_e(t.timestamp)}'>{_e(t.timestamp)}</a></td></tr>"
        for t in tasks
    )
    return (
        "<h2 id='tasks'>Поручения</h2>"
        "<table><thead><tr><th>Поручение</th><th>Ответственный</th><th>Срок</th><th>Таймкод</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _appendix(timestamp_refs) -> str:
    if not timestamp_refs:
        return ""
    rows = "".join(
        f"<tr id='t-{_e(r.timestamp)}'><td>{_e(r.timestamp)}</td><td>{_e(r.speaker)}</td>"
        f"<td>{_e(r.label)}</td><td>{r.chunk_index}</td></tr>"
        for r in timestamp_refs
    )
    return (
        "<h2 id='appendix'>Приложение: таймкоды</h2>"
        "<table><thead><tr><th>Таймкод</th><th>Говорящий</th><th>Фрагмent</th><th>Блок</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def build_body(doc: ProtocolDocument) -> str:
    toc_items = [
        ("summary", "Резюме"), ("topics", "Темы"), ("decisions", "Решения"),
        ("tasks", "Поручения"), ("open_questions", "Открытые вопросы"),
        ("risks", "Риски"), ("disagreements", "Разногласия"), ("next_steps", "Следующие шаги"),
        ("unverified", "Неподтверждённые пункты"), ("appendix", "Таймкоды"),
    ]
    toc = "<nav class='toc'>" + "".join(f"<a href='#{a}'>{_e(t)}</a>" for a, t in toc_items) + "</nav>"

    meta = (
        f"<div class='meta'>Файл: {_e(doc.source_filename)} &middot; "
        f"Обработано: {_e(doc.processed_at)} &middot; "
        f"Длительность: {_e(doc.duration_seconds)} с &middot; "
        f"Участники: {_e(', '.join(doc.participants) or 'не указаны')} &middot; "
        f"Модель: {_e(doc.model_id)}</div>"
    )

    sections = [
        f"<h2 id='summary'>Краткое резюме</h2><p>{_e(doc.summary)}</p>",
        _list_section("Обсуждавшиеся темы", "topics", doc.topics),
        _decisions_table(doc.decisions),
        _tasks_table(doc.tasks),
        _list_section("Открытые вопросы", "open_questions", doc.open_questions),
        _list_section("Риски", "risks", doc.risks),
        _list_section("Разногласия", "disagreements", doc.disagreements),
        _list_section("Следующие шаги", "next_steps", doc.next_steps),
        _list_section("Неподтверждённые пункты", "unverified", doc.unverified_items, unverified=True),
        _appendix(doc.timestamp_refs),
    ]
    actions = (
        "<div class='actions'>"
        "<button onclick='window.print()'>🖨 Печать</button>"
        "<a href='protocol.html' download>Скачать HTML</a>"
        "<a href='protocol.json' download>Скачать JSON</a>"
        "</div>"
    )
    return f"<h1>{_e(doc.meeting_title)}</h1>{meta}{actions}{toc}" + "".join(sections)


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


def document_to_json_text(doc: ProtocolDocument) -> str:
    return json.dumps(doc.to_json_dict(), ensure_ascii=False, indent=2)
