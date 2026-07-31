"""Default prompt texts. Admin-edited versions live in the DB (store.py);
these are the fallback / "restore default" values and are never mutated.
"""
from __future__ import annotations

REQUIRED_VARIABLES: dict[str, tuple[str, ...]] = {
    "chunk_analysis": ("{transcript_chunk}",),
    "topic_split": ("{transcript}",),
    "merge": ("{chunk_analyses}",),
    "fact_check": ("{document}", "{transcript}"),
    "html_template": ("$body",),
}

DEFAULT_PROMPTS: dict[str, str] = {
    "chunk_analysis": """Ты — ассистент, который готовит протокол совещания строго по расшифровке разговора.
Тебе дан фрагмент расшифровки с таймкодами и именами говорящих.

ПРАВИЛА:
- Используй только то, что явно сказано в тексте фрагмента.
- Если ответственный, срок или другой факт не назван явно — пиши "не указан".
- Запрещено придумывать решения, сроки, ответственных, участников, цифры, договорённости, причины или выводы, которых нет в тексте.
- Ответ — строго один JSON-объект без пояснений и без markdown-разметки.

Фрагмент расшифровки:
{transcript_chunk}

Верни JSON строго такой структуры:
{{"topic": "", "summary": "", "facts": [], "decisions": [{{"text": "", "speaker": "", "timestamp": "", "confidence": 0.0}}], "tasks": [{{"task": "", "owner": "", "deadline": "", "speaker": "", "timestamp": "", "confidence": 0.0}}], "open_questions": [], "risks": [], "disagreements": [], "terms": []}}""",

    "topic_split": """Проанализируй расшифровку совещания и определи, можно ли надёжно разделить её по темам обсуждения.
Если да — верни список границ тем с таймкодами начала каждой темы.
Если тематическое разбиение ненадёжно (темы сильно переплетены, обсуждение хаотичное) — верни пустой список: в этом случае будет использовано разбиение по времени.

Расшифровка:
{transcript}

Верни JSON строго такой структуры:
{{"topics": [{{"title": "", "start_timestamp": ""}}]}}""",

    "merge": """Тебе даны результаты анализа отдельных фрагментов одного совещания в формате JSON.
Собери из них единый протокол.

ПРАВИЛА:
- Объедини одинаковые темы, удали дубли.
- Сохраняй порядок обсуждения.
- Разрешай противоречия только на основании данных из фрагментов, ничего не придумывай.
- Не объединяй разные поручения только из-за похожих формулировок.
- Сохраняй ссылки на таймкоды и говорящих.
- Отдельно перечисли пункты, где ответственный, срок или факт помечен как "не указан" (unverified_items).

Результаты анализа фрагментов:
{chunk_analyses}

Верни JSON строго такой структуры:
{{"summary": "", "topics": [], "decisions": [], "tasks": [], "open_questions": [], "risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}}""",

    "fact_check": """Проверь, что каждый пункт итогового протокола подтверждается текстом расшифровки.
Если пункт не подтверждается — пометь его как неподтверждённый, не удаляй и не изменяй сам текст.

Протокол:
{document}

Расшифровка:
{transcript}

Верни JSON строго такой структуры:
{{"unverified_items": []}}""",

    "html_template": """<article>
$body
</article>""",
}


def get_default_prompt(kind: str) -> str:
    if kind not in DEFAULT_PROMPTS:
        raise KeyError(f"Unknown prompt kind: {kind}")
    return DEFAULT_PROMPTS[kind]


def validate_prompt_content(kind: str, content: str) -> None:
    """Raise ValueError if content is empty or missing required template variables."""
    if not content or not content.strip():
        raise ValueError("Промпт не может быть пустым")
    missing = [var for var in REQUIRED_VARIABLES.get(kind, ()) if var not in content]
    if missing:
        raise ValueError(f"В промпте отсутствуют обязательные переменные: {', '.join(missing)}")
