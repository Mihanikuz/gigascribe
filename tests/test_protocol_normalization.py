"""Item 5/6/9: JSON normalization (string vs object list items, no
str(dict) leakage), the structured Topic type, and order-preserving dedup.
"""
from protocol.schemas import (
    ChunkAnalysis, DecisionItem, Topic, TaskItem, as_str_list, dedup_items_preserve_order,
    dedup_preserve_order, normalize_text,
)


def test_normalize_text_passes_through_plain_strings():
    assert normalize_text("  риск потери данных  ") == "риск потери данных"


def test_normalize_text_extracts_known_field_names_from_objects():
    assert normalize_text({"question": "Когда релиз?"}) == "Когда релиз?"
    assert normalize_text({"risk": "Задержка поставки"}) == "Задержка поставки"
    assert normalize_text({"fact": "Бюджет утверждён"}) == "Бюджет утверждён"
    assert normalize_text({"topic": "Архитектура"}) == "Архитектура"
    assert normalize_text({"text": "Общий пункт"}) == "Общий пункт"
    assert normalize_text({"task": "Подготовить отчёт"}) == "Подготовить отчёт"


def test_normalize_text_never_produces_python_dict_repr():
    result = normalize_text({"unexpected_key": "value"})
    assert "{'" not in result
    assert "unexpected_key" not in result
    assert result == ""  # no recognizable field -> dropped, not stringified


def test_as_str_list_drops_unrecognizable_objects_and_keeps_good_ones():
    raw = ["риск A", {"risk": "риск B"}, {"nonsense": "junk"}, {"text": "риск C"}]
    result = as_str_list(raw)
    assert result == ["риск A", "риск B", "риск C"]
    assert all("{" not in r for r in result)


def test_chunk_analysis_normalizes_object_shaped_list_items():
    raw = {
        "topic": "t", "summary": "s", "facts": [{"fact": "Бюджет одобрен"}],
        "decisions": [], "tasks": [], "open_questions": [{"question": "Кто отвечает?"}],
        "risks": [{"risk": "Сроки сдвинутся"}], "disagreements": [], "terms": [],
    }
    analysis = ChunkAnalysis.from_llm_json(0, raw)
    assert analysis.facts == ("Бюджет одобрен",)
    assert analysis.open_questions == ("Кто отвечает?",)
    assert analysis.risks == ("Сроки сдвинутся",)


def test_topic_from_json_accepts_plain_string():
    t = Topic.from_json("Обсуждение бюджета")
    assert t.title == "Обсуждение бюджета"
    assert t.summary == ""


def test_topic_from_json_accepts_object_with_title_and_summary():
    t = Topic.from_json({"title": "Архитектура", "summary": "Обсудили переход на микросервисы."})
    assert t.title == "Архитектура"
    assert t.summary == "Обсудили переход на микросервисы."


def test_topic_from_json_falls_back_to_topic_or_text_field():
    assert Topic.from_json({"topic": "X"}).title == "X"
    assert Topic.from_json({"text": "Y"}).title == "Y"


def test_topic_from_json_rejects_object_without_any_title_field():
    assert Topic.from_json({"unrelated": "value"}) is None
    assert Topic.from_json({}) is None
    assert Topic.from_json(123) is None


def test_dedup_preserve_order_removes_exact_and_near_duplicates():
    items = ("Требует уточнения.", "требует уточнения", "  Требует   уточнения  ", "Другой пункт.")
    result = dedup_preserve_order(items)
    assert result == ("Требует уточнения.", "Другой пункт.")


def test_dedup_preserve_order_folds_yo_and_ye():
    items = ("Всё сделано", "Все сделано")
    assert dedup_preserve_order(items) == ("Всё сделано",)


def test_dedup_preserve_order_keeps_distinct_items_and_original_order():
    items = ("Первый", "Второй", "Третий")
    assert dedup_preserve_order(items) == items


def test_dedup_items_preserve_order_for_decisions():
    d1 = DecisionItem(text="Принять предложение.")
    d2 = DecisionItem(text="принять предложение", speaker="Иван")  # duplicate text, different metadata
    d3 = DecisionItem(text="Другое решение")
    result = dedup_items_preserve_order((d1, d2, d3), lambda d: d.text)
    assert result == (d1, d3)


def test_dedup_items_preserve_order_for_tasks():
    t1 = TaskItem(task="Подготовить отчёт")
    t2 = TaskItem(task="подготовить отчёт.")
    result = dedup_items_preserve_order((t1, t2), lambda t: t.task)
    assert len(result) == 1
