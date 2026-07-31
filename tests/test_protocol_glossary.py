"""Glossary CRUD, application, and the four suggestion sources (items 18-21)."""
import tempfile
from pathlib import Path

import pytest

import job_store
import protocol.glossary as gl
from protocol.schemas import GlossaryTerm
from protocol.store import ProtocolStore


@pytest.fixture()
def store():
    tmp = Path(tempfile.mkdtemp()) / "jobs.sqlite3"
    job_store.JobStore(tmp)
    return ProtocolStore(tmp)


def test_manual_term_can_be_added_and_applied(store):
    store.add_term(GlossaryTerm(id=None, canonical="PostgreSQL", aliases=("постгрес", "постгресс"), category="technology"))
    terms = store.resolved_terms()
    text = "Мы используем постгрес для аналитики."
    fixed, corrections = gl.apply_safe_replacements(text, terms, job_id="job-1", store=store)
    assert "PostgreSQL" in fixed
    assert corrections and corrections[0].corrected_text == "PostgreSQL"


def test_word_boundary_prevents_partial_alias_match(store):
    store.add_term(GlossaryTerm(id=None, canonical="PostgreSQL", aliases=("постгрес",)))
    terms = store.resolved_terms()
    text = "постгресоидный подход не то же самое"
    fixed, corrections = gl.apply_safe_replacements(text, terms, job_id=None)
    assert fixed == text
    assert not corrections


def test_llm_proposed_term_is_not_applied_until_confirmed(store):
    sug = gl.propose_from_llm(store, wrong_text="постгрес скуль", suggested_text="PostgreSQL",
                               context="c", timestamp="00:01", confidence=0.6, job_id="job-1")
    assert sug.status == "proposed"
    terms_before = store.resolved_terms()
    assert not any("постгрес скуль" in t.aliases for t in terms_before)
    text = "мы обсуждали постгрес скуль вчера"
    fixed, _ = gl.apply_safe_replacements(text, terms_before, job_id=None)
    assert fixed == text  # unconfirmed suggestion must not affect replacement


def test_confirming_suggestion_makes_it_applicable(store):
    sug = gl.propose_from_llm(store, wrong_text="постгрес скуль", suggested_text="PostgreSQL",
                               context="", timestamp="", confidence=0.6)
    gl.confirm_suggestion_as_term(store, sug, resolved_by="admin")
    terms = store.resolved_terms()
    fixed, _ = gl.apply_safe_replacements("постгрес скуль", terms, job_id=None)
    assert fixed == "PostgreSQL"


def test_user_correction_creates_a_suggestion_not_a_direct_edit(store):
    sug = gl.propose_from_user_correction(store, wrong_text="азуре", suggested_text="Azure", job_id="job-2")
    assert sug.status == "proposed"
    assert sug.source == "user_correction"
    pending = store.list_suggestions("proposed")
    assert any(s.id == sug.id for s in pending)


def test_document_diff_only_proposes_repeated_substitutions(store):
    original = "мы используем азуре и азуре снова и снова"
    corrected = "мы используем Azure и Azure снова и снова"
    suggestions = gl.propose_from_document_diff(store, original_transcript=original,
                                                 corrected_transcript=corrected, job_id="job-3", min_repeats=2)
    assert suggestions
    assert all(s.status == "proposed" for s in suggestions)


def test_reject_suggestion_does_not_create_term(store):
    sug = gl.propose_from_user_correction(store, wrong_text="азуре", suggested_text="Azure")
    store.resolve_suggestion(sug.id, status="rejected", resolved_by="admin")
    assert store.get_suggestion(sug.id).status == "rejected"
    assert not any(t.canonical == "Azure" for t in store.list_terms())


def test_import_export_json_roundtrip(store):
    store.add_term(GlossaryTerm(id=None, canonical="PostgreSQL", aliases=("постгрес",), category="tech"))
    exported = gl.export_terms_json(store.resolved_terms())

    tmp2 = Path(tempfile.mkdtemp()) / "jobs.sqlite3"
    job_store.JobStore(tmp2)
    store2 = ProtocolStore(tmp2)
    n = gl.import_terms_json(store2, exported)
    assert n == 1
    assert any(t.canonical == "PostgreSQL" for t in store2.list_terms())


def test_import_export_csv_roundtrip(store):
    store.add_term(GlossaryTerm(id=None, canonical="Kubernetes", aliases=("кубер", "к8с"), category="tech"))
    exported = gl.export_terms_csv(store.resolved_terms())
    assert "Kubernetes" in exported

    tmp2 = Path(tempfile.mkdtemp()) / "jobs.sqlite3"
    job_store.JobStore(tmp2)
    store2 = ProtocolStore(tmp2)
    n = gl.import_terms_csv(store2, exported)
    assert n == 1
    imported_term = next(t for t in store2.list_terms() if t.canonical == "Kubernetes")
    assert set(imported_term.aliases) == {"кубер", "к8с"}
