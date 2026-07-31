"""Glossary application pipeline and the four ways entries get proposed.

Automatic, unconditional replacement is intentionally not offered anywhere
here beyond confirmed terms with an exact, word-boundary match: anything
fuzzier (LLM proposals, user corrections, diffs against corrected
documents) becomes a `glossary_suggestions` row with status "proposed" and
is never applied on its own.
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Optional

from .schemas import GlossaryCorrection, GlossarySuggestion, GlossaryTerm
from .store import ProtocolStore


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)


def apply_safe_replacements(text: str, terms: list[GlossaryTerm], *, job_id: Optional[str],
                             store: Optional[ProtocolStore] = None) -> tuple[str, list[GlossaryCorrection]]:
    """Replace confirmed aliases with their canonical form using word-boundary
    matches only (never a blind substring replace). Returns the corrected
    text and the list of corrections made (also logged to `store` if given).
    """
    corrections: list[GlossaryCorrection] = []
    result = text
    # Longer aliases first so "постгрес скуль" doesn't get partially matched
    # by a shorter alias that happens to be a prefix of it.
    ordered = sorted(
        ((term, alias) for term in terms for alias in term.aliases if alias.lower() != term.canonical.lower()),
        key=lambda pair: len(pair[1]), reverse=True,
    )
    for term, alias in ordered:
        pattern = _alias_pattern(alias)

        def _sub(match: re.Match, term=term) -> str:
            original = match.group(0)
            corrections.append(GlossaryCorrection(
                id=None, job_id=job_id, original_text=original, corrected_text=term.canonical,
                rule=f"alias:{alias}", source="glossary",
            ))
            if store is not None:
                store.bump_usage(term.id)
            return term.canonical

        result = pattern.sub(_sub, result)
    if store is not None:
        for c in corrections:
            store.log_correction(c)
    return result, corrections


# --- sources of new suggestions -----------------------------------------

def propose_from_user_correction(store: ProtocolStore, *, wrong_text: str, suggested_text: str,
                                  job_id: Optional[str] = None, context: str = "") -> GlossarySuggestion:
    return store.add_suggestion(GlossarySuggestion(
        id=None, source="user_correction", wrong_text=wrong_text, suggested_text=suggested_text,
        context=context, job_id=job_id, confidence=1.0,
    ))


def propose_from_llm(store: ProtocolStore, *, wrong_text: str, suggested_text: str, context: str,
                      timestamp: str, confidence: float, job_id: Optional[str] = None) -> GlossarySuggestion:
    return store.add_suggestion(GlossarySuggestion(
        id=None, source="llm_proposal", wrong_text=wrong_text, suggested_text=suggested_text,
        context=context, timestamp=timestamp, confidence=confidence, job_id=job_id,
    ))


def propose_from_document_diff(store: ProtocolStore, *, original_transcript: str, corrected_transcript: str,
                                job_id: Optional[str] = None, min_repeats: int = 2) -> list[GlossarySuggestion]:
    """Compare a raw transcript against a user-corrected version word by word
    and turn *repeated* substitutions into suggestions (a one-off edit is
    more likely a typo fix than a terminology correction).
    """
    orig_words = re.findall(r"\w+", original_transcript.lower())
    corr_words = re.findall(r"\w+", corrected_transcript.lower())
    counts: dict[tuple[str, str], int] = {}
    for o, c in zip(orig_words, corr_words):
        if o != c:
            counts[(o, c)] = counts.get((o, c), 0) + 1
    out = []
    for (wrong, right), count in counts.items():
        if count >= min_repeats:
            out.append(propose_from_llm(
                store, wrong_text=wrong, suggested_text=right, context="document_diff",
                timestamp="", confidence=min(1.0, count / 5), job_id=job_id,
            ))
    return out


# --- confirm / reject applies the suggestion into the term table --------

def confirm_suggestion_as_term(store: ProtocolStore, suggestion: GlossarySuggestion, *, resolved_by: str,
                                scope: str = "global", project: Optional[str] = None,
                                category: str = "") -> GlossaryTerm:
    existing = next((t for t in store.list_terms(status="confirmed")
                      if t.canonical.lower() == suggestion.suggested_text.lower()), None)
    if existing:
        store.add_alias(existing.id, suggestion.wrong_text)
        term = store.get_term(existing.id)
    else:
        term = store.add_term(GlossaryTerm(
            id=None, canonical=suggestion.suggested_text, aliases=(suggestion.wrong_text,),
            category=category, scope=scope, project=project, status="confirmed",
        ))
    store.resolve_suggestion(suggestion.id, status="confirmed", resolved_by=resolved_by)
    return term


# --- import / export -----------------------------------------------------

def export_terms_json(terms: list[GlossaryTerm]) -> str:
    return json.dumps([
        {"canonical": t.canonical, "aliases": list(t.aliases), "category": t.category, "context": t.context,
         "scope": t.scope, "project": t.project, "status": t.status}
        for t in terms
    ], ensure_ascii=False, indent=2)


def import_terms_json(store: ProtocolStore, data: str) -> int:
    rows = json.loads(data)
    count = 0
    for row in rows:
        store.add_term(GlossaryTerm(
            id=None, canonical=row["canonical"], aliases=tuple(row.get("aliases") or ()),
            category=row.get("category", ""), context=row.get("context", ""),
            scope=row.get("scope", "global"), project=row.get("project"), status=row.get("status", "confirmed"),
        ))
        count += 1
    return count


def export_terms_csv(terms: list[GlossaryTerm]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["canonical", "aliases", "category", "context", "scope", "project", "status"])
    for t in terms:
        writer.writerow([t.canonical, "|".join(t.aliases), t.category, t.context, t.scope, t.project or "", t.status])
    return buf.getvalue()


def import_terms_csv(store: ProtocolStore, data: str) -> int:
    reader = csv.DictReader(io.StringIO(data))
    count = 0
    for row in reader:
        aliases = tuple(a for a in (row.get("aliases") or "").split("|") if a)
        store.add_term(GlossaryTerm(
            id=None, canonical=row["canonical"], aliases=aliases, category=row.get("category", ""),
            context=row.get("context", ""), scope=row.get("scope") or "global",
            project=row.get("project") or None, status=row.get("status") or "confirmed",
        ))
        count += 1
    return count
