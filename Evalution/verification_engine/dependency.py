"""
Cross-Criterion Dependency Resolution.

## Why this file exists
Each key point in tender_output["key_points"] is currently evaluated by
VerificationEngine.run() in complete isolation -- one criterion in, one
VerificationReport out, no visibility into any other key point. That's fine
for self-contained criteria ("submit a CISA certificate"), but it breaks
down for CONDITIONAL criteria whose applicability depends on a fact
established by a *different* key point, e.g.:

    "Turnover of Rs. 25 Lakhs in the last 3 years. Not applicable to
     Joint Ventures."

Whether the exemption applies depends on the bidder-type key point
elsewhere in the same tender -- data this module has no way to see on
its own.

## What this file does and does not do
This module resolves hints ALREADY DETECTED by
criterion_parser.parse_criterion() (see ParsedCriterion.cross_reference_
hints) against a `sibling_context` supplied by the caller. It does NOT:
  - build the dependency graph itself
  - decide evaluation order across key points
  - know where sibling_context comes from

## Where the missing piece belongs (explicitly out of scope here)
Building the graph and supplying sibling_context is a job for the caller
that owns the full tender_output -- almost certainly pipeline.py's
_evaluate_planned_key_points (not present in this package) or an upstream
extraction/planning stage. That caller needs to:
  1. Identify which key points establish "bidder facts" (entity type,
     MSME/startup status, turnover category, etc.) versus which key
     points merely consume them.
  2. Evaluate (or otherwise resolve) the fact-establishing key points
     before the ones that depend on them -- a topological pass over a
     small dependency graph, or a simpler two-phase design (facts first,
     then everything else) if the fact-establishing key points are
     identifiable in advance.
  3. Build a `sibling_context` dict and pass it into
     VerificationEngine.run(..., sibling_context=sibling_context) for any
     criterion whose ParsedCriterion.cross_reference_hints is non-empty.

## sibling_context contract
    {
        "<criterion_id>": {
            "verdict": "PASS" | "FAIL" | "REVIEW",
            "extracted_values": {"<category_key>": "<free-text value>", ...},
        },
        ...
    }

`extracted_values` is intentionally free-text/keyword based rather than a
fixed enum -- matching against a hint's `category` string (e.g. "joint
ventures", "MSME bidders") is done as a case-insensitive substring/keyword
overlap, not exact-match. This is a deliberately low-precision generic
first pass; a real implementation may want fuzzy matching or an LLM-based
matcher once real tender data shows how much vocabulary varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResolvedHint:
    """Outcome of trying to resolve a single cross_reference_hint."""
    phrase: str
    category: str
    effect: str  # "exempt" | "relaxed"
    status: str  # "triggered" | "not_triggered" | "unresolved"
    matched_criterion_id: str = ""
    matched_value: str = ""
    note: str = ""


@dataclass
class DependencyResolution:
    """Aggregate result across all hints found in a criterion."""
    hints: list[ResolvedHint] = field(default_factory=list)

    @property
    def any_triggered(self) -> bool:
        return any(h.status == "triggered" for h in self.hints)

    @property
    def any_unresolved(self) -> bool:
        return any(h.status == "unresolved" for h in self.hints)

    def to_notes(self) -> list[str]:
        """Human-readable notes suitable for VerificationReport.dependency_notes."""
        notes = []
        for h in self.hints:
            if h.status == "triggered":
                notes.append(
                    f"Conditional {h.effect} clause (\"{h.phrase} {h.category}\") "
                    f"appears to apply, based on key point '{h.matched_criterion_id}' "
                    f"(value: '{h.matched_value}')."
                )
            elif h.status == "unresolved":
                notes.append(
                    f"Criterion contains a conditional {h.effect} clause "
                    f"(\"{h.phrase} {h.category}\") that could not be automatically "
                    f"checked -- no sibling key point data was supplied to confirm "
                    f"or rule out whether it applies to this bidder. Needs manual review."
                )
            # "not_triggered" is the normal/expected case and doesn't need a
            # note -- the criterion should just be evaluated as written.
        return notes


def _category_matches(category: str, value: str) -> bool:
    """
    Loose, generic overlap check between a hint's free-text category
    (e.g. "joint ventures") and a sibling's free-text extracted value
    (e.g. "Joint Venture"). Deliberately conservative: requires a
    meaningful word overlap, not just any shared character, to avoid
    false positives on short common words.
    """
    cat_words = {w for w in category.lower().split() if len(w) > 3}
    val_words = {w for w in value.lower().split() if len(w) > 3}
    if not cat_words or not val_words:
        return category.lower().strip() in value.lower() or value.lower().strip() in category.lower()
    return bool(cat_words & val_words)


def resolve_cross_references(
    hints: list[dict],
    sibling_context: Optional[dict[str, dict]],
) -> DependencyResolution:
    """
    Resolve a criterion's cross_reference_hints against sibling key point
    data, if any was supplied. Pure function -- no I/O, no LLM calls, safe
    to call unconditionally even when hints is empty.

    Resolution per hint:
      - No sibling_context supplied at all -> "unresolved" for every hint
        (we have literally nothing to check against).
      - sibling_context supplied, and some sibling's extracted_values has a
        value overlapping the hint's category -> "triggered".
      - sibling_context supplied, but nothing overlaps -> "not_triggered".
        NOTE: this is a conservative-by-omission choice, not a confident
        negative -- if sibling_context is incomplete (missing the one key
        point that would have matched), this can silently under-trigger.
        Callers with a real dependency graph should ensure sibling_context
        actually includes every key point a hint could plausibly reference.
    """
    resolution = DependencyResolution()
    for hint in hints:
        category = hint.get("category", "")
        phrase = hint.get("phrase", "")
        effect = hint.get("effect", "exempt")

        if not sibling_context:
            resolution.hints.append(ResolvedHint(
                phrase=phrase, category=category, effect=effect,
                status="unresolved",
                note="No sibling_context supplied to VerificationEngine.run().",
            ))
            continue

        matched = None
        for sib_criterion_id, sib_data in sibling_context.items():
            extracted_values = (sib_data or {}).get("extracted_values") or {}
            for _key, value in extracted_values.items():
                if isinstance(value, str) and _category_matches(category, value):
                    matched = (sib_criterion_id, value)
                    break
            if matched:
                break

        if matched:
            resolution.hints.append(ResolvedHint(
                phrase=phrase, category=category, effect=effect,
                status="triggered",
                matched_criterion_id=matched[0],
                matched_value=matched[1],
            ))
        else:
            resolution.hints.append(ResolvedHint(
                phrase=phrase, category=category, effect=effect,
                status="not_triggered",
            ))

    return resolution
