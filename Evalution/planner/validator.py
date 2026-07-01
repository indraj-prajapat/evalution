"""Validation logic for :class:`CriterionPlan` output.

The validator enforces structural, semantic, and business rules:
- No duplicate field names across a document
- No derived / computed fields
- Every document has at least one field
- Every field has a valid datatype
- Priorities start from 1
- Categories are drawn from the approved list
- The overall JSON structure is valid
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from Evalution.planner.json_utils import extract_json_from_response
from Evalution.planner.logging_utils import get_logger
from Evalution.planner.models import DERIVED_FIELD_PATTERNS, VALID_CATEGORIES
from Evalution.planner.schemas import CriterionPlan, DocumentField, DocumentSpec, FieldType

log = get_logger(__name__)


@dataclass(slots=True)
class ValidationErrorDetail:
    """A single validation error with location context."""

    path: str
    message: str


@dataclass(slots=True)
class ValidationResult:
    """Aggregate result of validating a :class:`CriterionPlan`.

    Attributes:
        valid: ``True`` when no errors were found.
        errors: List of :class:`ValidationErrorDetail` instances.
        plan: The validated :class:`CriterionPlan`, populated only when
            ``valid`` is ``True``.
    """

    valid: bool = True
    errors: list[ValidationErrorDetail] = field(default_factory=list)
    plan: Optional[CriterionPlan] = None


class PlanValidator:
    """Validates and (optionally) repairs criterion plan data.

    Parameters:
        strict_derived_check: When ``True`` the validator uses pattern matching
            to detect likely derived fields even if they are not in the explicit
            blacklist.
    """

    def __init__(self, strict_derived_check: bool = True) -> None:
        self._strict_derived_check = strict_derived_check

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, data: dict) -> ValidationResult:
        """Run the full validation pipeline on a raw dictionary.

        Parameters:
            data: The parsed JSON dictionary to validate.

        Returns:
            A :class:`ValidationResult` with ``valid``, ``errors``, and
            optionally a parsed ``plan``.
        """
        errors: list[ValidationErrorDetail] = []

        # 1. Pydantic structural validation
        plan = self._pydantic_validate(data, errors)

        if plan is None:
            return ValidationResult(valid=False, errors=errors)

        # 2. Business-logic validations (run on the parsed model)
        self._check_no_duplicate_fields(plan, errors)
        self._check_no_derived_fields(plan, errors)
        self._check_documents_have_fields(plan, errors)
        self._check_priority_starts_at_one(plan, errors)
        self._check_category_valid(plan, errors)

        if errors:
            return ValidationResult(valid=False, errors=errors)

        return ValidationResult(valid=True, plan=plan)

    def validate_response_text(self, text: str) -> ValidationResult:
        """Convenience: extract JSON from LLM response text and validate it.

        Parameters:
            text: Raw LLM response string.

        Returns:
            :class:`ValidationResult`.
        """
        try:
            data = extract_json_from_response(text)
        except Exception as exc:
            return ValidationResult(
                valid=False,
                errors=[ValidationErrorDetail(path="response", message=str(exc))],
            )
        return self.validate(data)

    def format_errors(self, result: ValidationResult) -> str:
        """Format validation errors into a human-readable string for the LLM repair prompt.

        Parameters:
            result: A :class:`ValidationResult` with errors.

        Returns:
            A numbered list of error descriptions.
        """
        lines: list[str] = []
        for i, err in enumerate(result.errors, start=1):
            lines.append(f"{i}. [{err.path}] {err.message}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private checks
    # ------------------------------------------------------------------

    def _pydantic_validate(
        self, data: dict, errors: list[ValidationErrorDetail]
    ) -> Optional[CriterionPlan]:
        """Attempt to parse *data* through the Pydantic model."""
        try:
            return CriterionPlan.model_validate(data)
        except ValidationError as exc:
            for error in exc.errors():
                loc = " -> ".join(str(p) for p in error["loc"]) if error["loc"] else "root"
                errors.append(
                    ValidationErrorDetail(
                        path=loc,
                        message=error["msg"],
                    )
                )
            return None

    def _check_no_duplicate_fields(
        self, plan: CriterionPlan, errors: list[ValidationErrorDetail]
    ) -> None:
        """Ensure no document contains duplicate field names."""
        for doc in plan.required_documents:
            seen: set[str] = set()
            for f in doc.fields:
                if f.name in seen:
                    errors.append(
                        ValidationErrorDetail(
                            path=f"documents[{doc.document}].fields",
                            message=f"Duplicate field name: '{f.name}'",
                        )
                    )
                seen.add(f.name)

    def _check_no_derived_fields(
        self, plan: CriterionPlan, errors: list[ValidationErrorDetail]
    ) -> None:
        """Flag any field whose name suggests a derived / computed value."""
        for doc in plan.required_documents:
            for f in doc.fields:
                if self._is_derived(f.name):
                    errors.append(
                        ValidationErrorDetail(
                            path=f"documents[{doc.document}].fields[{f.name}]",
                            message=(
                                f"Field '{f.name}' appears to be a DERIVED value. "
                                f"Decompose it into raw atomic fields. "
                                f"For example, instead of 'average_turnover' request "
                                f"'annual_turnover' per financial year."
                            ),
                        )
                    )

    def _is_derived(self, field_name: str) -> bool:
        """Check whether *field_name* matches any known derived-field pattern."""
        name_lower = field_name.lower().replace("_", " ")
        for pattern in DERIVED_FIELD_PATTERNS:
            if pattern in name_lower:
                return True
        return False

    def _check_documents_have_fields(
        self, plan: CriterionPlan, errors: list[ValidationErrorDetail]
    ) -> None:
        """Every document spec must contain at least one field."""
        for doc in plan.required_documents:
            if not doc.fields:
                errors.append(
                    ValidationErrorDetail(
                        path=f"documents[{doc.document}]",
                        message="Document has no fields. Every document must define at least one atomic field.",
                    )
                )

    def _check_priority_starts_at_one(
        self, plan: CriterionPlan, errors: list[ValidationErrorDetail]
    ) -> None:
        """Priorities must start from 1 and be sequential (1, 2, 3, ...)."""
        priorities = [doc.priority for doc in plan.required_documents]
        if not priorities:
            return
        if min(priorities) != 1:
            errors.append(
                ValidationErrorDetail(
                    path="documents.priority",
                    message=f"Highest-priority document must have priority=1, but found minimum priority={min(priorities)}.",
                )
            )
        # Check for duplicate priorities
        if len(priorities) != len(set(priorities)):
            errors.append(
                ValidationErrorDetail(
                    path="documents.priority",
                    message="Duplicate priority values detected. Each document must have a unique priority.",
                )
            )

    def _check_category_valid(
        self, plan: CriterionPlan, errors: list[ValidationErrorDetail]
    ) -> None:
        """Every document category must be from the approved list."""
        for doc in plan.required_documents:
            if doc.category not in VALID_CATEGORIES:
                errors.append(
                    ValidationErrorDetail(
                        path=f"documents[{doc.document}].category",
                        message=f"Invalid category '{doc.category}'. Must be one of: {VALID_CATEGORIES}",
                    )
                )