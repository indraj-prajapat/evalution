"""Main orchestrator: :class:`TenderPlanner`.

This is the primary public API of the package.  It coordinates the LLM
client, the validator, and the prompt templates to turn raw tender criteria
into structured, validated :class:`CriterionPlan` objects.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from Evalution.planner.client import LLMClient, LLMResponse
from Evalution.planner.config import LLMConfig, PlannerConfig, RetryConfig
from Evalution.planner.json_utils import extract_json_from_response, to_json_string
from Evalution.planner.logging_utils import get_logger
from Evalution.planner.models import CriterionInput, PlanningResult
from Evalution.planner.prompts import (
    REPAIR_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from Evalution.planner.schemas import CriterionPlan
from Evalution.planner.validator import PlanValidator, ValidationResult

log = get_logger(__name__)


class TenderPlanner:
    """Plan the minimum atomic information required to evaluate tender criteria.

    Parameters:
        config: Full :class:`PlannerConfig` with LLM, retry, and validation
            settings.  If omitted, sensible defaults are used (requires
            ``OPENAI_API_KEY`` environment variable or explicit key).

    Example::

        planner = TenderPlanner(
            config=PlannerConfig(
                llm=LLMConfig(api_key="sk-...", model="gpt-4o"),
            )
        )
        result = planner.plan(
            criteria=[
                {"criterion_id": "CRIT001", "text": "Average annual turnover ..."},
            ]
        )
        for r in result:
            print(r.plan.model_dump_json(indent=2))
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = PlannerConfig(
                llm=LLMConfig(
                    api_key=api_key or "",
                    base_url=base_url or LLMConfig.base_url,
                    model=model or LLMConfig.model,
                )
            )

        self._client = LLMClient(
            config=self._config.llm,
            retry_config=self._config.retry,
        )
        self._validator = PlanValidator(strict_derived_check=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        criteria: list[dict[str, str]],
    ) -> list[PlanningResult]:
        """Plan information requirements for a list of tender criteria.

        Each element of *criteria* must be a dictionary with at least:

        - ``"criterion_id"`` – a unique identifier string
        - ``"text"``         – the full criterion text

        Parameters:
            criteria: List of criterion dictionaries.

        Returns:
            A list of :class:`PlanningResult` objects, one per criterion,
            in the same order as the input.
        """
        results: list[PlanningResult] = []

        for idx, crit_dict in enumerate(criteria):
            criterion_id = crit_dict.get("criterion_id", f"CRIT{idx:03d}")
            text = crit_dict.get("text", "")

            if not text.strip():
                results.append(
                    PlanningResult(
                        criterion_id=criterion_id,
                        success=False,
                        error="Empty criterion text",
                    )
                )
                continue

            result = self._plan_single(
                CriterionInput(criterion_id=criterion_id, text=text, index=idx)
            )
            results.append(result)

        return results

    def plan_single(
        self,
        criterion_id: str,
        criterion_text: str,
    ) -> PlanningResult:
        """Plan information requirements for a single criterion.

        Convenience wrapper around :meth:`plan` for one-off calls.

        Parameters:
            criterion_id: Unique identifier for the criterion.
            criterion_text: The full criterion text.

        Returns:
            A single :class:`PlanningResult`.
        """
        return self.plan(
            [{"criterion_id": criterion_id, "text": criterion_text}]
        )[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _plan_single(self, criterion: CriterionInput) -> PlanningResult:
        """Plan a single criterion with validation-repair loop."""
        log.info(
            "planning_criterion",
            criterion_id=criterion.criterion_id,
            text_length=len(criterion.text),
        )

        start = time.monotonic()

        # --- First attempt ---
        user_prompt = USER_PROMPT_TEMPLATE.format(
            criterion_id=criterion.criterion_id,
            criterion_text=criterion.text,
        )

        response = self._client.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        validation = self._validator.validate_response_text(response.content)
        attempts = 1

        if validation.valid and validation.plan is not None:
            elapsed = time.monotonic() - start
            log.info(
                "planning_succeeded",
                criterion_id=criterion.criterion_id,
                attempts=attempts,
                elapsed_seconds=round(elapsed, 3),
                document_count=len(validation.plan.required_documents),
            )
            return PlanningResult(
                criterion_id=criterion.criterion_id,
                plan=validation.plan,
                raw_response=response.content,
                attempts=attempts,
                success=True,
            )

        # --- Validation-repair loop ---
        last_response_text = response.content
        last_validation = validation

        for repair_attempt in range(1, self._config.validation_retries + 1):
            log.warning(
                "validation_failed_repairing",
                criterion_id=criterion.criterion_id,
                repair_attempt=repair_attempt,
                max_repairs=self._config.validation_retries,
                error_count=len(last_validation.errors),
            )

            error_text = self._validator.format_errors(last_validation)
            repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
                criterion_id=criterion.criterion_id,
                criterion_text=criterion.text,
                previous_response=last_response_text,
                validation_errors=error_text,
            )

            repair_response = self._client.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=repair_prompt,
            )

            repair_validation = self._validator.validate_response_text(
                repair_response.content
            )
            attempts += 1
            last_response_text = repair_response.content
            last_validation = repair_validation

            if repair_validation.valid and repair_validation.plan is not None:
                elapsed = time.monotonic() - start
                log.info(
                    "planning_succeeded_after_repair",
                    criterion_id=criterion.criterion_id,
                    attempts=attempts,
                    elapsed_seconds=round(elapsed, 3),
                    document_count=len(repair_validation.plan.required_documents),
                )
                return PlanningResult(
                    criterion_id=criterion.criterion_id,
                    plan=repair_validation.plan,
                    raw_response=repair_response.content,
                    attempts=attempts,
                    success=True,
                )

        # --- All attempts exhausted ---
        elapsed = time.monotonic() - start
        error_summary = self._validator.format_errors(last_validation)
        log.error(
            "planning_failed",
            criterion_id=criterion.criterion_id,
            attempts=attempts,
            elapsed_seconds=round(elapsed, 3),
            error_summary=error_summary,
        )
        return PlanningResult(
            criterion_id=criterion.criterion_id,
            raw_response=last_response_text,
            attempts=attempts,
            success=False,
            error=error_summary,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def plan_from_tender_json(
        self,
        tender_json: dict[str, Any],
        key_points_key: str = "key_points",
        id_key: str = "criterion_id",
        text_key: str = "text",
    ) -> list[PlanningResult]:
        """Convenience: extract criteria from a tender JSON and plan them.

        The tender JSON is expected to contain a list under ``key_points_key``
        where each element has an identifier field (``id_key``) and a text
        field (``text_key``).

        Parameters:
            tender_json: The full tender dictionary.
            key_points_key: Key in *tender_json* that holds the criteria list.
            id_key: Key in each criterion dict for the identifier.
            text_key: Key in each criterion dict for the criterion text.

        Returns:
            List of :class:`PlanningResult` objects.
        """
        criteria_list = tender_json.get(key_points_key, [])
        if not isinstance(criteria_list, list):
            raise ValueError(
                f"Expected a list at tender_json['{key_points_key}'], "
                f"got {type(criteria_list).__name__}"
            )
        criteria = [
            {"criterion_id": item.get(id_key, f"CRIT{i:03d}"), "text": item.get(text_key, "")}
            for i, item in enumerate(criteria_list)
        ]
        return self.plan(criteria)