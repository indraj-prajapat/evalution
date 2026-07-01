"""Pipeline planner module.

Handles planning of key points using the TenderPlanner.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from ..planner.config import PlannerConfig
from ..planner.planner import TenderPlanner


def plan_key_points(
    planner_tender: dict[str, Any],
    config: PlannerConfig,
    max_workers: int,
    skip_indices: Optional[set[int]] = None,
) -> list[Any]:
    """
    Plan all key points in parallel or sequentially.
    
    Args:
        planner_tender: Tender structure with criteria to plan.
        config: Planner configuration.
        max_workers: Maximum number of parallel workers.
        skip_indices: Set of indices to skip (e.g., validation failures).
    
    Returns:
        List of planning results, one per criterion.
    """
    criteria = planner_tender.get("key_points", [])
    skip_indices = skip_indices or set()
    results: list[Any] = [None] * len(criteria)
    
    if not skip_indices and max_workers <= 1:
        # Fast path: nothing quarantined, use the bulk planner API.
        planner = TenderPlanner(config=config)
        return planner.plan(criteria)
    
    plannable = [
        (idx, criterion)
        for idx, criterion in enumerate(criteria)
        if idx not in skip_indices
    ]
    
    if max_workers <= 1:
        for idx, criterion in plannable:
            _, result = _plan_one_key_point(idx, criterion, config)
            results[idx] = result
        return results
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_plan_one_key_point, idx, criterion, config): idx
            for idx, criterion in plannable
        }
        for future in as_completed(future_map):
            idx, result = future.result()
            results[idx] = result
    
    return results


def _plan_one_key_point(
    idx: int,
    criterion: dict[str, str],
    config: PlannerConfig,
) -> tuple[int, Any]:
    """
    Plan a single key point.
    
    Args:
        idx: Index of the key point.
        criterion: Criterion dictionary with id and text.
        config: Planner configuration.
    
    Returns:
        Tuple of (index, planning result).
    """
    planner = TenderPlanner(config=config)
    result = planner.plan_single(
        criterion_id=criterion.get("criterion_id", f"CRIT{idx + 1:03d}"),
        criterion_text=criterion.get("text", ""),
    )
    return idx, result
