"""Debug utilities for the evaluation pipeline.

This module provides debugging capabilities to track inputs, outputs, 
and intermediate steps throughout the pipeline execution.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class DebugLogger:
    """Centralized debug logger for tracking pipeline execution."""
    
    def __init__(self, enabled: bool = False, save_dir: str | None = None):
        """
        Initialize the debug logger.
        
        Args:
            enabled: Whether debug logging is enabled.
            save_dir: Directory to save debug files. Defaults to ./debug_logs.
        """
        self.enabled = enabled
        self.save_dir = Path(save_dir) if save_dir else Path.cwd() / "debug_logs"
        self.step_logs: list[dict[str, Any]] = []
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def log_step(
        self,
        step_name: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a pipeline step's input, output, and metadata.
        
        Args:
            step_name: Name of the pipeline step.
            input_data: Input to the step (will be serialized).
            output_data: Output from the step (will be serialized).
            metadata: Additional metadata about the step.
            error: Error message if the step failed.
        """
        if not self.enabled:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "input": self._serialize(input_data),
            "output": self._serialize(output_data),
            "metadata": metadata or {},
            "error": error,
        }
        
        self.step_logs.append(log_entry)
        
        # Save individual step file
        step_filename = f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        step_path = self.save_dir / "steps" / step_filename
        step_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False, default=str)
    
    def _serialize(self, data: Any) -> Any:
        """Safely serialize data for JSON storage."""
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)
    
    def save_summary(self, filename: str = "debug_summary.json") -> None:
        """Save a summary of all logged steps."""
        if not self.enabled or not self.step_logs:
            return
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_steps": len(self.step_logs),
            "steps": self.step_logs,
        }
        
        summary_path = self.save_dir / filename
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    def get_step_logs(self) -> list[dict[str, Any]]:
        """Return all logged steps."""
        return self.step_logs
    
    def clear(self) -> None:
        """Clear all logged steps."""
        self.step_logs.clear()


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get the global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(enabled=False)
    return _debug_logger


def set_debug_logger(logger: DebugLogger) -> None:
    """Set the global debug logger instance."""
    global _debug_logger
    _debug_logger = logger


def log_step(
    step_name: str,
    input_data: Any,
    output_data: Any,
    metadata: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Convenience function to log a step using the global logger."""
    logger = get_debug_logger()
    logger.log_step(step_name, input_data, output_data, metadata, error)
