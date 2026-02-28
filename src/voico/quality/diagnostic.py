import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


@dataclass
class DiagnosticEvent:
    timestamp: str
    stage: str
    event_type: str
    data: Dict[str, Any]


@dataclass
class PipelineDiagnostics:
    pipeline_id: str
    input_file: str
    output_file: str
    quality_preset: str
    start_time: str
    end_time: str
    total_duration_seconds: float
    stage_timings: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    events: List[DiagnosticEvent] = field(default_factory=list)

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiagnosticLogger:
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.logger = logging.getLogger(__name__)
        self.diagnostics = PipelineDiagnostics(
            pipeline_id=pipeline_id,
            input_file="",
            output_file="",
            quality_preset="",
            start_time=datetime.now().isoformat(),
            end_time="",
            total_duration_seconds=0.0,
        )
        self._start_time = datetime.now()

    def log_input(
        self, input_file: str, output_file: str, quality_preset: str
    ) -> None:
        self.diagnostics.input_file = input_file
        self.diagnostics.output_file = output_file
        self.diagnostics.quality_preset = quality_preset

    def log_stage_timing(self, stage_name: str, duration_seconds: float) -> None:
        self.diagnostics.stage_timings[stage_name] = duration_seconds
        self.log_event(stage_name, "stage_completed", {"duration_s": duration_seconds})

    def log_quality_score(self, metric_name: str, score: float) -> None:
        self.diagnostics.quality_scores[metric_name] = score
        self.log_event(
            "quality", "score_recorded", {"metric": metric_name, "value": score}
        )

    def log_validation(
        self, component_name: str, passed: bool, issues: List[str]
    ) -> None:
        self.diagnostics.validation_results[component_name] = {
            "passed": passed,
            "issues": issues,
        }
        status = "passed" if passed else "failed"
        self.log_event(
            "validation",
            f"validation_{status}",
            {"component": component_name, "issues_count": len(issues)},
        )

    def log_error(self, error_message: str, stage: str = "unknown") -> None:
        self.diagnostics.errors.append(error_message)
        self.log_event(stage, "error", {"message": error_message})
        self.logger.error(f"[{self.pipeline_id}] {stage}: {error_message}")

    def log_warning(self, warning_message: str, stage: str = "unknown") -> None:
        self.diagnostics.warnings.append(warning_message)
        self.log_event(stage, "warning", {"message": warning_message})
        self.logger.warning(f"[{self.pipeline_id}] {stage}: {warning_message}")

    def log_event(
        self, stage: str, event_type: str, data: Dict[str, Any]
    ) -> None:
        event = DiagnosticEvent(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            event_type=event_type,
            data=data,
        )
        self.diagnostics.events.append(event)

    def finalize(self) -> PipelineDiagnostics:
        self.diagnostics.end_time = datetime.now().isoformat()
        elapsed = (datetime.now() - self._start_time).total_seconds()
        self.diagnostics.total_duration_seconds = elapsed
        return self.diagnostics

    def get_summary(self) -> str:
        return (
            f"Pipeline: {self.pipeline_id}\n"
            f"Input: {self.diagnostics.input_file}\n"
            f"Output: {self.diagnostics.output_file}\n"
            f"Quality Preset: {self.diagnostics.quality_preset}\n"
            f"Duration: {self.diagnostics.total_duration_seconds:.2f}s\n"
            f"Stages: {len(self.diagnostics.stage_timings)}\n"
            f"Errors: {len(self.diagnostics.errors)}\n"
            f"Warnings: {len(self.diagnostics.warnings)}\n"
        )

    def log_to_file(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.diagnostics.to_json())

    def print_summary(self) -> None:
        print(self.get_summary())
        if self.diagnostics.warnings:
            print("\nWarnings:")
            for w in self.diagnostics.warnings:
                print(f"  - {w}")
        if self.diagnostics.errors:
            print("\nErrors:")
            for e in self.diagnostics.errors:
                print(f"  - {e}")
