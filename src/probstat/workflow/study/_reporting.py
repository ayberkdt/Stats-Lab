"""Provenance, JSON record, and Markdown report of a study.

References:
    [REF-SMITH-UQ-2024, Chapter 1].

"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Self

import probstat
from probstat.uq import json_safe

from ._base import _StudyBase
from ._context import _format_settings


class _ReportingStages(_StudyBase):
    """Provenance and reporting methods of the study.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """

    def provenance(self) -> dict[str, object]:
        """Return library, Python, platform, time, and seed provenance.

        References:
            Project reporting policy. No equation is evaluated.

        """
        return {
            "library": "probstat",
            "library_version": probstat.__version__,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "created_utc": self._created,
            "root_seed": self._root.seed,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the study (no pickle needed).

        It holds provenance, the model (input laws, QoIs, fixed parameters),
        the current input law with its hash, the budget and totals, every
        stage record (:meth:`StageRecord.to_dict`), and the surrogate studies
        of :meth:`emulate`. Domain result objects are named, not serialized.

        References:
            Project reporting policy. No equation is evaluated.

        """
        model = self._model
        return {
            "format": "probstat.UQStudy/1",
            "name": self._name,
            "evaluation_kind": self._kind,
            "provenance": self.provenance(),
            "model": {
                "name": model.name,
                "inputs": [
                    {
                        "name": item.name,
                        "law": repr(item.distribution)
                        if len(repr(item.distribution)) <= 200
                        else type(item.distribution).__name__,
                        "mean": json_safe(item.mean),
                        "sd": json_safe(item.standard_deviation),
                    }
                    for item in model.inputs.inputs
                ],
                "correlation": json_safe(model.inputs.correlation),
                "quantities": [
                    {"name": q.name, "description": q.description, "unit": q.unit}
                    for q in model.quantities
                ],
                "fixed": json_safe(model.fixed),
                "on_failure": model.on_failure,
            },
            "input_law": {"label": self._law_label, "hash": self._law_hash},
            "evaluation_budget": self._budget,
            "totals": {
                "evaluations": self.total_evaluations,
                "failed_evaluations": self.total_failed_evaluations,
                "cache_hits": sum(r.cache_hits for r in self._records),
            },
            "stages": [record.to_dict() for record in self._records],
            "surrogate_studies": [
                {"after_stage": index, "study": child.to_dict()}
                for index, child in self._children
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return :meth:`to_dict` as JSON text (no NaN or infinity tokens).

        Args:
            indent: JSON indentation (``None`` for one line).

        References:
            Project reporting policy. No equation is evaluated.

        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def save_json(self, path: str | Path) -> Path:
        """Write :meth:`to_json` to a file (parents are created).

        Args:
            path: Destination path.

        Returns:
            The written path.

        References:
            Project reporting policy. No equation is evaluated.

        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    def report(self) -> str:
        """Return a Markdown report of every stage with costs and provenance.

        The header states the inputs, QoIs, dependence, current input law
        (label and hash), root seed, evaluation totals (failed runs and cache
        hits separately), the budget, and the library, Python, platform, and
        UTC creation time. Each stage states its status, cost, stream seed,
        input law, settings, and warnings, followed by its summary table.
        Surrogate studies created by :meth:`emulate` follow their emulation
        stage, with surrogate evaluations labelled as such.

        Returns:
            Markdown text: one section per stage with a two-column table.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        return "\n".join(self._report_lines(1))

    def _report_lines(self, level: int) -> list[str]:
        """Return the Markdown lines of this study at a heading level.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        hashes = "#" * level
        forward = self._kind == "forward-model"
        unit = "Forward-model evaluations" if forward else "Surrogate evaluations"
        dependence = (
            "independent"
            if self._model.inputs.is_independent
            else f"{self._model.inputs.copula_family.capitalize()} copula"
        )
        title = "UQ study" if forward else "Surrogate study"
        lines = [
            f"{hashes} {title}: {self._name}",
            "",
            f"- Inputs: {', '.join(self._model.inputs.names)}",
            f"- Quantities of interest: {', '.join(self._model.quantity_names)}",
            f"- Dependence: {dependence}",
            f"- Input law: {self._law_label} (hash {self._law_hash})",
            f"- Root seed: {self._root.seed}",
            f"- {unit}: {self.total_evaluations:,}",
        ]
        failed = self.total_failed_evaluations
        if failed:
            attempts = failed + self.total_evaluations
            lines.append(
                f"- **Failed evaluations: {failed:,} "
                f"({100.0 * failed / attempts:.3g}% of attempts)**"
            )
        hits = sum(record.cache_hits for record in self._records)
        if hits:
            lines.append(f"- Cache hits (not evaluations): {hits:,}")
        if self._budget is not None:
            lines.append(
                f"- Evaluation budget: {self._budget:,} "
                f"({self.remaining_budget:,} remaining)"
            )
        if level == 1:
            info = self.provenance()
            lines.append(
                f"- Software: probstat {info['library_version']} · Python "
                f"{info['python_version']} · {info['platform']}"
            )
            lines.append(f"- Created (UTC): {info['created_utc']}")
        lines.append("")
        children: dict[int, list[Self]] = {}
        for index, child in self._children:
            children.setdefault(index, []).append(child)
        count_label = "Evaluations" if forward else "Surrogate evaluations"
        for index, record in enumerate(self._records, start=1):
            lines.append(f"{hashes}# {index}. {record.stage}: {record.label}")
            lines.append("")
            parts = [f"{count_label}: {record.evaluations:,}"]
            if record.failed_evaluations:
                parts.append(f"failed: {record.failed_evaluations:,}")
            if record.cache_hits:
                parts.append(f"cache hits: {record.cache_hits:,}")
            parts.append(f"stream seed: {record.seed}")
            parts.append(f"input law: {record.input_law} ({record.input_law_hash})")
            if not record.complete:
                parts.append(f"status: **{record.status}**")
            lines.append(" · ".join(parts))
            if record.error:
                lines.extend(["", f"Error: {record.error}"])
            if record.settings:
                lines.extend(["", "Settings: " + _format_settings(record.settings)])
            for message in record.warnings:
                lines.extend(["", f"> **Warning:** {message}"])
            lines.append("")
            lines.append("| Quantity | Value |")
            lines.append("|---|---|")
            for key, value in record.summary.items():
                text = (
                    f"{value:.6g}"
                    if isinstance(value, float) and math.isfinite(value)
                    else str(value)
                )
                lines.append(f"| {key} | {text} |")
            lines.append("")
            for child in children.get(index, []):
                lines.extend(child._report_lines(level + 2))
        return lines
