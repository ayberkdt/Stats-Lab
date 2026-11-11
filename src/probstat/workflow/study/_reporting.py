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
from probstat.foundations.messages import (
    check_locale,
    format_number,
    message,
    message_or,
)
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
            Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is evaluated.

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
            Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is evaluated.

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
            Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is evaluated.

        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def save_json(self, path: str | Path) -> Path:
        """Write :meth:`to_json` to a file (parents are created).

        Args:
            path: Destination path.

        Returns:
            The written path.

        References:
            Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is evaluated.

        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    def report(self, *, locale: str = "en") -> str:
        """Return a Markdown report of every stage with costs and provenance.

        The header states the inputs, QoIs, dependence, current input law
        (label and hash), root seed, evaluation totals (failed runs and cache
        hits separately), the budget, and the library, Python, platform, and
        UTC creation time. Each stage states its status, cost, stream seed,
        input law, settings, and warnings, followed by its summary table.
        Surrogate studies created by :meth:`emulate` follow their emulation
        stage, with surrogate evaluations labelled as such.

        Every sentence comes from the message catalogue
        (:mod:`probstat.foundations.messages`, ADR-0085). The English report
        is unchanged byte for byte; ``locale="tr"`` gives the Turkish report
        with decimal commas. Stage kinds, labels, summary keys, settings,
        warnings, and error texts are data recorded by the stages and are
        shown as recorded.

        Args:
            locale: Catalogue locale (``"en"`` or ``"tr"``); never read from
                the environment.

        Returns:
            Markdown text: one section per stage with a two-column table.

        Raises:
            ValueError: If the locale is unknown.

        References:
            [REF-SMITH-UQ-2024, Chapter 1], [REF-UNICODE-CLDR].
            No equation is evaluated.

        """
        return "\n".join(self._report_lines(1, check_locale(locale)))

    def _report_lines(self, level: int, locale: str = "en") -> list[str]:
        """Return the Markdown lines of this study at a heading level.

        Args:
            level: Markdown heading level of the study title.
            locale: Catalogue locale.

        References:
            [REF-SMITH-UQ-2024, Chapter 1], [REF-UNICODE-CLDR].
            No equation is evaluated.

        """

        def say(key: str, **fields: object) -> str:
            """Return one catalogue sentence in the report locale.

            References:
                [REF-UNICODE-CLDR]. No equation is evaluated.

            """
            return message(key, locale, **fields)

        hashes = "#" * level
        kind = "forward" if self._kind == "forward-model" else "surrogate"
        inputs = self._model.inputs
        dependence = (
            say("study.dependence.independent")
            if inputs.is_independent
            else say(
                "study.dependence.copula", family=inputs.copula_family.capitalize()
            )
        )
        quantities = ", ".join(self._model.quantity_names)
        lines = [
            f"{hashes} " + say(f"study.title.{kind}", name=self._name),
            "",
            "- " + say("report.inputs", names=", ".join(inputs.names)),
            "- " + say("report.quantities", names=quantities),
            "- " + say("report.dependence", dependence=dependence),
            "- " + say("report.input_law", label=self._law_label, hash=self._law_hash),
            "- " + say("report.root_seed", seed=self._root.seed),
            "- " + say(f"report.total.{kind}", count=self.total_evaluations),
        ]
        failed = self.total_failed_evaluations
        if failed:
            share = 100.0 * failed / (failed + self.total_evaluations)
            failure = say("report.failed", count=failed, share=share)
            lines.append(f"- **{failure}**")
        hits = sum(record.cache_hits for record in self._records)
        if hits:
            lines.append("- " + say("report.cache_hits", count=hits))
        if self._budget is not None:
            remaining = self.remaining_budget
            lines.append(
                "- " + say("report.budget", budget=self._budget, remaining=remaining)
            )
        if level == 1:
            info = self.provenance()
            software = say(
                "report.software",
                version=info["library_version"],
                python=info["python_version"],
                platform=info["platform"],
            )
            lines.append("- " + software)
            lines.append("- " + say("report.created", created=info["created_utc"]))
        lines.append("")
        children: dict[int, list[Self]] = {}
        for index, child in self._children:
            children.setdefault(index, []).append(child)
        for index, record in enumerate(self._records, start=1):
            heading = say(
                "study.stage_heading",
                index=index,
                stage=record.stage,
                label=record.label,
            )
            lines.append(f"{hashes}# {heading}")
            lines.append("")
            parts = [say(f"report.stage.evaluations.{kind}", count=record.evaluations)]
            if record.failed_evaluations:
                failures = record.failed_evaluations
                parts.append(say("report.stage.failed", count=failures))
            if record.cache_hits:
                parts.append(say("report.stage.cache_hits", count=record.cache_hits))
            parts.append(say("report.stage.seed", seed=record.seed))
            parts.append(
                say(
                    "report.stage.input_law",
                    label=record.input_law,
                    hash=record.input_law_hash,
                )
            )
            if not record.complete:
                status = message_or(
                    f"study.status.{record.status}", record.status, locale
                )
                parts.append(say("report.stage.status", status=f"**{status}**"))
            lines.append(" · ".join(parts))
            if record.error:
                lines.extend(["", say("study.error", error=record.error)])
            if record.settings:
                settings = _format_settings(record.settings)
                lines.extend(["", say("study.settings", settings=settings)])
            for text in record.warnings:
                lines.extend(["", f"> **{say('study.warning_label')}** {text}"])
            lines.append("")
            lines.append(
                f"| {say('study.table.quantity')} | {say('study.table.value')} |"
            )
            lines.append("|---|---|")
            for key, value in record.summary.items():
                cell = (
                    format_number(value, ".6g", locale)
                    if isinstance(value, float) and math.isfinite(value)
                    else str(value)
                )
                lines.append(f"| {key} | {cell} |")
            lines.append("")
            for child in children.get(index, []):
                lines.extend(child._report_lines(level + 2, locale))
        return lines

