"""
Pure helpers for benchmark ranking explanations (UI-friendly text).

Kept out of Qt code so tests can assert on strings without widgets.
"""

from __future__ import annotations

from typing import Any, Optional

from core.ranking import get_supported_uncertainty_modes


def format_uncertainty_ranking_explanation_lines(
    ranked: list[dict[str, Any]],
    *,
    uncertainty_mode: str,
    uncertainty_weights: Optional[dict[str, float]] = None,
    uncertainty_metrics_enabled: bool = False,
    ranking_objective: str = "spec_pass_first",
) -> list[str]:
    """
    Build lines describing uncertainty ranking for the benchmark "best" panel.

    Parameters
    ----------
    ranked
        Output rows from ``rank_benchmark_suite`` (same order as ranking).
    uncertainty_mode
        One of ``ignore``, ``warn_only``, ``include_in_score``.
    uncertainty_weights
        Weights passed into ranking (e.g. ``interval_quality``).
    uncertainty_metrics_enabled
        Whether conformal interval metrics were computed during the benchmark run.
    """

    lines: list[str] = []
    meta = get_supported_uncertainty_modes().get(uncertainty_mode, {})
    dn = str(meta.get("display_name") or uncertainty_mode).strip()
    desc = str(meta.get("description") or "").strip()

    lines.append(f"Uncertainty ranking mode: {uncertainty_mode}")
    if dn and dn != uncertainty_mode:
        lines.append(f"- {dn}")
    if desc:
        lines.append(f"- {desc}")

    if uncertainty_mode != "ignore" and not uncertainty_metrics_enabled:
        lines.append(
            "Note: benchmark uncertainty (conformal) was disabled — interval metrics may be "
            "missing; ranking still runs with neutral interval quality where needed.",
        )

    if not ranked:
        return lines

    winner = ranked[0]
    uw = dict(uncertainty_weights or {})
    iqw = uw.get("interval_quality")

    if uncertainty_mode == "warn_only":
        warn = winner.get("uncertainty_warnings") or []
        if isinstance(warn, list) and warn:
            lines.append("Interval quality warnings (winner):")
            for w in warn:
                lines.append(f"- {w}")
        elif uncertainty_metrics_enabled:
            lines.append("No interval coverage warnings for the winner (within tolerance).")

    if uncertainty_mode == "include_in_score":
        if iqw is not None:
            try:
                fv = float(iqw)
            except Exception:
                fv = None
            if ranking_objective == "weighted_combined" and fv is not None:
                lines.append(f"interval_quality weight: {fv:.3f}")
            elif fv is not None:
                lines.append(
                    f"interval_quality weight ({fv:.3f}) applies to weighted combined; "
                    "for this objective uncertainty is a tie-break only.",
                )
        ew = winner.get("effective_weights") or {}
        if isinstance(ew, dict) and "interval_quality" in ew:
            try:
                lines.append(f"Effective interval_quality weight used: {float(ew['interval_quality']):.3f}")
            except Exception:
                pass

        wq = winner.get("uncertainty_quality")
        lines.append(f"Winner uncertainty quality: {wq if wq is not None else '-'}")

        if len(ranked) > 1:
            rq = ranked[1].get("uncertainty_quality")
            rb = ranked[1].get("bundle_name", "")
            lines.append(f"Runner-up ({rb}) uncertainty quality: {rq if rq is not None else '-'}")

    return lines
