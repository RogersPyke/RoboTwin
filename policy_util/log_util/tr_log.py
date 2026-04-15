"""
Single-line validation metrics for ACT / DP / TinyVLA (grep-friendly ASCII).

Dependencies:
- early_stop_util.EarlyStopRecord
"""

from __future__ import annotations

from early_stop_util import EarlyStopRecord


def _status_from_decision(decision: EarlyStopRecord) -> str:
    if decision.improved and decision.rel_improve is None:
        return "first"
    if decision.improved:
        return "improved"
    return "no_improve"


def _format_rel_improve(rel_improve: float | None) -> str:
    if rel_improve is None:
        return "N/A"
    pct = float(rel_improve) * 100
    if pct >= 0:
        return f"+{pct:.2f}%"
    return f"{pct:.2f}%"


def format_relative_early_stop_val_line(
    policy_tag: str,
    step: int,
    val_loss: float,
    decision: EarlyStopRecord,
) -> str:
    """
    @input: [str, policy_tag ACT|DP|TinyVLA], [int, optimizer/global step], [float, val_loss],
             [EarlyStopRecord, post-on_eval snapshot]
    @output: [str, multi-line ASCII log with [EARLY_STOP] prefix]
    @scenario: [Unified console line after RelativeEarlyStopTracker.on_eval]
    """
    status = _status_from_decision(decision)
    rel_str = _format_rel_improve(decision.rel_improve)
    tag = str(policy_tag).strip() or "POLICY"
    return (
        f"\n[EARLY_STOP] [{tag}]\n"
        f"  Step: {int(step)} | Val Loss: {float(val_loss):.5f}\n"
        f"  Status: {status} | Rel Improve: {rel_str}\n"
        f"  Best Val: {float(decision.best_val_loss):.5f} (Step {int(decision.best_step)})\n"
        f"  No Improve: {int(decision.no_improve_count)}"
    )


def print_relative_early_stop_val_line(
    policy_tag: str,
    step: int,
    val_loss: float,
    decision: EarlyStopRecord,
) -> None:
    """
    @input: Same as format_relative_early_stop_val_line
    @output: [None, prints multi-line log to stdout]
    @scenario: [Convenience wrapper with flush]
    """
    print(
        format_relative_early_stop_val_line(policy_tag, step, val_loss, decision),
        flush=True,
    )


def format_epoch_val_line(policy_tag: str, step: int, val_loss: float) -> str:
    """
    @input: [str, policy_tag], [int, step], [float, val_loss]
    @output: [str, multi-line log for per-epoch validation when early-stop cadence is off]
    @scenario: [DP non-early-stop path; optional alignment elsewhere]
    """
    tag = str(policy_tag).strip() or "POLICY"
    return (
        f"\n[EARLY_STOP] [{tag}]\n"
        f"  Step: {int(step)} | Val Loss: {float(val_loss):.5f}\n"
        f"  Status: epoch_val"
    )


def print_epoch_val_line(policy_tag: str, step: int, val_loss: float) -> None:
    """Print format_epoch_val_line with flush."""
    print(format_epoch_val_line(policy_tag, step, val_loss), flush=True)
