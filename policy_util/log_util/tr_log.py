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


def format_relative_early_stop_val_line(
    policy_tag: str,
    step: int,
    val_loss: float,
    decision: EarlyStopRecord,
) -> str:
    """
    @input: [str, policy_tag ACT|DP|TinyVLA], [int, optimizer/global step], [float, val_loss],
             [EarlyStopRecord, post-on_eval snapshot]
    @output: [str, one ASCII line with fixed key=value tokens]
    @scenario: [Unified console line after RelativeEarlyStopTracker.on_eval]
    """
    status = _status_from_decision(decision)
    if decision.rel_improve is None:
        rel_s = "na"
    else:
        rel_s = f"{float(decision.rel_improve):.6g}"
    tag = str(policy_tag).strip() or "POLICY"
    return (
        f"[{tag}][VAL] "
        f"step={int(step)} "
        f"val_loss={float(val_loss):.5f} "
        f"status={status} "
        f"rel_improve={rel_s} "
        f"best_val_loss={float(decision.best_val_loss):.5f} "
        f"best_step={int(decision.best_step)} "
        f"no_improve={int(decision.no_improve_count)}"
    )


def print_relative_early_stop_val_line(
    policy_tag: str,
    step: int,
    val_loss: float,
    decision: EarlyStopRecord,
) -> None:
    """
    @input: Same as format_relative_early_stop_val_line
    @output: [None, prints one line to stdout]
    @scenario: [Convenience wrapper with flush]
    """
    print(
        format_relative_early_stop_val_line(policy_tag, step, val_loss, decision),
        flush=True,
    )


def format_epoch_val_line(policy_tag: str, step: int, val_loss: float) -> str:
    """
    @input: [str, policy_tag], [int, step], [float, val_loss]
    @output: [str, one line for per-epoch validation when early-stop cadence is off]
    @scenario: [DP non-early-stop path; optional alignment elsewhere]
    """
    tag = str(policy_tag).strip() or "POLICY"
    return (
        f"[{tag}][VAL] "
        f"step={int(step)} "
        f"val_loss={float(val_loss):.5f} "
        f"status=epoch_val"
    )


def print_epoch_val_line(policy_tag: str, step: int, val_loss: float) -> None:
    """Print format_epoch_val_line with flush."""
    print(format_epoch_val_line(policy_tag, step, val_loss), flush=True)
