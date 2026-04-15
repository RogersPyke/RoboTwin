"""
Shared relative-improvement early-stop tracker.

Purpose:
- Provide a small, reusable state machine for val-loss based early stopping.
- Keep val loss history and best(min) val loss in one place.

Dependencies:
- Python standard library only.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EarlyStopRecord:
    """
    @input: [int, step >= 0], [float, val_loss finite]
    @output: [dataclass, early-stop decision snapshot]
    @scenario: [Return improvement/stop decision for one evaluation]
    """

    step: int
    val_loss: float
    rel_improve: Optional[float]
    improved: bool
    no_improve_count: int
    should_stop: bool
    stop_reason: str
    best_val_loss: float
    best_step: int


class RelativeEarlyStopTracker:
    """
    @input: [int, patience_evals >= 0], [float, rel_tol >= 0.0], [float, eps > 0]
    @output: [stateful tracker object]
    @scenario: [Track eval-loss history, best val, and patience-based early stop]
    """

    def __init__(self, patience_evals: int, rel_tol: float, eps: float = 1e-12):
        self.patience_evals = int(patience_evals)
        self.rel_tol = float(rel_tol)
        self.eps = float(eps)

        self.enabled = self.patience_evals > 0 and self.rel_tol > 0.0
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.best_step = -1
        self.no_improve_count = 0
        self.total_evals_run = 0
        self.last_rel_improve: Optional[float] = None

    def _relative_improvement(self, curr: float, best: float) -> float:
        denom = max(abs(best), self.eps)
        return (best - curr) / denom

    def on_eval(self, step: int, val_loss: float) -> EarlyStopRecord:
        """
        @input: [int, step], [float, val_loss]
        @output: [EarlyStopRecord, decision for current eval]
        @scenario: [Update best/min val and patience counter]
        """
        step_i = int(step)
        loss_f = float(val_loss)
        self.total_evals_run += 1
        self.val_losses.append(loss_f)

        stop_reason = "max_step_reached"
        if self.best_step < 0:
            self.best_val_loss = loss_f
            self.best_step = step_i
            self.no_improve_count = 0
            self.last_rel_improve = None
            return EarlyStopRecord(
                step=step_i,
                val_loss=loss_f,
                rel_improve=None,
                improved=True,
                no_improve_count=self.no_improve_count,
                should_stop=False,
                stop_reason=stop_reason,
                best_val_loss=self.best_val_loss,
                best_step=self.best_step,
            )

        rel_improve = self._relative_improvement(curr=loss_f, best=self.best_val_loss)
        self.last_rel_improve = rel_improve
        improved = rel_improve >= self.rel_tol
        if improved:
            self.best_val_loss = loss_f
            self.best_step = step_i
            self.no_improve_count = 0
            return EarlyStopRecord(
                step=step_i,
                val_loss=loss_f,
                rel_improve=rel_improve,
                improved=True,
                no_improve_count=self.no_improve_count,
                should_stop=False,
                stop_reason=stop_reason,
                best_val_loss=self.best_val_loss,
                best_step=self.best_step,
            )

        self.no_improve_count += 1
        should_stop = self.enabled and self.no_improve_count >= self.patience_evals
        if should_stop:
            stop_reason = (
                f"early_stop(patience={self.patience_evals}, rel_tol={self.rel_tol})"
            )
        return EarlyStopRecord(
            step=step_i,
            val_loss=loss_f,
            rel_improve=rel_improve,
            improved=False,
            no_improve_count=self.no_improve_count,
            should_stop=should_stop,
            stop_reason=stop_reason,
            best_val_loss=self.best_val_loss,
            best_step=self.best_step,
        )

    def export_state(self) -> Dict[str, object]:
        """
        @input: [None]
        @output: [dict, serializable tracker state]
        @scenario: [Persist tracker status to metadata/checkpoint fields]
        """
        return {
            "enabled": bool(self.enabled),
            "patience_evals": int(self.patience_evals),
            "rel_tol": float(self.rel_tol),
            "total_evals_run": int(self.total_evals_run),
            "best_val_loss": float(self.best_val_loss)
            if self.best_step >= 0
            else float("nan"),
            "best_step": int(self.best_step),
            "no_improve_count": int(self.no_improve_count),
            "last_rel_improve": (
                float(self.last_rel_improve)
                if self.last_rel_improve is not None
                else None
            ),
            "val_losses": [float(v) for v in self.val_losses],
        }
