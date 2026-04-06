"""
TinyVLA relative early-stop plugin (DP-style separate module).

Purpose:
- Subclass HuggingFace TrainerCallback and compose early_stop_util.RelativeEarlyStopTracker
  (same semantics as ACT and DP RobotWorkspaceEarlyStopPlugin).
- Best weights saved under output_dir/policy_best/ when validation improves.

Dependencies:
- transformers TrainerCallback
- early_stop_util.RelativeEarlyStopTracker (RoboTwin policy root on sys.path)
- vla model save helpers (safe_save_model_for_hf_trainer, PEFT helpers)
"""

import os
import pathlib
import shutil
import sys
from typing import Optional

import torch
from transformers import TrainerCallback

_POLICY_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))
from early_stop_util import RelativeEarlyStopTracker

from vla import safe_save_model_for_hf_trainer
from vla import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3


class RelativeEarlyStop(TrainerCallback):
    """
    @input: [output_dir, str], [early_stop_patience_evals, int], [early_stop_rel_tol, float]
    @output: [None], early stop via control.should_training_stop
    @scenario: [Val-loss early stopping aligned with ACT/DP RelativeEarlyStopTracker]
    """

    def __init__(
        self,
        output_dir: str,
        early_stop_patience_evals: int,
        early_stop_rel_tol: float,
        eval_steps_for_early_stop: int = 100,
        local_rank: int = -1,
        lora_enable: bool = False,
        lora_bias: str = "none",
    ):
        self.output_dir = output_dir
        self.best_dir = os.path.join(output_dir, "policy_best")

        self.early_stop_patience_evals = int(early_stop_patience_evals)
        self.early_stop_rel_tol = float(early_stop_rel_tol)
        self.eval_steps_for_early_stop = int(eval_steps_for_early_stop)
        self.local_rank = int(local_rank)
        self.lora_enable = bool(lora_enable)
        self.lora_bias = str(lora_bias)

        self._tracker = RelativeEarlyStopTracker(
            patience_evals=self.early_stop_patience_evals,
            rel_tol=self.early_stop_rel_tol,
        )

        # Public state for train_vla.py steps.txt.
        self.total_evals_run = 0
        self.stop_reason = "reached_training_end"
        self.best_eval_step = -1
        self.min_val_loss = float("nan")

        self._trainer = None
        self._warned_disabled = False

    def set_trainer(self, trainer) -> None:
        """Attach trainer for best checkpoint IO."""
        self._trainer = trainer

    def _should_save(self) -> bool:
        return self.local_rank in (0, -1)

    def _save_best(self) -> None:
        if self._trainer is None:
            return
        if not self._should_save():
            return

        if os.path.isdir(self.best_dir):
            shutil.rmtree(self.best_dir, ignore_errors=True)
        os.makedirs(self.best_dir, exist_ok=True)

        if self.lora_enable:
            model = self._trainer.model
            state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), self.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters(), require_grad_only=False
            )
            model.config.save_pretrained(self.best_dir)
            model.save_pretrained(self.best_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(self.best_dir, "non_lora_trainables.bin"))
        else:
            safe_save_model_for_hf_trainer(trainer=self._trainer, output_dir=self.best_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if self._warned_disabled:
            return
        if not self._tracker.enabled:
            print(
                "[WARN] Early stopping is disabled (patience_evals=0 and/or rel_tol<=0.0). "
                "Training will run for the full schedule (e.g. max_steps).",
                flush=True,
            )
            self._warned_disabled = True
        else:
            es = max(1, int(self.eval_steps_for_early_stop))
            print(
                f"Early stopping enabled: patience_evals={self.early_stop_patience_evals}, "
                f"rel_tol={self.early_stop_rel_tol}, eval_steps_for_early_stop={es}",
                flush=True,
            )
            self._warned_disabled = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if "eval_loss" not in metrics:
            return

        curr_val_loss = float(metrics["eval_loss"])

        if not self._tracker.enabled:
            self.total_evals_run += 1
            return

        decision = self._tracker.on_eval(
            step=int(state.global_step),
            val_loss=curr_val_loss,
        )
        self.total_evals_run = int(self._tracker.total_evals_run)

        if decision.improved:
            self.best_eval_step = int(decision.best_step)
            self.min_val_loss = float(decision.best_val_loss)
            self.stop_reason = "reached_training_end"
            if decision.rel_improve is None:
                print(
                    f"Val loss @ step{state.global_step}: {curr_val_loss:.5f}",
                    flush=True,
                )
            else:
                print(
                    f"Val loss @ step{state.global_step}: {curr_val_loss:.5f} (improved)",
                    flush=True,
                )
            self._save_best()
        else:
            print(
                f"Val loss @ step{state.global_step}: {curr_val_loss:.5f} "
                f"(no improve #{decision.no_improve_count})",
                flush=True,
            )

        if decision.should_stop:
            self.stop_reason = str(decision.stop_reason)
            control.should_training_stop = True
