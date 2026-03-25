"""
TinyVLA early stopping callback.

Purpose:
- Provide TinyVLA early stop semantics aligned with ACT:
  * relative improvement threshold: (best - curr) / max(abs(best), eps) > rel_tol
  * patience counts consecutive evaluate() calls without improvement
  * best weights are saved as `output_dir/policy_best/` (covering previous best)

Dependencies:
- transformers TrainerCallback
- TinyVLA model save helpers (from `vla.model_load_utils` via `from vla import ...`)
"""

import math
import os
import shutil
from typing import Optional

import torch
from transformers import TrainerCallback

from vla import safe_save_model_for_hf_trainer
from vla import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3


class TinyVLAEarlyStopCallback(TrainerCallback):
    """
    @input: [output_dir, str], [early_stop_patience_evals, int], [early_stop_rel_tol, float]
    @output: [None], early stop controlled via `control.should_training_stop`
    @scenario: [Stop training when eval_loss stops improving sufficiently]
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

        # Public state for train_vla.py steps.txt.
        self.total_evals_run = 0
        self.stop_reason = "reached_training_end"
        self.best_eval_step = -1
        self.min_val_loss = float("nan")

        # Internal state.
        self._enabled = self.early_stop_patience_evals > 0 and self.early_stop_rel_tol > 0.0
        self._min_val_loss = float("inf")
        self._epochs_no_improve = 0

        self._trainer = None

    def set_trainer(self, trainer) -> None:
        """
        @input: [transformers.Trainer, trainer]
        @output: [None]
        @scenario: [Provide trainer reference for best checkpoint saving]
        """
        self._trainer = trainer

    def _should_save(self) -> bool:
        """
        @input: [None]
        @output: [bool, True if this process should write `policy_best/`]
        @scenario: [Avoid distributed checkpoint save races]
        """
        return self.local_rank in (0, -1)

    def _relative_improvement(self, curr: float, best: float) -> float:
        """
        @input: [float, curr], [float, best]
        @output: [float, relative improvement value]
        @scenario: [Compute ACT-aligned (best-curr)/abs(best) improvement]
        """
        # Match ACT: rel_improve = (best - curr) / max(abs(best), eps)
        eps = 1e-12
        denom = max(abs(best), eps)
        return (best - curr) / denom

    def _save_best(self) -> None:
        """
        @input: [None], uses internal trainer/model
        @output: [None]
        @scenario: [Cover-save `policy_best/` when a new best is found]
        """
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
            # Match TinyVLA train_bc LoRA saving semantics, but to best_dir.
            model.config.save_pretrained(self.best_dir)
            model.save_pretrained(self.best_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(self.best_dir, "non_lora_trainables.bin"))
        else:
            safe_save_model_for_hf_trainer(trainer=self._trainer, output_dir=self.best_dir)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        @input: [HF eval callback inputs]
        @output: [None], modifies `control` and internal best state
        @scenario: [Update best and early stop decision based on eval_loss]
        """
        if metrics is None:
            return
        if "eval_loss" not in metrics:
            return

        self.total_evals_run += 1
        if not self._enabled:
            return

        eval_loss = metrics["eval_loss"]
        curr_val_loss = float(eval_loss)

        # Baseline: first evaluate sets best.
        if math.isinf(self._min_val_loss):
            self._min_val_loss = curr_val_loss
            self._epochs_no_improve = 0
            self.best_eval_step = int(state.global_step)
            self.min_val_loss = float(curr_val_loss)
            self.stop_reason = "reached_training_end"
            self._save_best()
            return

        rel_improve = self._relative_improvement(curr=curr_val_loss, best=self._min_val_loss)
        improved = rel_improve > self.early_stop_rel_tol

        if improved:
            self._min_val_loss = curr_val_loss
            self._epochs_no_improve = 0
            self.best_eval_step = int(state.global_step)
            self.min_val_loss = float(curr_val_loss)
            self.stop_reason = "reached_training_end"
            self._save_best()
            return

        self._epochs_no_improve += 1
        if self._epochs_no_improve >= self.early_stop_patience_evals:
            self.stop_reason = (
                f"early_stop(patience={self.early_stop_patience_evals}, rel_tol={self.early_stop_rel_tol})"
            )
            control.should_training_stop = True

