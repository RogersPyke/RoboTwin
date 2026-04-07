import os
import sys
import copy
import pathlib

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.robotworkspace import RobotWorkspace, create_dataloader

_POLICY_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))
from early_stop_util.early_stop import RelativeEarlyStopTracker


OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspaceEarlyStopPlugin(RobotWorkspace):
    include_keys = RobotWorkspace.include_keys + [
        "early_stop_val_loss_history",
        "early_stop_last_rel_improve",
        "early_stop_stop_reason",
    ]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.train_optimizer_steps = 0
        self.early_stop_best_val_loss = float("inf")
        self.early_stop_best_train_step = -1
        self.early_stop_no_improve_evals = 0
        self.early_stop_val_loss_history = []
        self.early_stop_last_rel_improve = float("nan")
        self.early_stop_stop_reason = "reached_training_end"

    def _sync_tracker_state(self, tracker: RelativeEarlyStopTracker) -> None:
        self.early_stop_best_val_loss = (
            float(tracker.best_val_loss)
            if tracker.best_step >= 0
            else float("inf")
        )
        self.early_stop_best_train_step = int(tracker.best_step)
        self.early_stop_no_improve_evals = int(tracker.no_improve_count)
        self.early_stop_val_loss_history = [float(v) for v in tracker.val_losses]
        self.early_stop_last_rel_improve = (
            float(tracker.last_rel_improve)
            if tracker.last_rel_improve is not None
            else float("nan")
        )

    def _write_training_summary(
        self,
        ckpt_rel_dir: str,
        stop_reason: str,
        last_val_loss: float,
        last_train_loss: float,
        tracker: RelativeEarlyStopTracker,
    ) -> None:
        summary_path = os.path.join(self.output_dir, ckpt_rel_dir, "steps.txt")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="ascii") as f:
            f.write(f"total_train_steps={int(self.train_optimizer_steps)}\n")
            f.write(f"last_val_loss={float(last_val_loss)}\n")
            f.write(f"last_train_loss={float(last_train_loss)}\n")
            f.write(f"stop_reason={str(stop_reason)}\n")
            f.write(f"best_eval_step={int(tracker.best_step)}\n")
            f.write(f"best_val_loss={float(tracker.best_val_loss) if tracker.best_step >= 0 else float('nan')}\n")
            f.write(f"total_evals_run={int(tracker.total_evals_run)}\n")
            f.write(f"no_improve_count={int(tracker.no_improve_count)}\n")

    def _val_mean_loss(self, cfg, dataset, val_dataloader, device):
        self.model.eval()
        try:
            losses = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    batch = dataset.postprocess(batch, device)
                    losses.append(self.model.compute_loss(batch))
                    if (
                        cfg.training.max_val_steps is not None
                        and batch_idx >= cfg.training.max_val_steps - 1
                    ):
                        break
            if not losses:
                return None
            return torch.mean(torch.tensor(losses)).item()
        finally:
            self.model.train()
            if cfg.training.freeze_encoder:
                self.model.obs_encoder.eval()

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        # ==== Early stop ====
        early_stop_patience_evals = int(getattr(cfg.training, "early_stop_patience_evals", 0))
        early_stop_rel_tol = float(getattr(cfg.training, "early_stop_rel_tol", 0.0))
        eval_steps_for_early_stop = int(getattr(cfg.training, "eval_steps_for_early_stop", 1))
        early_stop_tracker = RelativeEarlyStopTracker(
            patience_evals=early_stop_patience_evals,
            rel_tol=early_stop_rel_tol,
        )
        early_stop_enabled = early_stop_tracker.enabled
        if early_stop_enabled:
            eval_steps_for_early_stop = max(1, eval_steps_for_early_stop)
        # ==== Early stop ====

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        save_name = pathlib.Path(cfg.task.dataset.zarr_path).stem
        ckpt_rel_dir = f"checkpoints/{save_name}-{seed}"

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        self.early_stop_stop_reason = "max_epoch"
        last_train_loss = float("nan")
        last_val_loss = float("nan")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        val_fields = {}
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.train_optimizer_steps += 1
                            # ==== Early stop ====
                            if (
                                early_stop_enabled
                                and self.train_optimizer_steps % eval_steps_for_early_stop == 0
                            ):
                                val_loss_f = self._val_mean_loss(cfg, dataset, val_dataloader, device)
                                if val_loss_f is not None:
                                    last_val_loss = float(val_loss_f)
                                    decision = early_stop_tracker.on_eval(
                                        step=int(self.train_optimizer_steps),
                                        val_loss=float(val_loss_f),
                                    )
                                    self._sync_tracker_state(early_stop_tracker)
                                    self.early_stop_stop_reason = (
                                        str(decision.stop_reason)
                                        if decision.should_stop
                                        else "reached_training_end"
                                    )
                                    val_fields["val_loss"] = float(val_loss_f)
                                    val_fields["early_stop_best_val_loss"] = float(
                                        self.early_stop_best_val_loss
                                    )
                                    val_fields["early_stop_no_improve_evals"] = int(
                                        self.early_stop_no_improve_evals
                                    )
                                    val_fields["early_stop_stop_reason"] = str(
                                        self.early_stop_stop_reason
                                    )
                                    if decision.rel_improve is not None:
                                        val_fields["early_stop_rel_improve"] = float(
                                            decision.rel_improve
                                        )

                                    if decision.improved:
                                        # Design: best checkpoint saving is ONLY triggered by best-val refresh.
                                        if decision.rel_improve is None:
                                            print(
                                                f"Val loss @ step{self.train_optimizer_steps}: "
                                                f"{float(val_loss_f):.5f}",
                                                flush=True,
                                            )
                                        else:
                                            print(
                                                f"Val loss @ step{self.train_optimizer_steps}: "
                                                f"{float(val_loss_f):.5f} (improved)",
                                                flush=True,
                                            )
                                        self.save_checkpoint(
                                            path=f"{ckpt_rel_dir}/best_val.ckpt",
                                            use_thread=False,
                                        )
                                    else:
                                        print(
                                            f"Val loss @ step{self.train_optimizer_steps}: "
                                            f"{float(val_loss_f):.5f} (no improve "
                                            f"#{self.early_stop_no_improve_evals})",
                                            flush=True,
                                        )

                                    if decision.should_stop:
                                        # Design: early-stop only stops training and writes summary.
                                        # It MUST NOT trigger any best-ckpt saving.
                                        self.early_stop_stop_reason = str(decision.stop_reason)
                                        raw_loss_cpu = raw_loss.item()
                                        json_logger.log(
                                            {
                                                "train_loss": raw_loss_cpu,
                                                "global_step": self.global_step,
                                                "epoch": self.epoch,
                                                "lr": lr_scheduler.get_last_lr()[0],
                                                **val_fields,
                                                "early_stop": True,
                                                "early_stop_stop_reason": self.early_stop_stop_reason,
                                                "early_stop_best_train_step": int(
                                                    self.early_stop_best_train_step
                                                ),
                                                "early_stop_best_val_loss": float(
                                                    self.early_stop_best_val_loss
                                                ),
                                            }
                                        )
                                        print(
                                            f"[WARN] Early stop at opt_step={self.train_optimizer_steps}; "
                                            f"best: {ckpt_rel_dir}/best_val.ckpt",
                                            flush=True,
                                        )
                                        self._write_training_summary(
                                            ckpt_rel_dir=ckpt_rel_dir,
                                            stop_reason=self.early_stop_stop_reason,
                                            last_val_loss=last_val_loss,
                                            last_train_loss=last_train_loss,
                                            tracker=early_stop_tracker,
                                        )
                                        return
                            # ==== Early stop ====

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                            **val_fields,
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= (cfg.training.max_train_steps - 1)
                        ):
                            break

                train_loss = np.mean(train_losses)
                last_train_loss = float(train_loss)
                step_log["train_loss"] = train_loss

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if not early_stop_enabled:
                    should_run_val = (self.epoch % cfg.training.val_every) == 0
                    if should_run_val:
                        val_loss = self._val_mean_loss(cfg, dataset, val_dataloader, device)
                        if val_loss is not None:
                            last_val_loss = float(val_loss)
                            decision = early_stop_tracker.on_eval(
                                step=int(self.train_optimizer_steps),
                                val_loss=float(val_loss),
                            )
                            self._sync_tracker_state(early_stop_tracker)
                            self.early_stop_stop_reason = "reached_training_end"
                            step_log["val_loss"] = float(val_loss)
                            step_log["early_stop_best_val_loss"] = float(
                                self.early_stop_best_val_loss
                            )
                            step_log["early_stop_no_improve_evals"] = int(
                                self.early_stop_no_improve_evals
                            )
                            if decision.rel_improve is not None:
                                step_log["early_stop_rel_improve"] = float(decision.rel_improve)
                            print(f"Val loss:   {float(val_loss):.5f}", flush=True)
                            if decision.improved:
                                self.save_checkpoint(
                                    path=f"{ckpt_rel_dir}/best_val.ckpt",
                                    use_thread=False,
                                )

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]
                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint(f"{ckpt_rel_dir}/{self.epoch + 1}.ckpt")

                policy.train()
                step_log["early_stop_best_val_loss"] = float(self.early_stop_best_val_loss)
                step_log["early_stop_no_improve_evals"] = int(self.early_stop_no_improve_evals)
                step_log["early_stop_stop_reason"] = str(self.early_stop_stop_reason)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        self._write_training_summary(
            ckpt_rel_dir=ckpt_rel_dir,
            stop_reason=self.early_stop_stop_reason,
            last_val_loss=last_val_loss,
            last_train_loss=last_train_loss,
            tracker=early_stop_tracker,
        )

