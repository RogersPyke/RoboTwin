if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = [
        "global_step",
        "epoch",
        "train_optimizer_steps",
        "early_stop_best_val_loss",
        "early_stop_best_train_step",
        "early_stop_no_improve_evals",
    ]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.train_optimizer_steps = 0
        self.early_stop_best_val_loss = float("inf")
        self.early_stop_best_train_step = -1
        self.early_stop_no_improve_evals = 0

    def _val_mean_loss(self, cfg, dataset, val_dataloader, device):
        self.model.eval()
        try:
            losses = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    batch = dataset.postprocess(batch, device)
                    losses.append(self.model.compute_loss(batch))
                    if (cfg.training.max_val_steps is not None
                            and batch_idx >= cfg.training.max_val_steps - 1):
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
        early_stop_patience_evals = int(getattr(cfg.training, "early_stop_patience_evals", 0))
        early_stop_rel_tol = float(getattr(cfg.training, "early_stop_rel_tol", 0.0))
        eval_steps_for_early_stop = int(getattr(cfg.training, "eval_steps_for_early_stop", 1))
        early_stop_enabled = early_stop_patience_evals > 0 and early_stop_rel_tol > 0.0
        if early_stop_enabled:
            eval_steps_for_early_stop = max(1, eval_steps_for_early_stop)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        save_name = pathlib.Path(cfg.task.dataset.zarr_path).stem
        ckpt_rel_dir = f"checkpoints/{save_name}-{seed}"

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # ========= train for this epoch ==========
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
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.train_optimizer_steps += 1
                            if (early_stop_enabled
                                    and self.train_optimizer_steps % eval_steps_for_early_stop == 0):
                                val_loss_f = self._val_mean_loss(cfg, dataset, val_dataloader, device)
                                if val_loss_f is not None:
                                    val_fields["val_loss"] = float(val_loss_f)
                                    if np.isinf(self.early_stop_best_val_loss):
                                        self.early_stop_best_val_loss = float(val_loss_f)
                                        self.early_stop_best_train_step = int(self.train_optimizer_steps)
                                        self.early_stop_no_improve_evals = 0
                                        print(
                                            f"Val loss @ step{self.early_stop_best_train_step}: "
                                            f"{self.early_stop_best_val_loss:.5f}",
                                            flush=True,
                                        )
                                        self.save_checkpoint(
                                            path=f"{ckpt_rel_dir}/best_val.ckpt", use_thread=False)
                                    else:
                                        rel = (
                                            (self.early_stop_best_val_loss - float(val_loss_f))
                                            / max(abs(self.early_stop_best_val_loss), 1e-12))
                                        if rel > early_stop_rel_tol:
                                            self.early_stop_best_val_loss = float(val_loss_f)
                                            self.early_stop_best_train_step = int(
                                                self.train_optimizer_steps)
                                            self.early_stop_no_improve_evals = 0
                                            print(
                                                f"Val loss @ step{self.early_stop_best_train_step}: "
                                                f"{self.early_stop_best_val_loss:.5f} (improved)",
                                                flush=True,
                                            )
                                            self.save_checkpoint(
                                                path=f"{ckpt_rel_dir}/best_val.ckpt", use_thread=False)
                                        else:
                                            self.early_stop_no_improve_evals += 1
                                            print(
                                                f"Val loss @ step{self.train_optimizer_steps}: "
                                                f"{float(val_loss_f):.5f} (no improve "
                                                f"#{self.early_stop_no_improve_evals})",
                                                flush=True,
                                            )
                                            if (self.early_stop_no_improve_evals
                                                    >= early_stop_patience_evals):
                                                raw_loss_cpu = raw_loss.item()
                                                json_logger.log({
                                                    "train_loss": raw_loss_cpu,
                                                    "global_step": self.global_step,
                                                    "epoch": self.epoch,
                                                    "lr": lr_scheduler.get_last_lr()[0],
                                                    **val_fields,
                                                    "early_stop": True,
                                                    "early_stop_best_train_step": int(
                                                        self.early_stop_best_train_step),
                                                    "early_stop_best_val_loss": float(
                                                        self.early_stop_best_val_loss),
                                                })
                                                print(
                                                    f"[WARN] Early stop at opt_step="
                                                    f"{self.train_optimizer_steps}; "
                                                    f"best: {ckpt_rel_dir}/best_val.ckpt",
                                                    flush=True,
                                                )
                                                return

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

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch (epoch-end val only when early stop is off) ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if not early_stop_enabled:
                    should_run_val = (self.epoch % cfg.training.val_every) == 0
                    if should_run_val:
                        val_loss = self._val_mean_loss(cfg, dataset, val_dataloader, device)
                        if val_loss is not None:
                            step_log["val_loss"] = float(val_loss)
                            print(f"Val loss:   {float(val_loss):.5f}", flush=True)

                # run diffusion sampling on a training batch
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

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint(f"{ckpt_rel_dir}/{self.epoch + 1}.ckpt")

                # ========= eval end for this epoch ==========
                policy.train()

                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
