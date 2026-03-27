# ACT

## 训练早停（防止过拟合，推荐开启）

当前 ACT 训练默认会跑固定的 `num_epochs`。为防止过拟合，我们新增了 **Step-level 的 early stopping**，在训练过程中以固定间隔对 **validation loss** 做评估，并使用“相对提升”判定是否有进步：

- **相对提升**：当新的 `val_loss` 相比历史最优 `best_val_loss` 的相对下降幅度超过 `rel_tol`，才算“有提升”并重置耐心计数。
  - 直观形式：`(best_val_loss - val_loss) / max(abs(best_val_loss), eps) > rel_tol`
- **触发停止**：当连续 `patience_evals` 次 `eval_loss` 评估都未达到上述相对提升（即“改进幅度不超过 `rel_tol`”），则提前 `break` 结束训练，并照常保存 best/last checkpoint。

评估顺序：与先验证再训练不同，本实现会在执行完一段训练后（每隔 `eval_steps_for_early_stop=K` 步）再进行验证，使“验证对象”始终是“已训练之后的模型”。

### 默认行为（重要）

- 默认 `early_stop_patience_evals=0` 且 `rel_tol=0.0`，表示 **不启用早停**。
- 当早停未启用时，训练会继续使用原本的固定训练步数/epoch，并打印警告：
  - `[WARN] Early stopping is disabled ... Training will run for the full num_epochs.`

### 训练步数记录（steps.txt）

训练结束后（无论是否早停），会在 checkpoint 目录写入一个 `steps.txt`，记录：

- `total_train_steps`：实际执行的 optimizer step 总数（遍历 train_dataloader 的累计 batch 数）
- `total_evals_run`：实际运行的 eval 次数（驱动 best/early stop 的评估次数）
- `stop_reason`：停止原因（`reached_training_end` 或 `early_stop(patience=..., rel_tol=...)`）
- `best_eval_step / min_val_loss`：best checkpoint 对应的评估 step 与指标
- `early_stop_patience_evals`、`early_stop_rel_tol`：早停的关键参数取值

## Basic commands

to process data for training, run:
```
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh beat_block_hammer demo_clean 50
# bash process_data.sh stack_blocks_three demo_clean 100
```

to train, run:
```
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
# bash train.sh beat_block_hammer demo_clean 50 0 0
```

Seed controls training randomness: model initialization RNG (torch/numpy), dataloader shuffle, and random start-timestep sampling. It also appears in checkpoint/plot filenames. It does not affect process_data dataset generation.

By default, the model is trained for 6,000 epochs (historical setting). If early stopping is disabled, it will run the full num_epochs.

to eval ACT (single-task checkpoint), run:
```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
# ckpt_setting is the training data configuration; task_config is the evaluation environment.
#
# To evaluate a policy trained on demo_clean and tested on demo_randomized:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0
```

The task_config field refers to the evaluation environment configuration, while the ckpt_setting field refers to the training data configuration used during policy learning.

## Multi-task (joint dataset) training and evaluation

Two modes:

- **Single-task**: positional args for `process_data.sh`, `train.sh`, and `eval.sh` as above. Checkpoints live under `act_ckpt/act-<task_name>/<task_config>-<expert_data_num>/`.

- **Joint multi-task**: train and eval only through YAML configs. Combined names use `__` (double underscore) between tasks or configs. There is **one** checkpoint directory per joint run:

  `act_ckpt/act-<task1>__<task2>/.../<cfg1>__<cfg2>-<total_episodes>/`

  Do **not** expect a joint model under `act_ckpt/act-<single_task>/...`; evaluating a joint policy must use `bash _eval.sh <name>` so `_ev_wrapper.py` resolves that path from `TRAIN_TASKS` in `_ev_cfg/<name>.yaml`.

**Train (multi-task) [Recommended]**  
Config: `_tr_cfg/<name>.yaml`

Recommended entrypoint:

```
bash _train.sh <name>
# Example: bash _train.sh hanging_mug_pair
```

Note: `bash _train.sh/_eval.sh` 直接接收 `<name>` 位置参数；不需要向最外层再传 `--config`。

The wrapper will copy the YAML into the checkpoint folder (so every ckpt has a config snapshot), and `steps.txt` will record the actual training steps/epochs.

Example `_tr_cfg/hanging_mug_pair.yaml`:

```yaml
TRAIN_TASKS:
  - [hanging_mug, demo_clean, 100]
  - [unhanging_mug, demo_clean, 100]
TRAIN_SEED: 0
TRAIN_GPU_ID: 0

# Optional (recommended) early stopping settings (set null to disable and use defaults):
EARLY_STOP_PATIENCE_EVALS: 30
EARLY_STOP_REL_TOL: 0.01
EVAL_STEPS_FOR_EARLY_STOP: 100
```

**Eval (multi-task / joint checkpoint) [Recommended]**  
Config: `_ev_cfg/<name>.yaml` (`TRAIN_TASKS` must match the joint training run; `EVAL_TASKS` lists sim settings to evaluate; plus `EVAL_SEED`, `EVAL_GPU_ID`).

Recommended entrypoint:

```
bash _eval.sh <name>
# Example: bash _eval.sh hanging_mug_pair
```

Run `process_data.sh` for each subtask before multi-task training.

## Experiment 1:

test if fake forward task data can benefit to forward task.

Run on 4 tasks:

```
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}

bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}

bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
```

- hanging_mug, demo_clean:
```
bash process_data.sh hanging_mug demo_clean 100
bash train.sh hanging_mug demo_clean 100 0 0
bash eval.sh hanging_mug demo_clean demo_clean 100 0 0
bash _train.sh hanging_mug_pair
```
- move_pillbottle_pad, demo_clean:
```
bash process_data.sh move_pillbottle_pad demo_clean 100
bash train.sh move_pillbottle_pad demo_clean 100 0 0
bash eval.sh move_pillbottle_pad demo_clean demo_clean 100 0 0
```
## **Experiment 2**:

- test if reverse task data can benefit to forward task.

- hanging_mug, demo_clean:
- Baseline train on hanging_mug, 100.
- Reverse train on unhanging_mug, 100.
- Diff train on hanging_mug, 100; unhanging_mug, 100
- Eval on task: hanging_mug.
```
# === Process data ===
bash process_data.sh hanging_mug demo_clean 100
bash process_data.sh unhanging_mug demo_clean 100

# === Train and eval baseline ===
bash train.sh hanging_mug demo_clean 100 0 0
bash eval.sh hanging_mug demo_clean demo_clean 100 0 0

# Joint train / joint eval (example)
bash _train.sh hanging_mug_pair
bash _eval.sh hanging_mug_pair
```

## Experiment 3:

- test if reverse task data can benefit to forward task.
