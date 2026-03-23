# ACT

conda env: robotwin-act
All should be done in ACT/ folder.

ACT is always trained on scratch.

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

By default, the model is trained for 6,000 steps.

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

  Do **not** expect a joint model under `act_ckpt/act-<single_task>/...`; evaluating a joint policy **must** use `bash eval.sh --config <name>` so `_ev_wrapper.py` resolves that path from `TRAIN_TASKS` in `_ev_cfg/<name>.yaml`.

**Train (multi-task)**  
Config: `_tr_cfg/<name>.yaml` (requires `TRAIN_TASKS`, `TRAIN_SEED`, `TRAIN_GPU_ID`).

```
bash train.sh --config <name>
# Example: bash train.sh --config hanging_mug_pair
```

Example `_tr_cfg/hanging_mug_pair.yaml`:

```yaml
TRAIN_TASKS:
  - [hanging_mug, demo_clean, 100]
  - [unhanging_mug, demo_clean, 100]
TRAIN_SEED: 0
TRAIN_GPU_ID: 0
```

**Eval (multi-task / joint checkpoint)**  
Config: `_ev_cfg/<name>.yaml` (`TRAIN_TASKS` must match the joint training run; `EVAL_TASKS` lists sim settings to evaluate; plus `EVAL_SEED`, `EVAL_GPU_ID`).

```
bash eval.sh --config <name>
# Example: bash eval.sh --config hanging_mug_pair
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
bash train.sh --config hanging_mug_pair
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
bash train.sh --config hanging_mug_pair
bash eval.sh --config hanging_mug_pair
```

## Experiment 3:

- test if reverse task data can benefit to forward task.
