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

to eval ACT, run:
```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
# This command trains the policy using the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0
```

The task_config field refers to the evaluation environment configuration, while the ckpt_setting field refers to the training data configuration used during policy learning.

## Multi-task (joint dataset) training and evaluation

Only two modes are supported:

- **Single-task**: use positional args as above; no change.
- **Multi-task**: you must use `--config <name>`. Config files must be **YAML** under `_tr_cfg/` (train) and `_ev_cfg/` (eval). Wrappers: `_tr_wrapper.py` (train), `_ev_wrapper.py` (eval).

**Train (multi-task)**  
Config file: `_tr_cfg/<name>.yaml`

```
bash train.sh --config <name>
# Example: bash train.sh --config example
# (loads _tr_cfg/example.yaml)
```

Example `_tr_cfg/example.yaml`:

```yaml
TASK_TO_TRAIN:
  - task1
  - task2
TASK_CFG_TO_TRAIN:
  - taskCfg1
  - taskCfg2
TASK_NUM_TO_TRAIN:
  - taskNum1
  - taskNum2
TRAIN_SEED: 0
TRAIN_GPU_ID: 0
```

**Eval (multi-task)**  
Config file: `_ev_cfg/<name>.yaml`

```
bash eval.sh --config <name>
# Example: bash eval.sh --config example
# (loads _ev_cfg/example.yaml)
```

Example `_ev_cfg/example.yaml`: same task/cfg/num lists to identify the combined ckpt; plus `EVAL_TASK`, `EVAL_TASK_CFG`, `EVAL_SEED`, `EVAL_GPU_ID`.

Run `process_data.sh` for each subtask before multi-task training. Combined ckpt and dataset names use `__` (double underscore) internally.

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
bash train.sh --config example
```
- move_pillbottle_pad, demo_clean:
```
bash process_data.sh move_pillbottle_pad demo_clean 100
bash train.sh move_pillbottle_pad demo_clean 100 0 0
bash eval.sh move_pillbottle_pad demo_clean demo_clean 100 0 0
```
## Experiment 2:

- test if reverse task data can benefit to forward task.


## Experiment 3:

- test if reverse task data can benefit to forward task.