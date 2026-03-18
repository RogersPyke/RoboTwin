conda env: robotwin-act
---
to process data for training, run:
```
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh beat_block_hammer demo_clean 50
# bash process_data.sh stack_blocks_three demo_clean 100
in path .../ACT/
---
to train, run:
```
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
# bash train.sh beat_block_hammer demo_clean 50 0 0
```
By default, the model is trained for 6,000 steps.
---
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
---
The task_config field refers to the evaluation environment configuration, while the ckpt_setting field refers to the training data configuration used during policy learning.

---

Experiment 1: test if reverse task data can transfer to forward task.