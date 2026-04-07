# Tiny-VLA (Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation)
## Install
To guarantee clean isolation between training and evaluation environments for both DexVLA and TinyVLA, we provide two distinct, self-contained setups.The training and testing environment can be used for both DexVLA and TinyVLA.

Training Environment： dexvla-robo
```bash
cd policy/TinyVLA
conda env create -f Train_Tiny_DexVLA_train.yml
conda activate dexvla-robo
cd policy_heads
pip install -e .
```
Evaluation Environment: robotwin-tinyvla-eval

Follow the RoboTwin 2.0 documentation to set up the RoboTwin environment. Once the environment is activated, run the following commands to install the required packages:
```bash
pip install einops==0.8.1
pip install transformers==4.47.0
pip install timm==1.0.16
pip install diffusers==0.34.0
pip install qwen-vl-utils==0.0.11
pip install accelerate==0.26.0
```
If you encounter the following error:
```bash
Unrecognized option 'crf'. 
Error splitting the argument list: Option not found.
```
Run the following command to install the required version of ffmpeg:
```bash
conda install -c conda-forge ffmpeg
```
## Prepare Training Data
This step performs data preprocessing, converting the original RoboTwin 2.0 data into the format required for TinyVLA training. The `expert_data_num` parameter specifies the number of trajectory pairs to be used as training data.
```bash
python process_data.py ${task_name} ${task_config} ${expert_data_num}
# python process_data.py beat_block_hammer demo_randomized 50
```
If success, you will find the `sim_${task_name}/${setting}_${expert_data_num}` folder under `policy/Tinyvla/data`.

## Train Policy
This step launches the training process.
First, download the VLM model InternVL3-1B ([huggingface](https://huggingface.co/OpenGVLab/InternVL3-1B/tree/main)) to the path `.../policy/TinyVLA/model_param/InternVL3-1B`. Then modify the `config.json` file in the folder as follows:
```
{
    "_name_or_path": ".../robotiwin/policy/TinyVLA/vla/models/internvl", # Modify this.
    "architectures": [
        "TinyVLA" # Change this.
    ],
    # "auto_map":{...} # Delete this.
    ...
    "llm_config": {}, # Don't Change.
    "min_dynamic_patch": 1,
    "model_type": "tinyvla", # Change this.
    ...
}
```
Then add a base-task config item in `.../policy/TinyVLA/aloha_scripts/constants.py`
```python
TASK_CONFIGS = {
    ...
    "your_task": {
        'dataset_dir': [DATA_DIR + "/sim-your_task/aloha-agilex-1-m1_b1_l1_h0.03_c0_D435-100"],
        'episode_len': 500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    }
}
```
Only the base task must be registered here. For joint training, the wrapper will synthesize the combined task spec automatically, so you do not need to add a joint `task_name` entry such as `task_a__task_b`.

Then begin the training
```bash
bash ./scripts/franks/train_robotwin_aloha.sh
```
### Multi-run (YAML configs, wrappers)
TinyVLA supports ACT-aligned YAML wrappers for joint training and joint-checkpoint evaluation:
- Train: `bash _train.sh <cfg_name>` (config: `_tr_cfg/<cfg_name>.yaml`)
- Eval:  `bash _eval.sh <cfg_name>` (config: `_ev_cfg/<cfg_name>.yaml`)

Notes:
- `_train.sh/_eval.sh` take `<cfg_name>` directly. Do not add an extra top-level `--config`.
- Joint training output directories are auto-derived as:
  `policy/TinyVLA/tinyvla_ckpt/tinyvla-<task1>__<task2>/<cfg1>__<cfg2>-<sum_expert_num>/`
- Use `_eval.sh` for joint checkpoints. The wrapper resolves the shared checkpoint from `TRAIN_TASKS` and expands `EVAL_TASKS` into one eval run per row.

Recommended joint-train YAML shape:
```yaml
TRAIN_TASKS:
  - [task_name_a, demo_clean, 100]
  - [task_name_b, demo_clean, 100]
TRAIN_SEED: 0
TRAIN_GPU_ID: "0"
EARLY_STOP_PATIENCE_EVALS: 3
EARLY_STOP_REL_TOL: 0.01
EVAL_STEPS_FOR_EARLY_STOP: 100
VLA_TRAIN_ARGS:
  model_name_or_path: /path/to/InternVL3-1B
  max_steps: 5000
  per_device_train_batch_size: 64
  # output_dir is wrapper-owned in joint mode
```

Recommended joint-eval YAML shape:
```yaml
TRAIN_TASKS:
  - [task_name_a, demo_clean, 100]
  - [task_name_b, demo_clean, 100]
EVAL_TASKS:
  - [task_name_a, demo_clean, 100]
  - [task_name_b, demo_clean, 100]
EVAL_SEED: 0
EVAL_GPU_ID: "0"
MODEL_BASE: /path/to/InternVL3-1B
USE_POLICY_BEST: true
```

Configure the training by modifying the following items in the `train_robotwin_aloha.sh` file.
```
TASK=your_task # Set the Task
ROOT=.../robotiwin/policy/TinyVLA # Set Root Path
mnop=.../robotiwin/policy/TinyVLA/model_param/InternVL3-1B/ # Set The Path of base VLM
```

### Early stopping (TinyVLA)
TinyVLA supports an ACT-aligned early stopping mechanism driven by `eval_loss` during HF `Trainer` evaluation. Implementation: `vla/train/relative_early_stop.py` (`TrainerCallback`), reusing `policy/rw_common/early_stop.py` (`RelativeEarlyStopTracker`) like ACT/DP.

It is now recommended to configure early stopping in the wrapper YAML, not inside `VLA_TRAIN_ARGS`:
```yaml
EARLY_STOP_PATIENCE_EVALS: 3
EARLY_STOP_REL_TOL: 0.01
EVAL_STEPS_FOR_EARLY_STOP: 100
```

By default it is disabled when `EARLY_STOP_PATIENCE_EVALS <= 0` or `EARLY_STOP_REL_TOL <= 0.0`.

When enabled, evaluations run every `eval_steps_for_early_stop` steps, and the best checkpoint is saved by coverage into:
`$OUTPUT/policy_best/`

## Eval Policy
For direct `eval.sh` usage, you still need to modify the corresponding path in the `deploy_policy.yml` file:
1. **model_path** : Path to the trained model, in the OUTPUT path.
2. **state_path** : Path to `dataset_stats.pkl`, in the OUTPUT path.
3. **model_base** : Path to InternVL3-1B.

Then execute:
```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_randomized 0 50 0 0
```

For joint checkpoints produced by `_train.sh`, prefer:
```bash
bash _eval.sh <cfg_name>
```
The wrapper will:
- infer the joint checkpoint directory from `TRAIN_TASKS`
- set `model_path` / `state_path` automatically
- run one evaluation per `EVAL_TASKS` row against the same joint checkpoint

## Citation

If you find Tiny-VLA useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{wen2024tinyvla,
    title={Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation},
    author={Wen, Junjie and Zhu, Yichen and Li, Jinming and Zhu, Minjie and Wu, Kun and Xu, Zhiyuan and Liu, Ning and Cheng, Ran and Shen, Chaomin and Peng, Yaxin and others},
    booktitle={IEEE Robotics and Automation Letters (RA-L)},
    year={2025}
}
```