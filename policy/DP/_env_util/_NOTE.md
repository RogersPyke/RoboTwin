# DP 环境

conda env: robotwin-dp

## 官方安装流程

Under robotwin run:
```
cd policy/DP
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor sympy
pip install -e .
```

为进行环境隔离建议复制一个环境然后进行安装，同ACT。当一键复原失败时可以尝试进行手动构建。

## lock 复原

本目录采用 lock-only 方式进行环境复原：

- 训练推理环境：`robotwin-dp`

复原脚本：

- `rebuild_robotwin_dp.sh`

复原依据：

- `robotwin-dp.conda-lock.yml`

使用方式：

1) 确认 lock 文件存在。  
2) 运行 `rebuild_robotwin_dp.sh`。  
3) 脚本先执行 `conda-lock install` 重建环境，不再动态求解。  
4) 环境重建后执行 `pip install -e policy/DP` 补装可编辑包。

## 多任务 DryRun（训练+推理）执行顺序

仅测试多任务配置，推荐使用：

- 训练配置：`policy/DP/_tr_cfg/hanging_mug_pair_dryrun.yaml`
- 推理配置：`policy/DP/_ev_cfg/hanging_mug_pair_dryrun.yaml`

在 `third_party/RoboTwin` 目录执行：

```bash
conda run -n robotwin-dp bash -lc 'cd policy/DP && bash _train.sh hanging_mug_pair_dryrun'
conda run -n robotwin-dp bash -lc 'cd policy/DP && bash _eval.sh hanging_mug_pair_dryrun'
```

## DryRun 迭代方法（直到只剩 OOM）

1) 固定只跑上述多任务 dryrun。  
2) 先跑训练，再跑推理，记录首个非 OOM 报错。  
3) 每次仅修复一个非 OOM 根因后重跑，避免一次改动过多。  
4) 重复步骤 2)-3)，直到训练/推理都只剩 `torch.OutOfMemoryError`。  
5) 若推理使用的 ckpt 尚未由当前训练产出，可在 `hanging_mug_pair_dryrun.yaml` 中设置：
   - `CHECKPOINT_EXPERT_DATA_NUM`：指向已有 checkpoint 的 episodes 数；
   - `CHECKPOINT_NUM`：指定 checkpoint 编号（例如 `1`）。

## 本轮已确认的非 OOM 问题与处理

- `CXXABI_1.3.15 not found`（`matplotlib`/`libstdc++` 运行时冲突）：
  - 在 `policy/DP/_ev_wrapper.py` 中前置 `LD_LIBRARY_PATH=<CONDA_PREFIX>/lib` 后解决。
- 推理 ckpt 路径不存在（训练因 OOM 未产出对应 checkpoint）：
  - 在 `_ev_wrapper.py` 增加 `CHECKPOINT_EXPERT_DATA_NUM` 支持，并在 dryrun eval 配置中设置该字段以复用已有多任务 ckpt。
