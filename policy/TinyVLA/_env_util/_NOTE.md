# TinyVLA 环境复原说明

本目录采用 lock-only 方式进行环境复原：

- 训练环境：`dexvla-robo`
- 推理环境：`robotwin-tinyvla-evla`

## 官方构建流程

训练环境：
```
cd policy/TinyVLA
conda env create -f Train_Tiny_DexVLA_train.yml
conda activate dexvla-robo
cd policy_heads
pip install -e .
```

推理环境：

```
conda activate your_RoboTwin_env
pip install -r Eval_Tiny_DexVLA_requirements.txt 
```

为保证隔离建议使用复制而不是直接在 robotwin 环境安装。
lock 构建失败时，可以尝试上述方法。但需要注意的是原始项目里的 txt 和 yaml 均不可用，因为其中包含了大量本地路径和不可解析的路径，以及被修改过的本地版本代码(+ dirty)，因此最推荐的方法还是直接使用复原脚本。

## lock 构建

复原脚本：

- `rebuild_dexvla_robo.sh`
- `rebuild_robotwin_tinyvla_evla.sh`

复原依据：

- `dexvla-robo.conda-lock.yml`
- `robotwin-tinyvla-evla.conda-lock.yml`

使用方式：

1) 确认 lock 文件存在。  
2) 运行对应 `rebuild_*.sh`。  
3) 脚本先执行 `conda-lock install` 重建环境，不再动态求解。
4) 环境重建后统一补装 `curobo`：
   - 源码目录：`RoboTwin/envs/curobo`
   - 安装方式：`pip install -e <repo>/envs/curobo --no-build-isolation`

## 近期 dryrun 调试记录（robotwin-dp）

为方便后续排查 DP/TinyVLA 共享依赖问题，记录一次 `robotwin-dp` 的训练/推理 dryrun 结果：

1) 推理阶段出现动态库冲突（非 OOM）：
   - 报错：`ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found`
   - 触发链：`eval_policy.py -> test_render.py -> toppra -> matplotlib`
   - 原因：运行时优先加载了系统 `libstdc++.so.6`，其 ABI 版本低于当前 Python 二进制扩展所需版本。
   - 处理：在启动前注入 conda lib 路径，优先使用环境内 `libstdc++`：
     - `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`

2) `task_config` 缺失导致推理提前失败（非 OOM）：
   - 缺失文件示例：`task_config/demo_clean.yml`、`task_config/_embodiment_config.yml`、`task_config/_camera_config.yml`
   - 处理：补齐对应配置文件后重试。

3) ckpt 路径/命名不匹配导致推理找不到模型（非 OOM）：
   - 表现：wrapper 按配置拼接的 ckpt 路径不存在。
   - 处理：确认 `_tr_cfg` 与 `_ev_cfg` 中 `expert_data_num`、任务拼接 slug、checkpoint 编号一致。

4) 最终状态：
   - 训练和推理均能推进到模型装载阶段，最终错误收敛为 `torch.OutOfMemoryError`（CUDA OOM）。
   - 若目标是“迭代到只剩 OOM”，则以上 1)-3) 需先清空。
