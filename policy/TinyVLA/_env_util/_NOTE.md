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
