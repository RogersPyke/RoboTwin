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
