# TinyVLA 环境复原说明

本目录采用 lock-only 方式进行环境复原：

- 训练环境：`dexvla-robo`
- 推理环境：`robotwin-tinyvla-evla`

复原脚本：

- `rebuild_dexvla_robo.sh`
- `rebuild_robotwin_tinyvla_evla.sh`

复原依据：

- `dexvla-robo.conda-lock.yml`
- `robotwin-tinyvla-evla.conda-lock.yml`

使用方式：

1) 确认 lock 文件存在。  
2) 运行对应 `rebuild_*.sh`。  
3) 脚本仅执行 `conda-lock install` 重建环境，不再动态求解。
