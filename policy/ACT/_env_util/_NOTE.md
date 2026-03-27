# 环境

conda env: robotwin-act

## 官方安装流程

在环境 robotwin 下执行：
```
cd policy/ACT

pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython

cd detr && pip install -e . && cd ..
```
为了保证更好的环境隔离性，建议的方法是复制一个 robotwin 环境编辑为 robotwin-act，然后执行以上命令，当直接使用 lock 生成环境失败时则可以回退到上述的手动操作方法进行重试。

## 环境一键重建

`env_requ/env.yml` 用于生成 lock 并重建 `robotwin-act`。  
`pytorch3d` 和 `curobo` 不写入 lock 解析流程（避免 `conda-lock` / PyPI 求解失败），改为在重建完成后单独补装。

先确保当前环境可用 `conda-lock`：

```
conda install -c conda-forge conda-lock
```

如果在远程机器首次使用，建议先检查：

```
conda-lock --version
```

在 `ACT/` 目录下执行：

```
bash env_requ/rebuild_env.sh
```

脚本固定行为：

1. 读取 `env_requ/env.yml` 生成 lock，并据此重建环境 `robotwin-act`
2. 若 `env_requ/env.yml` 不存在，直接报错退出
3. 环境重建后自动补装 `pytorch3d`：
   `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`
4. 环境重建后自动补装 `curobo`（对齐 `script/_install.sh`）：
   - 源码目录：`RoboTwin/envs/curobo`
   - 安装方式：`pip install -e <repo>/envs/curobo --no-build-isolation`
