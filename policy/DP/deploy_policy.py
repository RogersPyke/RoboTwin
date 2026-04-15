import numpy as np
from .dp_model import DP
import yaml
import sys
import os

_policy_util_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "policy_util")
)
if _policy_util_root not in sys.path:
    sys.path.insert(0, _policy_util_root)
from ckpt_util import resolve_dp_checkpoint, get_checkpoint_info


def encode_obs(observation):
    head_cam = (
        np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255
    )
    left_cam = (
        np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255
    )
    right_cam = (
        np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255
    )
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    train_task_name = usr_args.get("train_task_name", usr_args["task_name"])
    ckpt_dir = usr_args.get("ckpt_dir")
    if not ckpt_dir:
        ckpt_dir = (
            f"./policy/DP/checkpoints/{train_task_name}-{usr_args['ckpt_setting']}-"
            f"{usr_args['expert_data_num']}-{usr_args['seed']}"
        )

    checkpoint_num = usr_args.get("checkpoint_num")
    ckpt_file = resolve_dp_checkpoint(ckpt_dir, checkpoint_num)
    print(f"[DP] Resolved checkpoint: {ckpt_file}")

    action_dim = usr_args["left_arm_dim"] + usr_args["right_arm_dim"] + 2

    load_config_path = f"./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml"
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    n_obs_steps = model_training_config["n_obs_steps"]
    n_action_steps = model_training_config["n_action_steps"]

    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
