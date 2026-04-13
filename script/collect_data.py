#!/usr/bin/env python3
"""
Purpose: Data collection script for RoboTwin tasks.
Dependencies:
    - sapien, yaml, json, logging, os, sys, time, argparse
    - envs.* (task environments)
    - script.stats_tracker (statistics tracking)
    - script.log_utils (logging utilities)
Usage Example:
    python collect_data.py hanging_mug_pert demo_clean_pert

Features:
    - Statistics mode: Only collect success statistics without saving data
    - RRT vs Waypoint success tracking
    - Per-process logging with immediate flush
"""

import sys
import os
import time
import json
import traceback
import logging
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import yaml
import importlib

from envs import *
from script.stats_tracker import StatsTracker
from script.log_utils import setup_child_process_logging, flush_log, log_and_flush

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
UTC8 = timezone(timedelta(hours=8))
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):
    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    env_end_reset_to_init = os.environ.get("END_RESET_TO_INIT")
    if (
        env_end_reset_to_init is not None
        and env_end_reset_to_init.strip().lower() != "true"
    ):
        raise ValueError(
            f"Invalid value for END_RESET_TO_INIT: {env_end_reset_to_init!r}. Only 'true' is allowed."
        )
    if env_end_reset_to_init is not None:
        args["END_RESET_TO_INIT"] = True

    args["task_name"] = task_name
    args["task_config"] = task_config

    # Setup per-process logging
    log = setup_child_process_logging(task_name, task_config)

    # Check if statistics mode is enabled
    statistics_mode = args.get("statistics_mode", False)
    statistics_target_num = args.get(
        "statistics_target_num", args.get("episode_num", 50)
    )

    if statistics_mode:
        log.info(f"{YELLOW}[STATISTICS MODE ENABLED]{RESET}")
        log.info(f"Target: {statistics_target_num} successful episodes")
        log.info("Will skip actual data collection, only tracking success rates")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        # Use a single canonical path so Robot.set_planner sees equal left/right curobo paths
        # and sets left_planner/right_planner (avoids "no attribute left_planner" when
        # communication_flag would otherwise be True due to path string differences).
        canonical = os.path.normpath(
            os.path.abspath(get_embodiment_file(embodiment_type[0]))
        )
        args["left_robot_file"] = canonical
        args["right_robot_file"] = canonical
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    log.info("============= Config =============")
    log.info(
        f"{BLUE}Messy Table:{RESET} "
        + str(args["domain_randomization"]["cluttered_table"])
    )
    log.info(
        f"{BLUE}Random Background:{RESET} "
        + str(args["domain_randomization"]["random_background"])
    )
    if args["domain_randomization"]["random_background"]:
        log.info(
            " - Clean Background Rate: "
            + str(args["domain_randomization"]["clean_background_rate"])
        )
    log.info(
        f"{BLUE}Random Light:{RESET} "
        + str(args["domain_randomization"]["random_light"])
    )
    if args["domain_randomization"]["random_light"]:
        log.info(
            " - Crazy Random Light Rate: "
            + str(args["domain_randomization"]["crazy_random_light_rate"])
        )
    log.info(
        f"{BLUE}Random Table Height:{RESET} "
        + str(args["domain_randomization"]["random_table_height"])
    )
    log.info(
        f"{BLUE}Random Head Camera Distance:{RESET} "
        + str(args["domain_randomization"]["random_head_camera_dis"])
    )

    log.info(
        f"{BLUE}Head Camera Config:{RESET} "
        + str(args["camera"]["head_camera_type"])
        + f", "
        + str(args["camera"]["collect_head_camera"])
    )
    log.info(
        f"{BLUE}Wrist Camera Config:{RESET} "
        + str(args["camera"]["wrist_camera_type"])
        + f", "
        + str(args["camera"]["collect_wrist_camera"])
    )
    log.info(f"{BLUE}Embodiment Config:{RESET} " + embodiment_name)
    log.info(
        f"{BLUE}END_RESET_TO_INIT:{RESET} " + str(args.get("END_RESET_TO_INIT", True))
    )
    log.info("==================================")
    flush_log(log)

    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config
    args["save_path"] = os.path.join(
        args["save_path"], str(args["task_name"]), args["task_config"]
    )

    run(task, args, log)


def run(TASK_ENV, args, log):
    """
    Run data collection or statistics mode.

    @input:
        TASK_ENV: Task environment instance
        args: Configuration dictionary
        log: Logger instance
    @output: None
    @scenario: Collect data or statistics based on mode
    """
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    statistics_mode = args.get("statistics_mode", False)
    statistics_target_num = args.get(
        "statistics_target_num", args.get("episode_num", 50)
    )

    log.info(f"{BLUE}Task Name: {args['task_name']}{RESET}")
    flush_log(log)

    # Initialize statistics tracker
    stats_tracker = StatsTracker(
        save_path=args["save_path"], target_num=statistics_target_num
    )

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    def write_success_rate_file(success_count, total_tries):
        sr = 0.0 if total_tries <= 0 else float(success_count) / float(total_tries)
        sr_file = os.path.join(args["save_path"], "sr.txt")
        with open(sr_file, "w", encoding="utf-8") as file:
            file.write(f"{sr:.6f}\n")
            file.write(f"success={success_count}\n")
            file.write(f"tries={total_tries}\n")
            file.write(f"failed={total_tries - success_count}\n")
        log.info(
            f"Success rate: {sr:.2%} ({success_count}/{total_tries}), saved to {sr_file}"
        )

    if not args["use_seed"]:
        log.info(f"{YELLOW}[Start Seed and Pre Motion Data Collection]{RESET}")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            log.info(f"Exist seed file, Start from: {epid} / {suc_num}")

        target_num = statistics_target_num if statistics_mode else args["episode_num"]

        while suc_num < target_num:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                # Get strategy from perturbation mixin if available
                strategy = None
                if hasattr(TASK_ENV, "_pert_meta") and TASK_ENV._pert_meta:
                    last_aug = TASK_ENV._pert_meta.get("last_augmentation", {})
                    strategy = last_aug.get("strategy")

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    log.success(
                        f"simulate data episode {suc_num} success! (seed = {epid})"
                        + (f", strategy={strategy}" if strategy else "")
                    )
                    seed_list.append(epid)

                    stats_tracker.record_success(seed=epid, strategy=strategy)

                    # In statistics mode, skip saving trajectory data
                    if not statistics_mode:
                        TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    log.error(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    stats_tracker.record_failure(strategy=strategy)
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                log.error(" -------------")
                log.error(f"simulate data episode {suc_num} fail! (seed = {epid})")
                log.error(f"Error: {e}")
                log.error(" -------------")
                stats_tracker.record_failure()
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                stack_trace = traceback.format_exc()
                log.error(" -------------")
                log.error(f"simulate data episode {suc_num} fail! (seed = {epid})")
                log.error(f"Error: {stack_trace}")
                log.error(" -------------")
                stats_tracker.record_failure()
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            # Save seed file
            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

            # Save statistics
            stats_tracker.save()

            # Print summary periodically
            if epid % 10 == 0:
                stats_tracker.print_summary()

            flush_log(log)

        log.info(
            f"\nComplete simulation, failed {RED}{fail_num}{RESET} times / {epid} tries \n"
        )
        write_success_rate_file(suc_num, epid)

        # Print final statistics summary
        stats_tracker.print_summary()
        stats_tracker.save()

        if statistics_mode:
            log.info(f"{GREEN}[STATISTICS MODE COMPLETE]{RESET}")
            log.info(
                f"Reached target: {suc_num}/{statistics_target_num} successful episodes"
            )
            return
    else:
        log.info(f"{YELLOW}" + "Use Saved Seeds List".center(30, "-") + f"{RESET}")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========
    # Skip data collection in statistics mode
    if statistics_mode:
        log.info(f"{YELLOW}[STATISTICS MODE] Skipping data collection phase{RESET}")
        return

    if args["collect_data"]:
        log.info(f"{YELLOW}[Start Data Collection]{RESET}")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], "data", f"episode{idx}.hdf5")
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            log.info(f"{BLUE}Task name: {args['task_name']}{RESET}")

            TASK_ENV.setup_demo(
                now_ep_num=episode_idx, seed=seed_list[episode_idx], **args
            )

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            # Attach pert_meta from PerturbationMixin if present (pert tasks only).
            pert_meta = getattr(TASK_ENV, "_pert_meta", None)
            if pert_meta is not None:
                info["pert_meta"] = {
                    k: (v if not hasattr(v, "tolist") else v.tolist())
                    for k, v in pert_meta.items()
                }
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

            flush_log(log)

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
