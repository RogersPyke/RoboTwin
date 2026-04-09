import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import subprocess
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _parse_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _query_gpu_snapshot():
    """
    Query GPU memory and compute process occupancy by nvidia-smi.
    Returns:
      (gpu_rows, proc_map, err_msg)
      - gpu_rows: list[dict]
      - proc_map: dict[gpu_uuid] -> list[dict]
      - err_msg: str or None
    """
    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc_cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_out = subprocess.check_output(gpu_cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return [], {}, f"nvidia-smi gpu query failed: {e}"

    proc_out = ""
    try:
        proc_out = subprocess.check_output(proc_cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        proc_out = ""

    gpu_rows = []
    for line in gpu_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        gpu_rows.append(
            {
                "index": parts[0],
                "uuid": parts[1],
                "name": parts[2],
                "total_mb": _parse_int(parts[3], -1),
                "used_mb": _parse_int(parts[4], -1),
                "free_mb": _parse_int(parts[5], -1),
                "util": _parse_int(parts[6], -1),
            }
        )

    proc_map = {}
    if proc_out:
        for line in proc_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_uuid = parts[0]
            row = {
                "pid": parts[1],
                "name": parts[2],
                "used_mb": _parse_int(parts[3], -1),
            }
            proc_map.setdefault(gpu_uuid, []).append(row)
    return gpu_rows, proc_map, None


def _get_visible_gpu_indices():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    result = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            result.append(token)
    return result if result else None


def _format_gpu_snapshot(gpu_rows, proc_map, target_indices=None):
    lines = []
    rows = gpu_rows
    if target_indices is not None:
        filtered = [row for row in gpu_rows if row.get("index") in target_indices]
        if filtered:
            rows = filtered
    lines.append("GPU Snapshot:")
    if target_indices is not None:
        lines.append(f"  CUDA_VISIBLE_DEVICES={','.join(target_indices)}")
    if not rows:
        lines.append("  <no gpu rows>")
        return "\n".join(lines)

    for row in rows:
        lines.append(
            "  GPU {idx} | {name} | total={total}MiB used={used}MiB free={free}MiB util={util}%".format(
                idx=row["index"],
                name=row["name"],
                total=row["total_mb"],
                used=row["used_mb"],
                free=row["free_mb"],
                util=row["util"],
            )
        )
        proc_rows = proc_map.get(row["uuid"], [])
        if not proc_rows:
            lines.append("    - no compute process")
            continue
        for proc in proc_rows:
            lines.append(
                "    - pid={pid} mem={mem}MiB name={name}".format(
                    pid=proc["pid"], mem=proc["used_mb"], name=proc["name"]
                )
            )
    return "\n".join(lines)


def _gpu_memory_guard(stage_name, min_free_mb):
    gpu_rows, proc_map, err = _query_gpu_snapshot()
    if err is not None:
        print(f"[GPU-GUARD] {stage_name}: {err}")
        return

    target_indices = _get_visible_gpu_indices()
    observed_rows = gpu_rows
    if target_indices is not None:
        filtered = [row for row in gpu_rows if row.get("index") in target_indices]
        if filtered:
            observed_rows = filtered

    low_rows = [row for row in observed_rows if row.get("free_mb", -1) >= 0 and row["free_mb"] < min_free_mb]
    if low_rows:
        snapshot = _format_gpu_snapshot(gpu_rows, proc_map, target_indices=target_indices)
        low_desc = ", ".join([f"GPU {row['index']} free={row['free_mb']}MiB" for row in low_rows])
        raise RuntimeError(
            f"[GPU-GUARD] {stage_name}: insufficient free VRAM (< {min_free_mb}MiB). {low_desc}\n{snapshot}"
        )


def _print_gpu_snapshot(tag):
    gpu_rows, proc_map, err = _query_gpu_snapshot()
    if err is not None:
        print(f"[GPU-DIAG] {tag}: {err}")
        return
    target_indices = _get_visible_gpu_indices()
    print(f"[GPU-DIAG] {tag}")
    print(_format_gpu_snapshot(gpu_rows, proc_map, target_indices=target_indices))


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
    if env_end_reset_to_init is not None and env_end_reset_to_init.strip().lower() != "true":
        raise ValueError(
            f"Invalid value for END_RESET_TO_INIT: {env_end_reset_to_init!r}. Only 'true' is allowed."
        )
    if env_end_reset_to_init is not None:
        args["END_RESET_TO_INIT"] = True

    args['task_name'] = task_name

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
        canonical = os.path.normpath(os.path.abspath(get_embodiment_file(embodiment_type[0])))
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
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\033[94mEND_RESET_TO_INIT:\033[0m " + str(args.get("END_RESET_TO_INIT", True)))
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    min_free_mb = _parse_int(os.environ.get("CUROBO_MIN_FREE_MB", "3500"), 3500)

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    print(f"[GPU-GUARD] min_free_mb={min_free_mb}")

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
        print(f"Success rate: {sr:.2%} ({success_count}/{total_tries}), saved to {sr_file}")

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                _gpu_memory_guard(
                    stage_name=f"seed_phase_before_setup_demo episode={suc_num} seed={epid}",
                    min_free_mb=min_free_mb,
                )
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", stack_trace)
                _print_gpu_snapshot(tag="seed_phase_exception")
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
        write_success_rate_file(suc_num, epid)
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")
            try:
                _gpu_memory_guard(
                    stage_name=f"data_phase_before_setup_demo episode={episode_idx}",
                    min_free_mb=min_free_mb,
                )
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

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
            except Exception:
                _print_gpu_snapshot(tag=f"data_phase_exception episode={episode_idx}")
                raise

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
