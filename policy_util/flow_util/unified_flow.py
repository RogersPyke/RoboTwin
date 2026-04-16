#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified flow scheduler for ACT/DP/TinyVLA training pipelines.

Usage:
    cd policy/<MODEL>
    bash __flow.sh

The flow will:
1. Process data for all tasks defined in unified config
2. Train models in parallel using GPU slots from gpu_parallel config
"""

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import yaml


@dataclass
class Job:
    slot: int
    phase: str
    task_id: str
    queue_idx: int
    process: subprocess.Popen
    log_path: Path
    log_file: TextIO = field(repr=False)


class BaseFlowScheduler(ABC):
    """
    Base class for training flow schedulers.

    Subclasses only need to define:
    - MODEL_NAME: Model identifier (ACT, DP, TinyVLA)
    - get_process_data_cmd(): Return command for process_data step
    """

    MODEL_NAME: str = ""
    UNIFIED_CFG_PATH: Optional[Path] = None

    def __init__(self, policy_dir: Path):
        self.policy_dir = policy_dir
        self.active_jobs: Dict[int, Job] = {}
        self._setup_signal_handlers()

        policy_util_root = policy_dir.parent.parent / "policy_util"
        if self.UNIFIED_CFG_PATH is None:
            self.UNIFIED_CFG_PATH = policy_util_root / "config" / "tr.yaml"

        sys.path.insert(0, str(policy_util_root))
        from log_util.flow_log import (
            dump_log_tail_to_stderr,
            ensure_logs_dir,
            inject_flow_child_env,
            open_flow_text_log,
            rename_log_with_pid,
            safe_filename_part,
            utc8_now_str,
        )

        self.dump_log_tail_to_stderr = dump_log_tail_to_stderr
        self.ensure_logs_dir = ensure_logs_dir
        self.inject_flow_child_env = inject_flow_child_env
        self.open_flow_text_log = open_flow_text_log
        self.rename_log_with_pid = rename_log_with_pid
        self.safe_filename_part = safe_filename_part
        self.utc8_now_str = utc8_now_str

    def _setup_signal_handlers(self) -> None:
        atexit.register(self._cleanup_all_jobs)
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._on_signal)

    def _on_signal(self, signum: int, _frame) -> None:
        self._cleanup_all_jobs()
        raise SystemExit(128 + signum)

    def _kill_process_group(self, pid: int) -> None:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def _cleanup_all_jobs(self) -> None:
        for pid, job in list(self.active_jobs.items()):
            if job.process.poll() is None:
                self._kill_process_group(pid)
            try:
                job.log_file.close()
            except Exception:
                pass
        self.active_jobs.clear()

    def load_unified_config(self) -> Dict[str, Any]:
        with open(self.UNIFIED_CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_model_config(self, unified_cfg: Dict[str, Any]) -> Dict[str, Any]:
        global_cfg = unified_cfg.get("global", {})
        model_cfg = unified_cfg.get(self.MODEL_NAME, {})
        result = dict(global_cfg)
        for key, value in model_cfg.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = BaseFlowScheduler._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_process_data_tasks(self, tr_tasks: List[Dict[str, Any]]) -> List[str]:
        unique_tasks = []
        seen = set()
        for task in tr_tasks:
            if "data_folder" in task:
                task_name = task.get("task_name", task.get("task_id", "unknown"))
            elif "data_sources" in task:
                for src in task["data_sources"]:
                    task_name = src.get("task_name", "unknown")
                    if task_name not in seen:
                        seen.add(task_name)
                        unique_tasks.append(task_name)
                continue
            if task_name not in seen:
                seen.add(task_name)
                unique_tasks.append(task_name)
        return unique_tasks

    def _check_hdf5_integrity(self, hdf5_path: Path) -> Tuple[bool, str]:
        """
        Check HDF5 file integrity for processed data.

        @input: [Path, path to episode hdf5 file]
        @output: [Tuple[bool, str], (is_valid, error_message)]
        @scenario: [Verify HDF5 contains required keys for training]
        """
        try:
            import h5py

            with h5py.File(str(hdf5_path), "r") as f:
                if "action" not in f:
                    return False, "missing 'action' dataset"
                if "observations" not in f:
                    return False, "missing 'observations' group"
                obs = f["observations"]
                if "qpos" not in obs:
                    return False, "missing 'observations/qpos' dataset"
                if "images" not in obs:
                    return False, "missing 'observations/images' group"
                action_shape = f["action"].shape
                if len(action_shape) != 2 or action_shape[0] == 0:
                    return False, f"invalid action shape: {action_shape}"
                qpos_shape = obs["qpos"].shape
                if len(qpos_shape) != 2 or qpos_shape[0] == 0:
                    return False, f"invalid qpos shape: {qpos_shape}"
            return True, ""
        except Exception as e:
            return False, str(e)

    def check_data_item_ready(
        self, task_name: str, task_config: str, expert_num: int
    ) -> Tuple[bool, List[str]]:
        """
        Check if a single (task_name, task_config) data item is ready.

        @input: [str, task_name], [str, task_config], [int, expert_num]
        @output: [Tuple[bool, List[str]], (is_ready, missing_items)]
        @scenario: [Verify processed data exists and is valid for one data item]
        """
        missing = []
        sub_key = f"sim-{task_name}-{task_config}-{expert_num}"

        sim_cfg_path = self.policy_dir / "SIM_TASK_CONFIGS.json"
        if not sim_cfg_path.is_file():
            missing.append(f"SIM_TASK_CONFIGS.json not found")
            return False, missing

        try:
            with open(sim_cfg_path, "r", encoding="utf-8") as f:
                sim_configs = json.load(f)
        except Exception as e:
            missing.append(f"Failed to load SIM_TASK_CONFIGS.json: {e}")
            return False, missing

        if sub_key not in sim_configs:
            missing.append(f"Missing config key: {sub_key}")
            return False, missing

        entry = sim_configs[sub_key]
        dataset_dir = entry.get("dataset_dir", "")
        if not dataset_dir:
            missing.append(f"Empty dataset_dir for {sub_key}")
            return False, missing

        dataset_path = self.policy_dir / dataset_dir
        if not dataset_path.is_dir():
            missing.append(f"Dataset dir not found: {dataset_dir}")
            return False, missing

        expected_num = entry.get("num_episodes", expert_num)
        for j in range(expected_num):
            ep_file = dataset_path / f"episode_{j}.hdf5"
            if not ep_file.is_file():
                missing.append(f"Missing episode_{j}.hdf5 in {dataset_dir}")
                continue

            is_valid, err = self._check_hdf5_integrity(ep_file)
            if not is_valid:
                missing.append(f"Invalid episode_{j}.hdf5: {err}")

        return len(missing) == 0, missing

    def get_missing_data_items(
        self, tr_tasks: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, int]]:
        """
        Get list of missing data items that need processing.

        @input: [List[Dict], tr_tasks from config]
        @output: [List[Tuple[str, str, int]], list of (task_name, task_config, expert_num)]
        @scenario: [Identify which data items need to be processed]
        """
        missing_items = []
        seen = set()

        for task in tr_tasks:
            data_sources = task.get("data_sources", [])
            if not data_sources:
                continue

            for src in data_sources:
                task_name = src.get("task_name")
                task_config = src.get("task_config")
                expert_num = src.get("expert_num")

                if not all([task_name, task_config, expert_num]):
                    continue

                item_key = (task_name, task_config, expert_num)
                if item_key in seen:
                    continue
                seen.add(item_key)

                is_ready, _ = self.check_data_item_ready(
                    task_name, task_config, expert_num
                )
                if not is_ready:
                    missing_items.append(item_key)

        return missing_items

    @abstractmethod
    def get_process_data_cmd(
        self, task_name: str, task_config: str, expert_num: str
    ) -> List[str]:
        """Return command list for process_data step."""
        pass

    def get_env_prefix(self) -> str:
        """Return environment variable prefix for this model."""
        if self.MODEL_NAME == "TinyVLA":
            return "TVLA"
        return self.MODEL_NAME

    def run_process_data_steps(
        self, env: dict, task_data: List[str], gpu_tag: str
    ) -> int:
        logs = self.ensure_logs_dir(self.policy_dir)
        for q_idx, task_name in enumerate(task_data):
            ts = self.utc8_now_str()
            safe_task = self.safe_filename_part(task_name)
            tmp_path = (
                logs
                / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_tmp.log"
            )
            final_path = (
                logs
                / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_pid{{pid}}.log"
            )
            lf = self.open_flow_text_log(tmp_path)
            lf.write(
                f"# flow_meta kind=process_data task_name={task_name} slot=0 gpu={gpu_tag} queue_idx={q_idx} ts_utc8={ts}\n"
            )
            lf.flush()

            cmd = self.get_process_data_cmd(task_name, "demo_clean", "100")
            if shutil.which("stdbuf"):
                cmd = ["stdbuf", "-oL", "-eL"] + cmd

            print(
                f"[flow][slot=0][process_data][task={task_name}][gpu={gpu_tag}] log={tmp_path} starting",
                flush=True,
            )
            p = subprocess.Popen(
                cmd,
                cwd=str(self.policy_dir),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pid = p.pid
            lf.write(f"# child_pid={pid}\n")
            lf.flush()
            done_final = final_path.parent / final_path.name.format(pid=pid)
            self.rename_log_with_pid(tmp_path, done_final)
            code = p.wait()
            try:
                lf.close()
            except Exception:
                pass
            if code != 0:
                print(
                    f"[flow][process_data][task={task_name}] failed code={code} log={done_final}",
                    file=sys.stderr,
                    flush=True,
                )
                self.dump_log_tail_to_stderr(
                    done_final, f"[flow][process_data][task={task_name}]"
                )
                return code
            print(
                f"[flow][process_data][task={task_name}] done log={done_final}",
                flush=True,
            )
        return 0

    def run_process_data_items(
        self, env: dict, data_items: List[Tuple[str, str, int]], gpu_tag: str
    ) -> int:
        """
        Process only missing data items with fine-grained control.

        @input: [dict, env], [List[Tuple[str, str, int]], data items], [str, gpu_tag]
        @output: [int, exit code]
        @scenario: [Process only missing (task_name, task_config, expert_num) items]
        """
        logs = self.ensure_logs_dir(self.policy_dir)

        for q_idx, (task_name, task_config, expert_num) in enumerate(data_items):
            ts = self.utc8_now_str()
            safe_task = self.safe_filename_part(f"{task_name}_{task_config}")
            tmp_path = (
                logs
                / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_tmp.log"
            )
            final_path = (
                logs
                / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_pid{{pid}}.log"
            )
            lf = self.open_flow_text_log(tmp_path)
            lf.write(
                f"# flow_meta kind=process_data task_name={task_name} task_config={task_config} expert_num={expert_num} slot=0 gpu={gpu_tag} queue_idx={q_idx} ts_utc8={ts}\n"
            )
            lf.flush()

            cmd = self.get_process_data_cmd(task_name, task_config, str(expert_num))
            if shutil.which("stdbuf"):
                cmd = ["stdbuf", "-oL", "-eL"] + cmd

            print(
                f"[flow][slot=0][process_data][task={task_name}][config={task_config}][num={expert_num}][gpu={gpu_tag}] log={tmp_path} starting",
                flush=True,
            )
            p = subprocess.Popen(
                cmd,
                cwd=str(self.policy_dir),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pid = p.pid
            lf.write(f"# child_pid={pid}\n")
            lf.flush()
            done_final = final_path.parent / final_path.name.format(pid=pid)
            self.rename_log_with_pid(tmp_path, done_final)
            code = p.wait()
            try:
                lf.close()
            except Exception:
                pass
            if code != 0:
                print(
                    f"[flow][process_data][task={task_name}][config={task_config}] failed code={code} log={done_final}",
                    file=sys.stderr,
                    flush=True,
                )
                self.dump_log_tail_to_stderr(
                    done_final,
                    f"[flow][process_data][task={task_name}][config={task_config}]",
                )
                return code
            print(
                f"[flow][process_data][task={task_name}][config={task_config}] done log={done_final}",
                flush=True,
            )
        return 0

    def start_slot_job(
        self,
        slot: int,
        phase: str,
        task_id: str,
        gpu_id: int,
        env: dict,
        queue_idx: int,
        seed: int,
    ) -> Job:
        env_prefix = self.get_env_prefix()
        slot_env = env.copy()
        slot_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        slot_env[f"{env_prefix}_FLOW_GPU"] = str(gpu_id)
        slot_env[f"{env_prefix}_FLOW_SLOT"] = str(slot)
        slot_env[f"{env_prefix}_FLOW_TASK_ID"] = task_id
        slot_env[f"{env_prefix}_FLOW_PHASE"] = phase

        logs = self.ensure_logs_dir(self.policy_dir)
        ts = self.utc8_now_str()
        safe_task_id = self.safe_filename_part(task_id)
        tmp_path = (
            logs
            / f"flow_{phase}_{safe_task_id}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_tmp.log"
        )
        final_path = (
            logs
            / f"flow_{phase}_{safe_task_id}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_pid{{pid}}.log"
        )

        lf = self.open_flow_text_log(tmp_path)
        lf.write(
            f"# flow_meta kind={phase} task_id={task_id} slot={slot} gpu={gpu_id} queue_idx={queue_idx} ts_utc8={ts}\n"
        )
        lf.flush()

        cmd = [
            "python3",
            "_tr_wrapper.py",
            "--task-id",
            task_id,
            "--yaml",
            str(self.UNIFIED_CFG_PATH),
            "--gpu-id",
            str(gpu_id),
            "--seed",
            str(seed),
        ]

        print(
            f"[flow][slot={slot}][{phase}][task_id={task_id}][gpu={gpu_id}] log={tmp_path} starting",
            flush=True,
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(self.policy_dir),
            env=slot_env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = process.pid
        lf.write(f"# child_pid={pid}\n")
        lf.flush()
        done_final = final_path.parent / final_path.name.format(pid=pid)
        self.rename_log_with_pid(tmp_path, done_final)

        print(
            f"[flow][slot={slot}][{phase}][task_id={task_id}][gpu={gpu_id}] log={done_final} pid={pid}",
            flush=True,
        )
        return Job(
            slot=slot,
            phase=phase,
            task_id=task_id,
            queue_idx=queue_idx,
            process=process,
            log_path=done_final,
            log_file=lf,
        )

    def run(self) -> int:
        unified_cfg = self.load_unified_config()
        model_cfg = self.get_model_config(unified_cfg)
        parallel = model_cfg.get("gpu_parallel", [0])
        tr_tasks = model_cfg.get("tr_tasks", [])
        seed = model_cfg.get("seed", 0)

        if not parallel:
            print("[flow] gpu_parallel is empty", file=sys.stderr)
            return 1
        if not tr_tasks:
            print("[flow] tr_tasks is empty", file=sys.stderr)
            return 1

        gpu_tag = str(parallel[0])
        env = self.inject_flow_child_env(os.environ.copy())

        print(
            f"[flow] main MODEL={self.MODEL_NAME} PARALLEL={parallel} SEED={seed} TR_TASKS={len(tr_tasks)} CONFIG={self.UNIFIED_CFG_PATH}",
            flush=True,
        )

        missing_items = self.get_missing_data_items(tr_tasks)
        if missing_items:
            print(
                f"[flow] Data check: {len(missing_items)} items need processing",
                flush=True,
            )
            for task_name, task_config, expert_num in missing_items:
                is_ready, missing = self.check_data_item_ready(
                    task_name, task_config, expert_num
                )
                print(
                    f"  - {task_name}/{task_config}/{expert_num}: {missing[:3]}{'...' if len(missing) > 3 else ''}",
                    flush=True,
                )

            pd_code = self.run_process_data_items(env, missing_items, gpu_tag)
            if pd_code != 0:
                return pd_code
        else:
            print(
                "[flow] Data check: all items ready, skipping process_data", flush=True
            )

        next_idx = 0
        total = len(tr_tasks)
        slot_count = len(parallel)

        def try_fill_slots() -> None:
            nonlocal next_idx
            while next_idx < total and len(self.active_jobs) < slot_count:
                slot = None
                for s in range(slot_count):
                    if not any(j.slot == s for j in self.active_jobs.values()):
                        slot = s
                        break
                if slot is None:
                    break
                task = tr_tasks[next_idx]
                task_id = task.get("task_id", f"task_{next_idx}")
                gpu_id = parallel[slot]
                job = self.start_slot_job(
                    slot, "train", task_id, gpu_id, env, next_idx, seed
                )
                self.active_jobs[job.process.pid] = job
                next_idx += 1

        try_fill_slots()

        while self.active_jobs:
            failed = False
            for pid, job in list(self.active_jobs.items()):
                code = job.process.poll()
                if code is None:
                    continue
                del self.active_jobs[pid]
                try:
                    job.log_file.close()
                except Exception:
                    pass
                if code != 0:
                    failed = True
                    hdr = f"[flow][slot={job.slot}][{job.phase}][task_id={job.task_id}]"
                    print(
                        f"{hdr} failed code={code} log={job.log_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.dump_log_tail_to_stderr(job.log_path, hdr)
                    break
                try_fill_slots()
            if failed:
                self._cleanup_all_jobs()
                print("[flow] job failed, aborted", file=sys.stderr)
                return 1
            time.sleep(0.15)

        return 0
