import json
import pickle
import os

import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"

import torch
import transformers
from policy_heads import *
from data_utils.dataset import set_seed, load_data

from vla import *
from aloha_scripts.utils import *
from aloha_scripts.constants import TASK_CONFIGS
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from data_utils.data_collator import DataCollatorForSupervisedDataset
from data_utils.robot_data_processor import InternVL3Process
from dataclasses import dataclass, field, asdict
from typing import Optional

from vla.train.relative_early_stop import RelativeEarlyStop

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> parameters <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@dataclass
class ActionHeadArguments:
    policy_head_type: str = field(default="unet_diffusion_policy")
    state_dim: int = 7
    action_dim: int = 10
    noise_samples: int = 1

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn: bool = field(default=False)


@dataclass
class DataArguments:
    episode_first: bool = False
    task_name: str = field(default="stack_cube_2024_6_2")
    skip_mirrored_data: bool = field(default=False)
    chunk_size: int = field(default=16)
    joint_task_spec: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    local_debug: bool = field(default=False)

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)
    seed: int = field(default=0)

    freeze_vision_tower: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    # logger
    logging_dir: str = field(default='./logs')
    logging_strategy: str = field(default='steps')
    logging_steps: int = field(default=10)

    save_steps: int = field(default=10)  # 每隔多少步保存一次模型
    max_steps: int = field(default=10000)

    dataloader_pin_memory: bool = True
    # lora
    lora_enable: bool = False
    lora_module: str = "vit"
    lora_task_type: str = 'CAUSAL_LM'
    lora_r: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    policy_head_lr: Optional[float] = None

    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


@dataclass
class EarlyStopArguments:
    # Match ACT enable semantics: only enabled when BOTH are positive.
    early_stop_patience_evals: int = field(default=0)
    early_stop_rel_tol: float = field(default=0.0)
    eval_steps_for_early_stop: int = field(default=100)


def parse_param():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActionHeadArguments, EarlyStopArguments)
    )
    model_args, data_args, training_args, action_head_args, early_stop_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # print("模型路径：",model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=False, **asdict(action_head_args))

    cond_dim = config.hidden_size
    if  action_head_args.policy_head_type == 'unet_diffusion_policy':
        config.policy_head_config = AutoConfig.for_model(
            model_type=config.policy_head_type,
            global_cond_dim=cond_dim,
            action_dim=action_head_args.action_dim,
            state_dim=action_head_args.state_dim,
            noise_samples=action_head_args.noise_samples,
        )
    else:
        raise NotImplementedError(f"Unsupported policy head type {action_head_args.policy_head_type}")

    for k,v in asdict(model_args).items():
        setattr(config, k, v)

    return model_args, data_args, training_args, action_head_args, early_stop_args, config


def _parse_joint_task_spec(raw_spec):
    if raw_spec is None:
        return None
    if not isinstance(raw_spec, str) or not raw_spec.strip():
        raise ValueError("joint_task_spec must be a non-empty JSON string.")
    spec = json.loads(raw_spec)
    if not isinstance(spec, dict):
        raise ValueError("joint_task_spec must decode to a JSON object.")
    return spec


def _resolve_task_config(data_args):
    joint_task_spec = _parse_joint_task_spec(data_args.joint_task_spec)
    if joint_task_spec is not None:
        dataset_dir = joint_task_spec.get("dataset_dir")
        camera_names = joint_task_spec.get("camera_names")
        if not isinstance(dataset_dir, list) or not dataset_dir:
            raise ValueError("joint_task_spec.dataset_dir must be a non-empty list.")
        if not isinstance(camera_names, list) or not camera_names:
            raise ValueError("joint_task_spec.camera_names must be a non-empty list.")
        return {
            "task_name": str(joint_task_spec.get("task_name", data_args.task_name)),
            "dataset_dir": dataset_dir,
            "camera_names": camera_names,
            "episode_len": int(joint_task_spec.get("episode_len", 0)),
            "sample_weights": joint_task_spec.get("sample_weights", [1 for _ in dataset_dir]),
        }

    if data_args.task_name not in TASK_CONFIGS:
        raise KeyError(
            f"Task '{data_args.task_name}' is missing from TASK_CONFIGS and no joint_task_spec was provided."
        )
    return TASK_CONFIGS[data_args.task_name]

def train_bc(train_dataset=None, val_dataset=None, model=None, config=None, tokenizer=None):

    set_seed(config['training_args'].seed)
    training_args = config['training_args']
    early_stop_args = config.get('early_stop_args')

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    data_collator = DataCollatorForSupervisedDataset(computed_type=compute_dtype, tokenizer=tokenizer)

    model.config.use_cache = True
    if not isinstance(model.config.policy_head_config, dict):
        model.config.policy_head_config = model.config.policy_head_config.to_dict()
    model.config.save_pretrained(config['training_args'].output_dir)

    early_stop_enabled = (
        int(early_stop_args.early_stop_patience_evals) > 0
        and float(early_stop_args.early_stop_rel_tol) > 0.0
    )
    if early_stop_enabled:
        # Step-based validation cadence to drive TinyVLA early-stop.
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = int(early_stop_args.eval_steps_for_early_stop)

    callback = RelativeEarlyStop(
        output_dir=training_args.output_dir,
        early_stop_patience_evals=int(early_stop_args.early_stop_patience_evals),
        early_stop_rel_tol=float(early_stop_args.early_stop_rel_tol),
        eval_steps_for_early_stop=int(early_stop_args.eval_steps_for_early_stop),
        local_rank=int(training_args.local_rank) if training_args.local_rank is not None else -1,
        lora_enable=bool(training_args.lora_enable),
        lora_bias=str(training_args.lora_bias),
    )

    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer = VLATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[callback],
        **data_module,
    )
    callback.set_trainer(trainer)

    trainer.train(resume_from_checkpoint=config['training_args'].resume_from_checkpoint )

    trainer.save_state()

    # Persist early-stop summary (rank0 only).
    if int(trainer.args.local_rank) in (0, -1):
        stop_reason = str(callback.stop_reason)
        min_val_loss = float(callback.min_val_loss) if callback.min_val_loss is not None else float("nan")
        best_eval_step = int(callback.best_eval_step) if callback.best_eval_step is not None else -1
        steps_path = os.path.join(trainer.args.output_dir, "steps.txt")
        steps_meta = {
            "total_train_steps": int(trainer.state.global_step),
            "total_evals_run": int(callback.total_evals_run),
            "stop_reason": stop_reason,
            "best_eval_step": best_eval_step,
            "min_val_loss": min_val_loss,
            "early_stop_patience_evals": int(callback.early_stop_patience_evals),
            "early_stop_rel_tol": float(callback.early_stop_rel_tol),
        }
        with open(steps_path, "w", encoding="ascii") as f:
            for k in sorted(steps_meta.keys()):
                f.write(f"{k}={steps_meta[k]}\n")

    model.config.use_cache = True

    if config['training_args'].lora_enable:
        state_dict = model_load_utils.get_peft_state_maybe_zero_3(
            model.named_parameters(), config['training_args'].lora_bias
        )
        non_lora_state_dict = model_load_utils.get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if config['training_args'].local_rank == 0 or config['training_args'].local_rank == -1:
            model.config.save_pretrained(config['training_args'].output_dir)
            model.save_pretrained(config['training_args'].output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict,
                       os.path.join(config['training_args'].output_dir, 'non_lora_trainables.bin'))
    else:
        model_load_utils.safe_save_model_for_hf_trainer(trainer=trainer,
                                                  output_dir=config['training_args'].output_dir)



def main(all_config, model_config):
    set_seed(all_config["training_args"].seed)

    # get task parameters
    task_config = _resolve_task_config(all_config['data_args'])
    camera_names = task_config['camera_names']
    dataset_dir = task_config['dataset_dir']

    model_config.camera_names = task_config['camera_names']
    tokenizer = AutoTokenizer.from_pretrained(
        all_config['model_args'].model_name_or_path,
    )
    model, data_args = model_load_utils.load_model(config=all_config, vla_config=model_config, rank0_print=rank0_print)

    rank0_print(f"{RED} Using {all_config['model_args'].model_name_or_path} as VLA backbone {RESET}")
    vla_process = InternVL3Process(
        tokenizer=tokenizer,
        conv_template=model.conv_template,
        data_args=all_config['data_args'],
        camera_names=camera_names,
        num_image_token=model.num_image_token
    )

    train_dataset, val_dataset, stats = load_data(
        dataset_dir_l=dataset_dir,
        skip_mirrored_data=all_config['data_args'].skip_mirrored_data,
        camera_names=camera_names,
        chunk_size=all_config['data_args'].chunk_size,
        config=all_config,
        rank0_print=rank0_print,
        policy_class=all_config['action_head_args'].policy_head_type,
        vla_data_post_process=vla_process
    )

    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    train_bc(train_dataset=train_dataset,
             val_dataset=val_dataset,
             model=model,
             config=all_config,
             tokenizer=tokenizer
             )


if __name__ == '__main__':
    model_args, data_args, training_args, action_head_args, early_stop_args, model_config = parse_param()
    config = {
        'model_args': model_args,
        'data_args': data_args,
        'training_args': training_args,
        'action_head_args': action_head_args,
        'early_stop_args': early_stop_args,
    }

    config_dict = {k:asdict(v) if not isinstance(v, dict) else v for k,v in config.items()}

    ckpt = os.listdir(config['training_args'].output_dir)
    if config['training_args'].resume_from_checkpoint is not None:
        rank0_print(f"{RED}Resuming Training from {config['training_args'].resume_from_checkpoint}............{RESET}")
    main(all_config=config, model_config=model_config)