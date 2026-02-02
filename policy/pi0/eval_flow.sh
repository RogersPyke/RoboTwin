#!/bin/bash
# Model path: checkpoints/{train_config_name}/{model_name}/{checkpoint_id}/
# 
# Usage: 
#   Option 1: Set CHECKPOINT_FOLDER variable in this script (line 35)
#   Option 2: Set environment variable: CHECKPOINT_FOLDER=baseline bash eval_flow.sh
# 
# The script will create a symbolic link to map train_config_name (MODEL) to the 
# actual checkpoint folder (CHECKPOINT_FOLDER), allowing you to use different folder 
# names like "baseline" or "ours" without modifying other config files.

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_BLUE='\033[0;34m'
C_RESET='\033[0m'

TASKS=(
  "lift_pot"
  "place_can_basket"
  "place_dual_shoes"
  "put_object_cabinet"
  "scan_object"
  "blocks_ranking_rgb"
)

# Set the actual checkpoint folder name (e.g., "baseline" or "ours")
# This folder should exist in checkpoints/ directory
# Modify this line to change the checkpoint folder:
CHECKPOINT_FOLDERS=(
  # "baseline_3w"
  "ours_10w"
)
CONFIG="franka_clean"
MODEL="pi0_base_franka_robotwin_lora"
SEEDS=(
  0
)
GPU=0

# Get the script directory to ensure we're in the right location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_DIR="${SCRIPT_DIR}/checkpoints"

# Create symbolic link if CHECKPOINT_FOLDER is different from MODEL
# This allows using different folder names (like "baseline" or "ours") 
# while the code expects train_config_name (MODEL)
setup_checkpoint_link() {
  local actual_folder="${CHECKPOINTS_DIR}/${CHECKPOINT_FOLDER}"
  local expected_folder="${CHECKPOINTS_DIR}/${MODEL}"
  
  # Check if actual folder exists
  if [ ! -d "${actual_folder}" ]; then
    echo -e "${C_RED}Error: Checkpoint folder '${CHECKPOINT_FOLDER}' does not exist at ${actual_folder}${C_RESET}"
    exit 1
  fi
  
  # Remove existing link or directory if it exists
  if [ -L "${expected_folder}" ]; then
    rm "${expected_folder}"
    echo -e "${C_YELLOW}Removed existing symbolic link: ${expected_folder}${C_RESET}"
  elif [ -d "${expected_folder}" ] && [ ! -L "${expected_folder}" ]; then
    echo -e "${C_YELLOW}Warning: ${expected_folder} exists as a directory (not a link). Skipping link creation.${C_RESET}"
    echo -e "${C_YELLOW}If you want to use ${CHECKPOINT_FOLDER}, please rename or remove ${expected_folder} first.${C_RESET}"
    return 0
  fi
  
  # Create symbolic link
  if [ ! -e "${expected_folder}" ]; then
    ln -s "${CHECKPOINT_FOLDER}" "${expected_folder}"
    echo -e "${C_GREEN}Created symbolic link: ${expected_folder} -> ${CHECKPOINT_FOLDER}${C_RESET}"
  fi
}

# Cleanup function to remove symbolic link
cleanup_checkpoint_link() {
  local expected_folder="${CHECKPOINTS_DIR}/${MODEL}"
  if [ -L "${expected_folder}" ]; then
    rm "${expected_folder}"
    echo -e "${C_YELLOW}Removed symbolic link: ${expected_folder}${C_RESET}"
  fi
}

# Function to reorganize eval results to custom path format
# Target format: eval_result/{CHECKPOINT_FOLDER}_seed_{SEED}/{task_name}/...
# If the base directory exists, append to it. If the task directory exists, overwrite it.
reorganize_eval_result() {
  local checkpoint_folder=$1
  local seed=$2
  local task=$3
  
  # Get absolute path to eval_result directory
  local eval_result_dir="$(cd "${SCRIPT_DIR}/../../eval_result" && pwd)"
  local source_task_dir="${eval_result_dir}/${task}"
  
  # Check if source task directory exists
  if [ ! -d "${source_task_dir}" ]; then
    echo -e "${C_YELLOW}Warning: Source task directory not found: ${source_task_dir}${C_RESET}"
    return 1
  fi
  
  # Target base directory: eval_result/{CHECKPOINT_FOLDER}_seed_{SEED}
  local target_base="${eval_result_dir}/${checkpoint_folder}_seed_${seed}"
  local target_task_dir="${target_base}/${task}"
  
  # Create base directory if it doesn't exist
  mkdir -p "${target_base}"
  
  # Move the entire task directory structure to target location
  if [ -d "${target_task_dir}" ]; then
    # Task directory already exists, overwrite it
    echo -e "${C_YELLOW}Target task directory already exists: ${target_task_dir}${C_RESET}"
    echo -e "${C_YELLOW}Overwriting with new results...${C_RESET}"
    # Remove old task directory and move new one
    rm -rf "${target_task_dir}"
    mv "${source_task_dir}" "${target_task_dir}"
    echo -e "${C_GREEN}Overwritten results at: ${target_task_dir}${C_RESET}"
  else
    # Task directory doesn't exist, just move it
    mv "${source_task_dir}" "${target_task_dir}"
    echo -e "${C_GREEN}Moved results to: ${target_task_dir}${C_RESET}"
  fi
  
  return 0
}

TOTAL_TASKS=${#TASKS[@]}
TOTAL_SEEDS=${#SEEDS[@]}
TOTAL_CHECKPOINTS=${#CHECKPOINT_FOLDERS[@]}
TOTAL=$((TOTAL_TASKS * TOTAL_SEEDS * TOTAL_CHECKPOINTS))
CURRENT=0

echo -e "${C_BLUE}Starting evaluation flow: ${TOTAL_CHECKPOINTS} checkpoints × ${TOTAL_TASKS} tasks × ${TOTAL_SEEDS} seeds = ${TOTAL} evaluations${C_RESET}"
echo

# Trap to cleanup on exit
trap cleanup_checkpoint_link EXIT

for CHECKPOINT_FOLDER in "${CHECKPOINT_FOLDERS[@]}"; do
  # Setup checkpoint link before evaluation
  echo -e "${C_BLUE}Setting up checkpoint link: ${MODEL} -> ${CHECKPOINT_FOLDER}${C_RESET}"
  setup_checkpoint_link
  echo
  
  echo -e "${C_BLUE}=== Processing CHECKPOINT: ${CHECKPOINT_FOLDER} ===${C_RESET}"
  echo
  
  for seed in "${SEEDS[@]}"; do
    echo -e "${C_BLUE}=== Processing SEED: ${seed} ===${C_RESET}"
    echo
    
    for task in "${TASKS[@]}"; do
      CURRENT=$((CURRENT + 1))
      echo -e "${C_YELLOW}[${CURRENT}/${TOTAL}]${C_RESET} Evaluating: ${C_BLUE}${task}${C_RESET} (CHECKPOINT: ${CHECKPOINT_FOLDER}, SEED: ${seed})"
      
      if bash eval.sh "${task}" "${CONFIG}" "${MODEL}" "${CONFIG}" "${seed}" "${GPU}"; then
        echo -e "${C_GREEN}[${CURRENT}/${TOTAL}]${C_RESET} Completed: ${task} (CHECKPOINT: ${CHECKPOINT_FOLDER}, SEED: ${seed})"
        
        # Reorganize eval results to custom path format
        echo -e "${C_BLUE}Reorganizing results to custom path format...${C_RESET}"
        if reorganize_eval_result "${CHECKPOINT_FOLDER}" "${seed}" "${task}"; then
          echo -e "${C_GREEN}Results reorganized successfully${C_RESET}"
        else
          echo -e "${C_YELLOW}Warning: Failed to reorganize results, but evaluation completed${C_RESET}"
        fi
      else
        echo -e "${C_RED}[${CURRENT}/${TOTAL}]${C_RESET} Failed: ${task} (CHECKPOINT: ${CHECKPOINT_FOLDER}, SEED: ${seed})"
        exit 1
      fi
      echo
    done
    
    echo -e "${C_GREEN}Completed all tasks for SEED: ${seed}${C_RESET}"
    echo
  done
  
  echo -e "${C_GREEN}Completed all tasks for CHECKPOINT: ${CHECKPOINT_FOLDER}${C_RESET}"
  echo
done

echo -e "${C_GREEN}All tasks completed successfully for all checkpoints and seeds!${C_RESET}"