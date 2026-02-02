#!/bin/bash
# Purpose: Batch data collection script that iterates through multiple tasks
# Dependencies: Bash 4.0+, collect_data.sh script in the same directory
# Usage: bash collect_data_flow.sh
#        Or modify TASKS, CONFIG, and GPU variables at the top of this script
#
# This script loops through a list of tasks and collects data for each task
# using the collect_data.sh script. It provides progress tracking and colored
# output for better visibility.

# Color definitions for terminal output
C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_BLUE='\033[0;34m'
C_RESET='\033[0m'

# Configuration parameters - modify these as needed
# TASKS: Array of task names to collect data for
TASKS=(
  "lift_pot"
  "place_can_basket"
  "place_dual_shoes"
  "put_object_cabinet"
  # "scan_object"
  # "blocks_ranking_rgb"
)

# CONFIG: Task configuration name (e.g., "franka_clean", "demo_clean")
# CONFIG="franka_clean"
CONFIG="ARX-X5"

# GPU: GPU ID to use for data collection
GPU=0

# Get the script directory to ensure we're in the right location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log directory configuration
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_SCRIPT_NAME="collect_data_flow"

# Function: Generate timestamp in YYYYMMDDHHMMSS format (UTC+8)
# Input: None
# Output: Timestamp string (YYYYMMDDHHMMSS)
# Usage: Called to generate log file names
generate_timestamp() {
  # Get current time in UTC+8 (Asia/Shanghai timezone)
  TZ='Asia/Shanghai' date +"%Y%m%d%H%M%S"
}

# Function: Initialize log file and directory
# Input: None (uses LOG_DIR and LOG_SCRIPT_NAME)
# Output: Sets LOG_FILE variable, returns 0 on success, 1 on failure
# Usage: Called at script startup to set up logging
init_logging() {
  # Create log directory if it doesn't exist
  if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p "${LOG_DIR}"
    if [ $? -ne 0 ]; then
      echo -e "${C_RED}[ERROR]${C_RESET} Failed to create log directory: ${LOG_DIR}"
      return 1
    fi
  fi
  
  # Generate log file name with timestamp
  local timestamp=$(generate_timestamp)
  LOG_FILE="${LOG_DIR}/${LOG_SCRIPT_NAME}_${timestamp}.log"
  
  # Create log file with header
  {
    echo "=========================================="
    echo "Data Collection Flow Log"
    echo "Script: ${LOG_SCRIPT_NAME}.sh"
    echo "Start Time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Configuration: ${CONFIG}"
    echo "GPU ID: ${GPU}"
    echo "Total Tasks: ${#TASKS[@]}"
    echo "=========================================="
    echo
  } > "${LOG_FILE}"
  
  if [ $? -ne 0 ]; then
    echo -e "${C_RED}[ERROR]${C_RESET} Failed to create log file: ${LOG_FILE}"
    return 1
  fi
  
  return 0
}

# Function: Log message to both terminal and log file
# Input:
#   $1: log_level (string) - Log level: INFO, WARNING, ERROR, SUCCESS
#   $2: message (string) - Log message content
# Output: None (writes to terminal and log file)
# Usage: Called throughout script to log events
log_message() {
  local log_level=$1
  local message=$2
  local timestamp=$(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')
  local color_code=""
  local log_prefix=""
  
  # Determine color and prefix based on log level
  case "${log_level}" in
    "ERROR")
      color_code="${C_RED}"
      log_prefix="[ERROR]"
      ;;
    "WARNING")
      color_code="${C_YELLOW}"
      log_prefix="[WARNING]"
      ;;
    "SUCCESS")
      color_code="${C_GREEN}"
      log_prefix="[SUCCESS]"
      ;;
    "INFO")
      color_code="${C_BLUE}"
      log_prefix="[INFO]"
      ;;
    *)
      color_code="${C_RESET}"
      log_prefix="[LOG]"
      ;;
  esac
  
  # Output to terminal with color
  echo -e "${color_code}${log_prefix}${C_RESET} ${message}"
  
  # Output to log file without color codes
  echo "[${timestamp}] ${log_prefix} ${message}" >> "${LOG_FILE}"
}

# Function: Check if Python is available
# Input: None
# Output: Returns 0 if Python is available, 1 otherwise
# Usage: Called to validate Python availability
check_python() {
  if command -v python3 &> /dev/null; then
    return 0
  elif command -v python &> /dev/null; then
    return 0
  else
    return 1
  fi
}

# Function: Validate that collect_data.sh exists and is executable
# Input: None (uses SCRIPT_DIR)
# Output: Returns 0 if valid, 1 if invalid (exits on error)
# Usage: Called at script startup to ensure dependencies are available
validate_dependencies() {
  local collect_script="${SCRIPT_DIR}/collect_data.sh"
  
  if [ ! -f "${collect_script}" ]; then
    log_message "ERROR" "collect_data.sh not found at: ${collect_script}"
    exit 1
  fi
  
  if [ ! -x "${collect_script}" ]; then
    log_message "WARNING" "collect_data.sh is not executable, attempting to make it executable..."
    chmod +x "${collect_script}"
    if [ $? -ne 0 ]; then
      log_message "ERROR" "Failed to make collect_data.sh executable"
      exit 1
    fi
  fi
  
  # Check if Python is available
  if ! check_python; then
    log_message "ERROR" "Python is not available. Please install Python 3 or ensure it's in your PATH."
    log_message "ERROR" "Checked for: python3, python"
    exit 1
  fi
  
  return 0
}

# Function: Execute data collection for a single task
# Input: 
#   $1: task_name (string) - Name of the task to collect data for
#   $2: task_config (string) - Configuration name for the task
#   $3: gpu_id (integer) - GPU ID to use
# Output: Returns 0 on success, non-zero on failure
# Usage: Called for each task in the TASKS array
collect_task_data() {
  local task_name=$1
  local task_config=$2
  local gpu_id=$3
  local collect_script="${SCRIPT_DIR}/collect_data.sh"
  local temp_log="${LOG_FILE}.tmp.${task_name}"
  
  # Validate input parameters
  if [ -z "${task_name}" ] || [ -z "${task_config}" ] || [ -z "${gpu_id}" ]; then
    log_message "ERROR" "Missing required parameters for collect_task_data"
    log_message "ERROR" "Usage: collect_task_data <task_name> <task_config> <gpu_id>"
    return 1
  fi
  
  # Log task start
  log_message "INFO" "Starting data collection for task: ${task_name}, config: ${task_config}, GPU: ${gpu_id}"
  
  # Setup LD_LIBRARY_PATH for cusparselt library (required by PyTorch 2.7.1+)
  # Problem: libcusparseLt.so.0 is installed in Python site-packages/cusparselt/lib/
  # but this path is not in the default library search path (LD_LIBRARY_PATH)
  # Solution: Add the library path to LD_LIBRARY_PATH before running the script
  local cusparselt_lib_path=""
  if [ -n "${CONDA_PREFIX}" ]; then
    # Try conda environment path first
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")
    cusparselt_lib_path="${CONDA_PREFIX}/lib/python${python_version}/site-packages/cusparselt/lib"
    if [ ! -f "${cusparselt_lib_path}/libcusparseLt.so.0" ]; then
      cusparselt_lib_path=""
    fi
  fi
  
  if [ -z "${cusparselt_lib_path}" ]; then
    # Try user local path as fallback
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")
    cusparselt_lib_path="${HOME}/.local/lib/python${python_version}/site-packages/cusparselt/lib"
    if [ ! -f "${cusparselt_lib_path}/libcusparseLt.so.0" ]; then
      cusparselt_lib_path=""
    fi
  fi
  
  # Export LD_LIBRARY_PATH if library path found
  if [ -n "${cusparselt_lib_path}" ] && [ -f "${cusparselt_lib_path}/libcusparseLt.so.0" ]; then
    export LD_LIBRARY_PATH="${cusparselt_lib_path}:${LD_LIBRARY_PATH}"
    log_message "INFO" "Added cusparselt library path to LD_LIBRARY_PATH: ${cusparselt_lib_path}"
  else
    log_message "WARNING" "Could not find libcusparseLt.so.0 library - PyTorch import may fail"
  fi
  
  # Execute the data collection script and capture output to both temp log and main log
  # Use tee to output to both console and temp log file
  local exit_code=0
  bash "${collect_script}" "${task_name}" "${task_config}" "${gpu_id}" 2>&1 | tee "${temp_log}"
  exit_code=${PIPESTATUS[0]}
  
  # Append temp log to main log
  {
    echo "--- Output for task: ${task_name} ---"
    cat "${temp_log}"
    echo "--- End of output for task: ${task_name} ---"
    echo
  } >> "${LOG_FILE}"
  
  # Clean up temp log
  rm -f "${temp_log}"
  
  # Log exit code and return it (no error detection, just record the result)
  if [ ${exit_code} -eq 0 ]; then
    log_message "INFO" "Data collection script completed with exit code ${exit_code} for task: ${task_name}"
  else
    log_message "INFO" "Data collection script exited with code ${exit_code} for task: ${task_name}"
  fi
  
  return ${exit_code}
}

# Main execution flow
main() {
  # Initialize logging
  if ! init_logging; then
    echo -e "${C_RED}[ERROR]${C_RESET} Failed to initialize logging, exiting..."
    exit 1
  fi
  
  log_message "INFO" "Log file: ${LOG_FILE}"
  
  # Validate dependencies before starting
  validate_dependencies
  
  # Calculate total number of tasks
  TOTAL_TASKS=${#TASKS[@]}
  CURRENT=0
  SUCCESS_COUNT=0
  FAILED_COUNT=0
  FAILED_TASKS=()
  
  # Display startup information
  log_message "INFO" "Starting data collection flow: ${TOTAL_TASKS} tasks"
  log_message "INFO" "Configuration: ${CONFIG}"
  log_message "INFO" "GPU ID: ${GPU}"
  echo
  
  # Iterate through all tasks (continue regardless of success/failure)
  for task in "${TASKS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    log_message "INFO" "[${CURRENT}/${TOTAL_TASKS}] Collecting data for task: ${task}"
    log_message "INFO" "Task config: ${CONFIG}, GPU: ${GPU}"
    
    # Execute data collection for current task (always continue to next task)
    if collect_task_data "${task}" "${CONFIG}" "${GPU}"; then
      log_message "SUCCESS" "[${CURRENT}/${TOTAL_TASKS}] Successfully completed: ${task}"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      log_message "INFO" "[${CURRENT}/${TOTAL_TASKS}] Task completed with non-zero exit code: ${task}"
      FAILED_COUNT=$((FAILED_COUNT + 1))
      FAILED_TASKS+=("${task}")
    fi
    
    echo
  done
  
  # Display completion summary
  log_message "INFO" "All tasks processed. Summary:"
  log_message "INFO" "  Total tasks: ${TOTAL_TASKS}"
  log_message "INFO" "  Successful: ${SUCCESS_COUNT}"
  log_message "INFO" "  Failed: ${FAILED_COUNT}"
  
  if [ ${FAILED_COUNT} -gt 0 ]; then
    log_message "INFO" "  Failed tasks: ${FAILED_TASKS[*]}"
  fi
  
  # Log end time and summary
  {
    echo
    echo "=========================================="
    echo "Script completed"
    echo "End Time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Total tasks processed: ${TOTAL_TASKS}"
    echo "Successful: ${SUCCESS_COUNT}"
    echo "Failed: ${FAILED_COUNT}"
    if [ ${FAILED_COUNT} -gt 0 ]; then
      echo "Failed tasks: ${FAILED_TASKS[*]}"
    fi
    echo "=========================================="
  } >> "${LOG_FILE}"
}

# Execute main function
main

