#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME=cuda

for task_path in ../../data/*/; do
    task_name=$(basename "$task_path")
    if [ "$task_name" != "processed_data" ] && [ "$task_name" != "ori" ]; then
        echo "Processing $task_name..."
        # Data is now at ../../data/${task_name}/arx_clean
        bash process_data_pi0.sh "$task_name" arx_clean 50
    fi
done
