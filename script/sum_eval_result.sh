#!/bin/bash

# ==============================================================================
# Script: sum_eval_result.sh
# Description: Summarizes success rates from evaluation results and calculates
#              differences between 'ours' and 'baseline' models.
#
# Usage:
#   1. Auto-pairing mode (compare all ours/baseline pairs in a directory):
#      ./script/sum_eval_result.sh <eval_result_root>
#
#   2. Manual mode (compare two specific directories):
#      ./script/sum_eval_result.sh <ours_dir> <baseline_dir>
#
# Output:
#   - Prints a formatted table to stdout.
#   - In auto-pairing mode, also saves the combined result to <root>/sum.txt.
# ==============================================================================

# ------------------------------------------------------------------------------
# Function: get_success_rate
# Description: Finds the latest _result.txt in a directory and extracts the rate.
# Arguments: $1 - Directory path to search in.
# Returns: The success rate (float string) or "null".
# ------------------------------------------------------------------------------
get_success_rate() {
    local dir="$1"
    # Find the most recently modified _result.txt file
    local res_file=$(find "$dir" -name "_result.txt" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
    if [[ -z "$res_file" ]]; then echo "null"; return; fi
    # Extract the last numeric value (success rate) from the file
    grep -oE '[0-9]+\.[0-9]+|[0-9]+' "$res_file" | tail -n 1 || echo "null"
}

# ------------------------------------------------------------------------------
# Function: calculate_diff
# Description: Calculates (v1 - v2) and formats it as +X.XX or -X.XX.
# Arguments: $1 - Value 1, $2 - Value 2.
# Returns: Formatted difference string or "N/A".
# ------------------------------------------------------------------------------
calculate_diff() {
    local v1="$1" v2="$2"
    if [[ "$v1" == "null" || "$v2" == "null" ]]; then echo "N/A"; return; fi
    awk -v a="$v1" -v b="$v2" 'BEGIN {
        diff = a - b
        if (diff > 0) printf "+%.2f", diff
        else if (diff < 0) printf "%.2f", diff
        else printf "0.00"
    }'
}

# ------------------------------------------------------------------------------
# Function: scan_results
# Description: Scans a folder for subtasks and collects their success rates.
# Arguments: $1 - Folder path, $2 - Ref to results map, $3 - Ref to task list.
# ------------------------------------------------------------------------------
scan_results() {
    local dir="$1"
    local -n results_map="$2"
    local -n task_list="$3"
    local found=false
    
    # Priority 1: Scan subdirectories for _result.txt
    for sub in "$dir"/*; do
        if [[ -d "$sub" ]]; then
            local task_name=$(basename "$sub")
            # Skip common non-task directories
            [[ "$task_name" =~ ^(sum.txt|__pycache__)$ ]] && continue
            
            local rate=$(get_success_rate "$sub")
            if [[ "$rate" != "null" ]]; then
                results_map["$task_name"]="$rate"
                [[ ! " ${task_list[@]} " =~ " ${task_name} " ]] && task_list+=("$task_name")
                found=true
            fi
        fi
    done
    
    # Priority 2: Fallback to existing sum.txt if no subdirs found
    if [[ "$found" == false && -f "$dir/sum.txt" ]]; then
        while read -r line; do
            # Matches "task_name: 0.80" or "task_name 0.80"
            if [[ "$line" =~ ^([^[:space:]:]+)[:[:space:]]+([0-9]+\.[0-9]+|[0-9]+) ]]; then
                local name="${BASH_REMATCH[1]}"
                local rate="${BASH_REMATCH[2]}"
                [[ "$name" =~ ^(Task|====)$ ]] && continue
                results_map["$name"]="$rate"
                [[ ! " ${task_list[@]} " =~ " ${name} " ]] && task_list+=("$name")
            fi
        done < "$dir/sum.txt"
    fi
}

# ------------------------------------------------------------------------------
# Function: generate_pair_summary
# Description: Generates a comparison table for two directories.
# Arguments: $1 - Ours directory, $2 - Baseline directory.
# ------------------------------------------------------------------------------
generate_pair_summary() {
    local dir1="$1" dir2="$2"
    local dir1_name=$(basename "$dir1")
    local dir2_name=$(basename "$dir2")
    
    declare -A res1 res2
    declare -a tasks
    scan_results "$dir1" res1 tasks
    scan_results "$dir2" res2 tasks
    
    if [[ ${#tasks[@]} -eq 0 ]]; then
        echo "No results found for $dir1_name vs $dir2_name"
        return
    fi

    # Sort tasks alphabetically
    local IFS=$'\n'
    local sorted_tasks=($(sort <<<"${tasks[*]}"))
    unset IFS

    # Calculate column widths
    local w_task=25
    for t in "${tasks[@]}"; do [[ ${#t} -gt $w_task ]] && w_task=${#t}; done
    local w1=${#dir1_name}; [[ $w1 -lt 12 ]] && w1=12
    local w2=${#dir2_name}; [[ $w2 -lt 12 ]] && w2=12

    # Print Header
    printf "%-${w_task}s %${w1}s %${w2}s %12s\n" "Task" "$dir1_name" "$dir2_name" "Diff"
    printf "%-${w_task}s %${w1}s %${w2}s %12s\n" "$(printf '=%.0s' $(seq 1 $w_task))" "$(printf '=%.0s' $(seq 1 $w1))" "$(printf '=%.0s' $(seq 1 $w2))" "------------"

    # Print Rows
    for t in "${sorted_tasks[@]}"; do
        local v1=${res1[$t]:-null}
        local v2=${res2[$t]:-null}
        local diff=$(calculate_diff "$v1" "$v2")
        printf "%-${w_task}s %${w1}s %${w2}s %12s\n" "$t" "$v1" "$v2" "$diff"
    done
}

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

if [[ $# -eq 2 ]]; then
    # Mode: Compare two specific directories
    generate_pair_summary "$1" "$2"
elif [[ $# -eq 1 && -d "$1" ]]; then
    # Mode: Auto-pair folders in a root directory
    ROOT="$1"
    declare -A ours_folders baseline_folders
    declare -a params_list
    
    # 1. Collect all ours/baseline folders
    for d in "$ROOT"/*; do
        [[ ! -d "$d" ]] && continue
        name=$(basename "$d")
        if [[ "$name" =~ ^ours_(.+)$ ]]; then
            p="${BASH_REMATCH[1]}"
            ours_folders["$p"]="$d"
            [[ ! " ${params_list[@]} " =~ " ${p} " ]] && params_list+=("$p")
        elif [[ "$name" =~ ^baseline_(.+)$ ]]; then
            p="${BASH_REMATCH[1]}"
            baseline_folders["$p"]="$d"
            [[ ! " ${params_list[@]} " =~ " ${p} " ]] && params_list+=("$p")
        fi
    done

    # 2. Sort parameters for consistent output
    IFS=$'\n'
    params_list=($(sort <<<"${params_list[*]}"))
    unset IFS

    # 3. Process pairs and generate combined summary
    summary_file="$ROOT/sum.txt"
    {
        first=true
        for p in "${params_list[@]}"; do
            if [[ -n "${ours_folders[$p]}" && -n "${baseline_folders[$p]}" ]]; then
                [[ "$first" == false ]] && echo ""
                generate_pair_summary "${ours_folders[$p]}" "${baseline_folders[$p]}"
                first=false
            fi
        done
    } | tee "$summary_file"
    
    echo -e "\nSummary saved to: $summary_file"
else
    echo "RoboTwin Evaluation Result Summarizer"
    echo "Usage:"
    echo "  $0 <eval_result_root>           # Auto-pair ours/baseline folders"
    echo "  $0 <ours_dir> <baseline_dir>    # Compare specific folders"
    exit 1
fi
