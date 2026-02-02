#!/usr/bin/env python3
"""
Simple script to upload eval_result folder to Hugging Face
Usage: 
    python script/upload_to_hf.py --tgt_dir <target_directory> --repo_name <repository_name>
    (based on project root directory)
Environment variables: HF_TOKEN (optional, will use logged-in token if not set)
"""

from ast import parse
import os
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def arg_parser():
    parser = argparse.ArgumentParser(description="Upload eval_result folder to Hugging Face periodically")
    parser.add_argument("--interval", type=int, default=30, help="Interval in minutes between uploads (default: 30)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--tgt_dir", type=str, default="eval_result", help="Target directory to upload (default: eval_result)")
    parser.add_argument("--repo_name", type=str, default=None, help="Repository name to upload to (default: None)")
    return parser.parse_args()

def main():
    args = arg_parser()
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tgt_dir = project_root / args.tgt_dir

    # Check if eval_result folder exists
    if not tgt_dir.exists():
        raise FileNotFoundError(f"{args.tgt_dir} folder does not exist: {tgt_dir}")

    # Get token from environment variable (optional)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Using HF_TOKEN from environment variable")
    else:
        print("HF_TOKEN not found in environment, will use logged-in token if available")

    # Initialize API
    api = HfApi(token=hf_token)

    # Get username
    try:
        username = api.whoami()["name"]
        print(f"Detected Hugging Face username: {username}")
    except Exception as e:
        username = os.getenv("HF_USERNAME", os.getenv("USER", "your-username"))
        print(f"Could not auto-detect username, using: {username}")

    # Set repo name
    repo_name = args.repo_name or f"{args.tgt_dir}"
    repo_id = f"{username}/{repo_name}" if "/" not in repo_name else repo_name

    print(f"Target repository: {repo_id}")
    print(f"Source directory: {tgt_dir}")

    # Check if repo exists, create if not
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"[OK] Repo {repo_id} already exists")
    except Exception:
        # Repo does not exist, create it
        try:
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                repo_type="dataset",
                exist_ok=True,
            )
            print(f"[OK] Created new repo: {repo_id}")
        except Exception as e:
            print(f"[ERROR] Failed to create repo: {e}")
            raise

    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}] Starting upload...")
        
        try:
            upload_folder(
                folder_path=str(tgt_dir),
                repo_id=repo_id,
                token=hf_token,
                repo_type="dataset",
                ignore_patterns=[".git*", "__pycache__", "*.pyc"],
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [OK] Upload completed!")
            print(f"View at: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [ERROR] Upload failed: {e}")
            if args.once:
                raise

        if args.once:
            break

        next_run = datetime.now() + timedelta(minutes=args.interval)
        print(f"Waiting {args.interval} minutes for next upload... (Next run at: {next_run.strftime('%H:%M:%S')})")
        print("Press Ctrl+C to stop.")
        time.sleep(args.interval * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")

