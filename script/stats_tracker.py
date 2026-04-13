"""
Purpose: Statistics tracking for data collection with RRT vs Waypoint breakdown.
Dependencies: Python 3.6+, json, os
Usage Example:
    from script.stats_tracker import StatsTracker
    tracker = StatsTracker(save_path="./data/task/config")
    tracker.record_success(strategy="waypoint_chain")
    tracker.record_failure()
    tracker.save()

@input: success/failure events with optional strategy tag
@output: JSON file with success counts and rates
@scenario: Track planning success rates for RRT and Waypoint methods separately
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any


class StatsTracker:
    """
    Track success statistics for data collection with method breakdown.

    Tracks:
        - Total attempts
        - Total successes/failures
        - RRT successes (direct planning or rrt_guided strategy)
        - Waypoint successes (waypoint_chain strategy)
        - Success rates for each method
    """

    def __init__(self, save_path: str, target_num: int = 50):
        """
        Initialize stats tracker.

        @input:
            save_path: str, directory to save statistics file
            target_num: int, target number of successful episodes
        @output: None
        @scenario: Initialize counters and load existing stats if present
        """
        self.save_path = save_path
        self.target_num = target_num
        self.stats_file = os.path.join(save_path, "stats.json")

        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0

        self.rrt_successes = 0
        self.rrt_failures = 0
        self.waypoint_successes = 0
        self.waypoint_failures = 0

        self.seed_list = []

        self._load_existing()

    def _load_existing(self) -> None:
        """
        Load existing statistics from file if present.

        @input: None
        @output: None
        @scenario: Resume from previous statistics
        """
        if not os.path.exists(self.stats_file):
            return

        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.total_attempts = data.get("total_attempts", 0)
            self.total_successes = data.get("total_successes", 0)
            self.total_failures = data.get("total_failures", 0)
            self.rrt_successes = data.get("rrt_successes", 0)
            self.rrt_failures = data.get("rrt_failures", 0)
            self.waypoint_successes = data.get("waypoint_successes", 0)
            self.waypoint_failures = data.get("waypoint_failures", 0)
            self.seed_list = data.get("seed_list", [])
        except Exception:
            pass

    def record_success(self, seed: int, strategy: Optional[str] = None) -> None:
        """
        Record a successful planning attempt.

        @input:
            seed: int, random seed used
            strategy: str or None, "waypoint_chain", "rrt_guided", or None (direct)
        @output: None
        @scenario: Increment appropriate success counter
        """
        self.total_attempts += 1
        self.total_successes += 1
        self.seed_list.append(seed)

        if strategy == "waypoint_chain":
            self.waypoint_successes += 1
        else:
            self.rrt_successes += 1

    def record_failure(self, strategy: Optional[str] = None) -> None:
        """
        Record a failed planning attempt.

        @input:
            strategy: str or None, "waypoint_chain", "rrt_guided", or None (direct)
        @output: None
        @scenario: Increment appropriate failure counter
        """
        self.total_attempts += 1
        self.total_failures += 1

        if strategy == "waypoint_chain":
            self.waypoint_failures += 1
        else:
            self.rrt_failures += 1

    def is_target_reached(self) -> bool:
        """
        Check if target number of successes reached.

        @input: None
        @output: bool, True if target reached
        @scenario: Used in statistics mode to stop after target successes
        """
        return self.total_successes >= self.target_num

    def get_success_rate(self) -> float:
        """
        Get overall success rate.

        @input: None
        @output: float, success rate in [0, 1]
        @scenario: Calculate overall planning success rate
        """
        if self.total_attempts <= 0:
            return 0.0
        return float(self.total_successes) / float(self.total_attempts)

    def get_rrt_success_rate(self) -> float:
        """
        Get RRT method success rate.

        @input: None
        @output: float, RRT success rate in [0, 1]
        @scenario: Calculate success rate for RRT/direct planning
        """
        rrt_total = self.rrt_successes + self.rrt_failures
        if rrt_total <= 0:
            return 0.0
        return float(self.rrt_successes) / float(rrt_total)

    def get_waypoint_success_rate(self) -> float:
        """
        Get Waypoint method success rate.

        @input: None
        @output: float, Waypoint success rate in [0, 1]
        @scenario: Calculate success rate for waypoint chain planning
        """
        wp_total = self.waypoint_successes + self.waypoint_failures
        if wp_total <= 0:
            return 0.0
        return float(self.waypoint_successes) / float(wp_total)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary.

        @input: None
        @output: Dict with all statistics
        @scenario: Prepare data for JSON serialization
        """
        UTC8 = timezone(timedelta(hours=8))
        return {
            "timestamp": datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S"),
            "target_num": self.target_num,
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.get_success_rate(),
            "rrt": {
                "successes": self.rrt_successes,
                "failures": self.rrt_failures,
                "total": self.rrt_successes + self.rrt_failures,
                "success_rate": self.get_rrt_success_rate(),
            },
            "waypoint": {
                "successes": self.waypoint_successes,
                "failures": self.waypoint_failures,
                "total": self.waypoint_successes + self.waypoint_failures,
                "success_rate": self.get_waypoint_success_rate(),
            },
            "seed_list": self.seed_list,
        }

    def save(self) -> None:
        """
        Save statistics to JSON file.

        @input: None
        @output: None
        @scenario: Persist statistics to disk
        """
        os.makedirs(self.save_path, exist_ok=True)
        data = self.to_dict()
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        """
        Print statistics summary to console.

        @input: None
        @output: None
        @scenario: Display current statistics
        """
        print("\n" + "=" * 60)
        print("STATISTICS SUMMARY")
        print("=" * 60)
        print(
            f"Target: {self.target_num} | Current: {self.total_successes}/{self.total_attempts}"
        )
        print(f"Overall Success Rate: {self.get_success_rate():.2%}")
        print("-" * 60)
        print("RRT/Direct Method:")
        print(f"  Successes: {self.rrt_successes} | Failures: {self.rrt_failures}")
        print(f"  Success Rate: {self.get_rrt_success_rate():.2%}")
        print("-" * 60)
        print("Waypoint Chain Method:")
        print(
            f"  Successes: {self.waypoint_successes} | Failures: {self.waypoint_failures}"
        )
        print(f"  Success Rate: {self.get_waypoint_success_rate():.2%}")
        print("=" * 60 + "\n")
