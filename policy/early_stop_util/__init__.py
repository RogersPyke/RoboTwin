# Compatibility shim: canonical implementation is rw_common.early_stop.
from rw_common.early_stop import EarlyStopRecord, RelativeEarlyStopTracker

__all__ = ["EarlyStopRecord", "RelativeEarlyStopTracker"]
