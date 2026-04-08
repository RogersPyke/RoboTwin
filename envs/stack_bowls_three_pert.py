from ._pert_mixin import PerturbationMixin
from .stack_bowls_three import stack_bowls_three


class stack_bowls_three_pert(PerturbationMixin, stack_bowls_three):
    pass
# Purpose: Perturbation variant of stack_bowls_three for diverse data collection.
# Dependencies: _pert_mixin.PerturbationMixin, stack_bowls_three
# Usage: ./collect_data.sh stack_bowls_three_pert demo_clean_pert <gpu_id>
#   Same-seed: SOURCE_SEED_PATH=./data/stack_bowls_three/demo_clean/seed.txt \
#              ./collect_data.sh stack_bowls_three_pert demo_clean_pert <gpu_id>

from ._pert_mixin import PerturbationMixin
from .stack_bowls_three import stack_bowls_three


class stack_bowls_three_pert(PerturbationMixin, stack_bowls_three):
    """
    Diverse data collection for stack_bowls_three.
    Grasp cone sampling + pre-place XY offset for diverse approach trajectories.

    @input:  same args as stack_bowls_three (passed via collect_data.py)
    @output: same data format as stack_bowls_three (diverse hdf5 episodes)
    @scenario: Augmented expert data for the three-bowl stacking task.
    """
    pass
