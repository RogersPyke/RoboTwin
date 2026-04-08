from ._pert_mixin import PerturbationMixin
from .stack_blocks_three import stack_blocks_three


class stack_blocks_three_pert(PerturbationMixin, stack_blocks_three):
    pass
# Purpose: Perturbation variant of stack_blocks_three for diverse data collection.
# Dependencies: _pert_mixin.PerturbationMixin, stack_blocks_three
# Usage: ./collect_data.sh stack_blocks_three_pert demo_clean_pert <gpu_id>
#   Same-seed: SOURCE_SEED_PATH=./data/stack_blocks_three/demo_clean/seed.txt \
#              ./collect_data.sh stack_blocks_three_pert demo_clean_pert <gpu_id>

from ._pert_mixin import PerturbationMixin
from .stack_blocks_three import stack_blocks_three


class stack_blocks_three_pert(PerturbationMixin, stack_blocks_three):
    """
    Diverse data collection for stack_blocks_three.
    Grasp approaches sampled from a cone; pre-place poses offset randomly in XY.

    @input:  same args as stack_blocks_three (passed via collect_data.py)
    @output: same data format as stack_blocks_three (diverse hdf5 episodes)
    @scenario: Augmented expert data for the three-block stacking task.
    """
    pass
