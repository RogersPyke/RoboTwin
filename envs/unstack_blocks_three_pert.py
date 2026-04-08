from ._pert_mixin import PerturbationMixin
from .unstack_blocks_three import unstack_blocks_three


class unstack_blocks_three_pert(PerturbationMixin, unstack_blocks_three):

    def setup_demo(self, *args, **kwargs):
        kwargs["conservative_mode"] = True
        return super().setup_demo(*args, **kwargs)
# Purpose: Perturbation variant of unstack_blocks_three (conservative mode).
# Dependencies: _pert_mixin.PerturbationMixin, unstack_blocks_three
# Usage: ./collect_data.sh unstack_blocks_three_pert demo_clean_pert <gpu_id>
#   Same-seed: SOURCE_SEED_PATH=./data/unstack_blocks_three/demo_clean/seed.txt \
#              ./collect_data.sh unstack_blocks_three_pert demo_clean_pert <gpu_id>

from ._pert_mixin import PerturbationMixin
from .unstack_blocks_three import unstack_blocks_three


class unstack_blocks_three_pert(PerturbationMixin, unstack_blocks_three):
    """
    Diverse data collection for unstack_blocks_three with conservative perturbation.
    Conservative mode preserves success rate for the constrained unstack task.

    @input:  same args as unstack_blocks_three (passed via collect_data.py)
    @output: same data format as unstack_blocks_three (diverse hdf5 episodes)
    @scenario: Augmented expert data for the three-block unstack task.
    """

    def setup_demo(self, *args, **kwargs):
        """
        Force conservative_mode before delegating to PerturbationMixin.setup_demo.

        @input:  same as unstack_blocks_three.setup_demo(*args, **kwargs)
        @output: delegates to super().setup_demo after injecting conservative_mode
        @scenario: Tighter perturbation bounds for block unstack un-task.
        """
        pert = kwargs.get("perturbation") or {}
        kwargs["perturbation"] = pert
        pert.setdefault("conservative_mode", True)
        return super().setup_demo(*args, **kwargs)
