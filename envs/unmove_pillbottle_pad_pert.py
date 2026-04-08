from ._pert_mixin import PerturbationMixin
from .unmove_pillbottle_pad import unmove_pillbottle_pad


class unmove_pillbottle_pad_pert(PerturbationMixin, unmove_pillbottle_pad):

    def setup_demo(self, *args, **kwargs):
        kwargs["conservative_mode"] = True
        return super().setup_demo(*args, **kwargs)
