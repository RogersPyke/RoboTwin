
from .configuration_internvl_chat import InternVLChatConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class TinyVLAConfig(InternVLChatConfig):
    model_type = "tinyvla"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        policy_head_type='unet_diffusion_policy',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy_head_type = policy_head_type


from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

AutoConfig.register("tinyvla", TinyVLAConfig)
# TinyVLA reuses Qwen2-compatible tokenizer; required for AutoTokenizer.from_pretrained on tinyvla configs.
AutoTokenizer.register(
    TinyVLAConfig,
    slow_tokenizer_class=Qwen2Tokenizer,
    fast_tokenizer_class=Qwen2TokenizerFast,
)
