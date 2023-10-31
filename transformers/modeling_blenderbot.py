""""BlenderbotForConditionalGeneration which inherits from BART"""

import torch
from .configuration_blenderbot import BlenderbotConfig
from .file_utils import add_start_docstrings
from .modeling_bart import BartForConditionalGeneration
BLENDER_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n'
BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/blenderbot-3B', 'facebook/blenderbot-90M']


@add_start_docstrings('The BART Model with a language modeling head. Can be used for summarization.', BLENDER_START_DOCSTRING)
class BlenderbotForConditionalGeneration(BartForConditionalGeneration):
    """
    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = BlenderbotConfig
    
    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        logits[:, self.config.bos_token_id] = -torch.finfo(torch.float16).max
        if (cur_len == max_length - 1 and self.config.eos_token_id is not None):
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits


