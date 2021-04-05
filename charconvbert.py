from transformers import PreTrainedTokenizerFast
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

class CharConvBertTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer_file, **kwargs):
        super().__init__(
             tokenizer_file=tokenizer_file,
             model_max_length=2048,
             mask_token='[MASK]',
             cls_token='[CLS]',
             sep_token='[SEP]',
             pad_token='[PAD]',
             unk_token='[UNK]',
             **kwargs
         )
        
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
