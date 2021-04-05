from transformers import PreTrainedTokenizerFast, DataCollatorForWholeWordMask
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import _collate_batch
from transformers.utils import logging
import os

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

@dataclass
class CharBertDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            atts = [e['global_attention_mask'] for e in examples]
        else:
            atts = examples
            examples = [{'global_attention_mask': e} for e in examples]

        batch_attention = _collate_batch(atts, self.tokenizer)
        res = super().__call__(examples)
        res['global_attention_mask'] = batch_attention
        return res

class CharBertTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer_file, max_global_attention=512, **kwargs):
        super().__init__(
             tokenizer_file=tokenizer_file,
             model_max_length=4096,
             mask_token='[MASK]',
             cls_token='[CLS]',
             sep_token='[SEP]',
             pad_token='[PAD]',
             unk_token='[UNK]',
             **kwargs
         )
        self.word_start = set(v for k, v in self.vocab.items() if '##' not in k and k != '[PAD]' and k != '[UNK]')
        self.max_global_attention = max_global_attention
    
    def create_global_attention_mask(self, sentence):
        mask = []
        attn_count = 0
        for char in sentence:
            if char in self.word_start and attn_count < self.max_global_attention:
                mask.append(1)
                attn_count += 1
            else:
                mask.append(0)
        return mask
    
    def word_start_attention(self, output):
        ids = output['input_ids']
        atts = []
        if not ids:
            return []
        else:
            if type(ids[0]) is list:
                for sentence in ids:
                    atts.append(self.create_global_attention_mask(sentence))
            else:
                atts = self.create_global_attention_mask(ids)
        return atts
        
    
    def __call__(self, *argv, **kwargs):
        output = super().__call__(*argv, **kwargs)    
        output['global_attention_mask'] = self.word_start_attention(output)
        return output
    
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