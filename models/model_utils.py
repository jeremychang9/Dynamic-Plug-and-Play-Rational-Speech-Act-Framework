# coding=utf-8

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (PreTrainedTokenizer, PreTrainedModel, PretrainedConfig)


class BaseModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.toker = None
    
    def tie_tokenizer(self, toker: PreTrainedTokenizer):
        self.toker = toker

        toker_len = len(self.toker)
        vocab_size = self.toker.vocab_size

        if toker_len > vocab_size:
            resize_target = None

            # Check if model is wrapped (e.g. PEFT)
            if hasattr(self, "model") and hasattr(self.model, "resize_token_embeddings"):
                resize_target = self.model
            elif hasattr(self, "resize_token_embeddings"):
                resize_target = self
            else:
                raise RuntimeError("Cannot find a valid model to resize embeddings.")

            resize_target.resize_token_embeddings(toker_len)

            # Optional: log for debugging
            print(f"[Tokenizer] Resized embeddings from {vocab_size} → {toker_len}")
        else:
            print(f"[Tokenizer] No resizing needed: tokenizer size = {toker_len}, vocab_size = {vocab_size}")
            
