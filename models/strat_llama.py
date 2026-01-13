# coding=utf-8

import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import get_peft_model, LoraConfig, TaskType

from models.model_utils import BaseModel
from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        # Step 1: Load config
        config = AutoConfig.from_pretrained(model_path)

        # Step 2: Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.float16,
        )

        # Step 3: Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        peft_model = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters (good for debugging LoRA setup)
        print('===============')
        peft_model.print_trainable_parameters()
        print('===============')
        # Step 4: Initialize Model with config
        instance = cls(config)
        instance.model = peft_model
        instance.toker = kwargs.get('tokenizer', None) # Pass tokenizer via kwargs
        
        return instance    
        
    def forward(
        self,
        input_ids=None,         # This is the 'context' from strat_pp.py
        attention_mask=None,    # Mask for the 'context'
        decoder_input_ids=None, # This is the 'shifted response' from strat_pp.py (starts with BOS/strat_id)
        labels=None,            # This is the 'target response' from strat_pp.py (starts with strat_id)
        validation=False,
        output_hidden_states=None,
        **kwargs
    ):
        assert self.toker is not None
        assert (self.training or validation) == (labels is not None) == (decoder_input_ids is not None)
      
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels, 
            return_dict=True,
            output_hidden_states=output_hidden_states,
            **kwargs 
        )
    
        lm_logits = outputs.logits # Logits from the base model
        masked_lm_loss = outputs.loss # Loss computed by the underlying model, respecting -100
    
        # REMOVED: Truncation of lm_logits is no longer needed/correct
        # if validation:
        #    lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()
    
        ppl_value = None # Initialize to None
        if labels is not None and masked_lm_loss is not None:
            ppl_value = torch.exp(masked_lm_loss)
        else:
            ppl_value = torch.tensor(float('inf')).to(lm_logits.device)

        if not self.training and not validation: # Inference mode (not used for training/validation, but for completion)
            return CausalLMOutputWithPast(
                loss=masked_lm_loss, # Will be None if labels were not provided
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    
        elif self.training: # Training mode
            return {'all': masked_lm_loss, 'ppl': ppl_value}
        else: # Validation mode
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(masked_lm_loss)
            return masked_lm_loss, label_size
    
    def get_hidden_states(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states[-1]  # last layer hidden states
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)
        
    # The 'generate' method already handles context and generates autoregressively,
    # and appends the strategy token. No changes needed here based on the
    # current forward pass modification.
    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, -1, -8:]
        
        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                probs = F.softmax(logits / TEMPERATURE, dim=-1)
                pred = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
        
        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None

        encoded_info = kwargs
        assert input_ids is not None and attention_mask is not None
        assert input_ids.size(1) >= 1

        logits_processor = LogitsProcessorList([
            TopKLogitsWarper(top_k=50),
            TopPLogitsWarper(top_p=0.9),
        ])

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        lm_logits = outputs.logits

        self.predict_strategy(lm_logits, encoded_info)

        strategy_token_id = encoded_info['pred_strat_id'].unsqueeze(-1) + len(self.toker) - 8
        input_ids = torch.cat([input_ids, strategy_token_id], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(strategy_token_id.shape, dtype=attention_mask.dtype)], dim=-1)

        kwargs['min_length'] = kwargs.get('min_length', 0) + input_ids.size(1)
        kwargs['max_length'] = kwargs.get('max_length', 0) + input_ids.size(1)
        kwargs['use_cache'] = True

        if len(self.toker) > self.toker.vocab_size:
            kwargs['bad_words_ids'] = [[i] for i in range(self.toker.vocab_size, len(self.toker))]

        for key in ['strat_id', 'batch_size', 'other_res', 'pred_strat_id', 'pred_strat_id_top1', 'pred_strat_id_top3', 'pred_strat_id_dist', 'decoder_input_ids']:
            if key in kwargs:
                del kwargs[key]

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        return encoded_info, generations[:, input_ids.size(1):]