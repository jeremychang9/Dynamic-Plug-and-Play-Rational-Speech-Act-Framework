#! /usr/bin/env python3
# coding=utf-8

import os
import sys
import re
import time
import string
import logging
import argparse
import numpy as np
from enum import IntEnum
from tqdm import trange
from difflib import SequenceMatcher
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from utils.building_utils import boolean_string, build_model, deploy_model
from inputters import inputters
from inputters.inputter_utils import _norm

from transformers import logging as tf_logging
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.cache_utils import DynamicCache

sys.path.append("..") 
sys.path.insert(0, '../PAGE')

from StrategyClassifier_head.pplm_classification_head import ClassificationHead

tf_logging.set_verbosity_error()

file_log = None
strat_pred_acc = 0

class LossType(IntEnum):
    NONE = 0
    ENGAGEMENT = 1
    STRATEGY = 2
    EMOTION = 3
    ALL = 4

SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

DISCRIMINATOR_MODELS_PARAMS = {
    # "Engagement": {
    #     "path": "../EngagementClassifier_head/output_master/NSP_classifier_head_epoch_5.pt",
    #     "class_size": 2,
    #     "embed_size": 1024,
    #     "class_vocab": {"0": 0, "1": 1},
    #     "default_class": 0,
    #     "pretrained_model": "../DialoGPT/model-medium", 
    # },
    "Strategy": {
        "path": "../StrategyClassifier_head/output_llama/ESC_classifier_head_epoch_3.pt",
        "class_size": 8,
        "embed_size": 2048,
        "pretrained_model": "../GenerationModel/DATA/strat_llama.strat_llama/2025-09-14141406.5e-05.8.2gpu/epoch-3.bin",
        "class_vocab": {"Others": 0, "Reflection of feelings": 1, "Information": 2, "Restatement or Paraphrasing": 3,
                        "Providing Suggestions": 4, "Affirmation and Reassurance": 5, "Question": 6, "Self-disclosure": 7}
    },
    "Emotion_GO": {
        "path": "../EmotionClassifier_head/output_llama/GO_classifier_head_epoch_3.pt",
        "class_size": 28,
        "embed_size": 2048,
        "pretrained_model": "../GenerationModel/DATA/strat_llama.strat_llama/2025-09-14141406.5e-05.8.2gpu/epoch-3.bin",
        "class_vocab": {"surprise": 0, "embarrassment": 1, "anger": 2, "love": 3, "amusement": 4,
                        "gratitude": 5, "grief": 6, "confusion": 7, "caring": 8, "joy": 9,
                        "annoyance": 10, "disapproval": 11, "admiration": 12, "pride": 13, "excitement": 14,
                        "relief": 15, "nervousness": 16, "optimism": 17, "fear": 18, "sadness": 19,
                        "neutral": 20, "approval": 21, "curiosity": 22, "disappointment": 23, "remorse": 24,
                        "desire": 25, "disgust": 26, "realization": 27}
    }
   
}

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    # logits: (batch_size, vocab_size)
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        # Remove tokens with a probability less than the top-k tokens
        topk_threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(logits < topk_threshold, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Mask tokens where cumulative probability > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the mask to the right to always keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to the original index locations
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )

        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

def postprocess_response(response):
    # Split into sentences using regex to match sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', response)
    
    # Check if the last sentence is incomplete (does not end with proper punctuation)
    if sentences and not re.search(r'[.!?]$', sentences[-1]):
        sentences.pop()  # Remove the last incomplete sentence
        
    # Truncate to max_sentences and join them back into a complete string
    truncated_text = ' '.join(sentences).strip()
    
    if not truncated_text:
        return response
    else:
        return truncated_text

def encode_page(dialog, tokenizer, device):
    dialog_len = len(dialog)
    
    tk_id = []
    at_mk = []
    spk = []
    spk_flag = False
    for index, u in enumerate(dialog): 
        encoded_output = tokenizer(u)
        tid = encoded_output.input_ids
        atm = encoded_output.attention_mask
        tk_id.append(torch.tensor(tid, dtype=torch.long, device=device))
        at_mk.append(torch.tensor(atm, device=device))
        if spk_flag == False:
            s = 0
            spk.append(s)
            spk_flag = True
        else:
            s = 1
            spk.append(s)
            spk_flag = False
    
    spk = torch.tensor(spk)
    same_spk = spk.unsqueeze(1) == spk.unsqueeze(0)
    other_spk = same_spk.eq(False).long().tril(0)
    same_spk = same_spk.long().tril(0)
    msk = torch.ones_like(same_spk, dtype=torch.long).tril(0)
    
    adj = torch.stack([same_spk, other_spk], dim=0)
    tk_id = pad_sequence(tk_id, batch_first=True, padding_value=1)
    at_mk = pad_sequence(at_mk, batch_first=True, padding_value=0)
    
    return tk_id, at_mk, msk, adj, dialog_len

def get_emotional_state(histories, tokenizer_page, PAGE, speaker_roles, emo_weight, str_weight, device):
    tk_id, at_mk, msk, _, _ = encode_page(histories, tokenizer_page, device)
    
    msk = msk.unsqueeze(0).cuda()
    tk_id = tk_id.unsqueeze(0)
    at_mk = at_mk.unsqueeze(0)
    adj = msk.clone()
    adj = adj.long().cuda()
    
    label = 1 # burden parameter
    
    # Predict ECE and emotion effect
    prediction, emotion, _ = PAGE(tk_id, at_mk, msk, adj, label)

    # Default threshold (0.5) for evidence prediction
    ed_prediction = (prediction[0] > 0.5).long()

    # Emotion argmax (whether emotion effect happened)
    emotion_argmax = emotion.argmax(dim=-1)[0]

    # Lower threshold (0.35) for rows where emotion effect occurred
    affected_rows = (emotion_argmax == 1).nonzero(as_tuple=True)[0]
    for row in affected_rows.tolist():
        ed_prediction[row, :row + 1] = (prediction[0][row, :row + 1] > 0.35).long()
    
    usr_eds = []
    for i, role in reversed(list(enumerate(speaker_roles[0]))):
        if role == 'usr':
            usr_eds.append(ed_prediction[i])
        elif role == 'sys' and usr_eds:
            break
        else:
            usr_eds.append(None)
                        
    user_emotional_state, emo_w, str_w = get_user_emotional_state(
                                    usr_eds, speaker_roles, emo_weight, str_weight, prediction)
    
    return user_emotional_state, emo_w, str_w

def get_user_emotional_state(user_emotions, speaker_roles, emo_weight, str_weight, prediction):
    """
    Determine the user's emotional state based on predicted probabilities and speaker roles.

    Args:
        user_emotions (list): List of user emotion states (None for system entries).
        speaker_roles (list): Nested list of roles (e.g., [['usr', 'sys', ...]]).
        emo_weight (float): Initial weight for emotional dimension.
        str_weight (float): Initial weight for strategic dimension.
        prediction (tensor): Model prediction tensor.

    Returns:
        tuple: (user_emotional_state (str), emo_w (float), str_w (float))
    """

    emo_w, str_w = 0.0, 0.0
    user_emotional_state = "neutral"

    for i, emotion in enumerate(user_emotions):
        if emotion is None:  # skip system entries
            continue

        self_prob, inter_prob = 0.0, 0.0
        role_index = i + 1

        # Collect max probabilities for user and system roles
        for j, role in enumerate(speaker_roles[0]):
            prob = round(prediction[0][-role_index][j].item(), 2)
            if role == "usr":
                self_prob = max(self_prob, prob)
            elif role == "sys":
                inter_prob = max(inter_prob, prob)

        # Case 1: User in inter-personal state
        if inter_prob >= 0.5:
            user_emotional_state = "inter-personal"  # system should be self-contagion
            
            weighted_str = 1.0 * inter_prob
            str_w = max(str_w, weighted_str)
            
            if self_prob >= 0.5:  # hybrid state
                weighted_emo = 1.0  * self_prob
                emo_w = max(emo_w, weighted_emo)
            else:                
                weighted_emo = 0.5  * ( 1 - inter_prob )
                emo_w = weighted_emo
                emo_w = max(emo_w, weighted_emo)
                
            return user_emotional_state, emo_w, str_w

        # Case 2: User in self-contagion state
        elif self_prob >= 0.5:
            user_emotional_state = "self-contagion"  # system should be inter-personal
            
            weighted_emo = 0.5 * self_prob # 0.5
            emo_w = max(emo_w, weighted_emo)
            
            weighted_str = 0.5 * self_prob # 0.5
            str_w = max(str_w, weighted_str)

    return user_emotional_state, emo_w, str_w

def RSA_inference_llama(
    log_score,          # (bsz, world_num, vocab)
    worldpriors,        # (bsz, world_num)
    bad_words_mask,     # (bsz, vocab)
    top_k,
    top_p,
):
    """
    Shape-preserving RSA inference for LLaMA + LoRA.

    Assumptions:
        - world index 0 is the target persona
        - log_score is S0 logits for each world
    """

    beta = 0.7   # soften S0
    alpha = 1.0  # listener contrast strength
    lambda_rsa = 0.05

    bsz, world_num, vocab = log_score.shape

    # ------------------------------------------------
    # 1. Broadcast world priors to token level
    #    (bsz, vocab, world_num)
    # ------------------------------------------------
    worldprior_t = worldpriors.repeat(1,log_score.size(1),1).transpose(dim0=1, dim1=2).contiguous()

    # ------------------------------------------------
    # 2. Literal speaker S0 for target persona
    #    (bsz, vocab)
    # ------------------------------------------------
    speaker_prior = log_score.select(1, 0)

    # ------------------------------------------------
    # 3. Prepare S0 for listener (transpose preserved)
    #    (bsz, vocab, world_num)
    # ------------------------------------------------
    log_score = log_score.transpose(1, 2).contiguous()

    # Normalize over vocab dimension (critical for LLaMA)
    log_score = log_score - log_score.mean(dim=1, keepdim=True)
    log_score = log_score / (log_score.std(dim=1, keepdim=True) + 1e-6)

    # Soften literal speaker
    log_score = log_score * beta

    # ------------------------------------------------
    # 4. L0: Listener posterior p(w | u)
    #    (bsz, vocab, world_num)
    # ------------------------------------------------
    listener_logits = log_score + worldprior_t
    listener_posterior = listener_logits - torch.logsumexp(
        listener_logits, dim=2, keepdim=True
    )

    # ------------------------------------------------
    # 5. Contrastive listener score 
    #    (bsz, vocab)
    # ------------------------------------------------
    l_pos = listener_posterior[:, :, 0]
    l_neg = torch.logsumexp(listener_posterior[:, :, 1:], dim=2)
    
    listener_score = l_pos - l_neg

    # scale-free normalization (CRITICAL)
    listener_score = listener_score - listener_score.mean(dim=1, keepdim=True)
    listener_score = listener_score / (listener_score.std(dim=1, keepdim=True) + 1e-6)
    
    listener_score = alpha * listener_score

    # ------------------------------------------------
    # 6. S1 speaker (weak fusion, shape preserved)
    #    (bsz, vocab)
    # ------------------------------------------------    
    speaker_posterior = speaker_prior + lambda_rsa * listener_score
    speaker_posterior = speaker_posterior - torch.logsumexp(
        speaker_posterior, dim=1, keepdim=True
    )

    # ------------------------------------------------
    # 7. Sampling constraints
    # ------------------------------------------------
    pert_logits = speaker_posterior.masked_fill(
        bad_words_mask, float("-inf")
    )

    pert_logits = top_k_top_p_filtering(
        pert_logits, top_k=top_k, top_p=top_p
    )

    pert_logits = torch.nan_to_num(
        pert_logits, nan=0.0, posinf=1e4, neginf=-1e4
    )

    rsa_probs = F.softmax(pert_logits, dim=-1)

    # ------------------------------------------------
    # 8. Optional: world posterior for analysis / update
    #    (bsz, vocab)
    # ------------------------------------------------
    new_world_priors = listener_posterior[:, :, 0]

    return rsa_probs, new_world_priors
           
def classifying_emotion(dec_sentence, model, tokenizer, emotion_classifier, device):
    temp = None
    respon_list=[]
    respon_keys=[]
    
    # bos = torch.tensor([tokenizer.bos_token_id], device=device, dtype=torch.long).unsqueeze(0)
    respon_split = re.split(r'([.!?])', dec_sentence)

    for i, respon in enumerate(respon_split):
        respon = respon.strip()
        if respon in string.punctuation:
            try:
                temp = temp + respon
            except:
                continue
            respon_list.append(temp)
            temp = None
        elif respon == '':
            continue
        elif (i+1) == len(respon_split):
            respon_list.append(respon)
        else:
            temp = respon
    
    for i, respon in enumerate(respon_list):
        # pert_response = tokenizer.encode(respon)
        # pert_response = torch.tensor(pert_response, device=device, dtype=torch.long).unsqueeze(0)
           
        # _, _, response_all_hidden = model(pert_response,
        #                                 output_hidden_states=True,
        #                                 return_dict=False) 
        
        inputs = tokenizer(
            respon,
            return_tensors='pt',
            padding=False,       # or True if batching
            truncation=False     # or True if you're limiting max length
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        response_all_hidden = model.get_hidden_states(input_ids, attention_mask)
        response_hidden = torch.mean(response_all_hidden,dim=1).float()

        response_pred = emotion_classifier(response_hidden)    
        class_pred = torch.argmax(response_pred).item() 
        
        emotiondict = DISCRIMINATOR_MODELS_PARAMS['Emotion_GO']['class_vocab']
        class_pred_key = list(emotiondict.keys())[list(emotiondict.values()).index(class_pred)]
        
        respon_keys.append(class_pred_key)
    
    return set(respon_keys)


def classifying_strategy(dec_sentence, model, tokenizer, strategy_classifier, device):
    temp = None
    respon_list=[]
    respon_keys=[]
    
    # bos = torch.tensor([tokenizer.bos_token_id], device=device, dtype=torch.long).unsqueeze(0)
    respon_split = re.split(r'([.!?])', dec_sentence)

    for i, respon in enumerate(respon_split):
        respon = respon.strip()
        if respon in string.punctuation:
            try:
                temp = temp + respon
            except:
                continue
            respon_list.append(temp)
            temp = None
        elif respon == '':
            continue
        elif (i+1) == len(respon_split):
            respon_list.append(respon)
        else:
            temp = respon
    
    for i, respon in enumerate(respon_list):
        with torch.no_grad():        
            inputs = tokenizer(
                respon,
                return_tensors='pt',
                padding=False,       # or True if batching
                truncation=False     # or True if you're limiting max length
            )
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            response_all_hidden = model.get_hidden_states(input_ids, attention_mask)
            response_hidden = torch.mean(response_all_hidden,dim=1).float()
    
            response_pred = strategy_classifier(response_hidden)    
            class_pred = torch.argmax(response_pred).item() 
            
            strategydict = DISCRIMINATOR_MODELS_PARAMS['Strategy']['class_vocab']
            class_pred_key = list(strategydict.keys())[list(strategydict.values()).index(class_pred)]
            
            respon_keys.append(class_pred_key)
    
    return set(respon_keys)

def _initialize_worldpriors_unigram(pertrub_num):
    """
    initialize the world prior with a uniform distribution
    """
    torch_dtype=torch.float
    
    ones = torch.ones(1, pertrub_num, dtype=torch_dtype, requires_grad=False).cuda()
    uniform_world_prior = torch.log(ones / pertrub_num)
    world_priors = uniform_world_prior.detach()

    return world_priors

def get_classifier(
        name: Optional[str],
        device: str,
        dtype: torch.dtype = torch.half  
    ) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device=device, dtype=dtype) 

    resolved_archive_file = params["path"]

    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device, weights_only=True))
    classifier.eval()

    return classifier

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

def compute_window_mask(
    past,
    window_length,
    decay=True,
    device="cuda",
    min_window=2,
    decay_type="linear",   # "linear", "exp", or "gaussian"
    eps=1e-6
):
    """
    Safer window mask for PPLM-style perturbation on LLaMA hidden states.

    Args:
        past: past_key_values list[ layers ][ (k or v) ][ batch, heads, length, dim ]
        window_length: number of most recent time steps to perturb.
                       If 0 → use adaptive window instead of full-length (safer).
        decay: whether to decay weights inside the window.
        min_window: minimum window length to avoid infinite accumulation.
        decay_type: type of decay: "linear", "exp", or "gaussian".
        eps: numerical stability constant.

    Returns:
        window_mask: same shape as past[0][0], values in [0,1].
    """

    # Extract length
    _, _, curr_length, _ = past[0][0].shape

    # ------------------------------------------------------------------
    # 1. Adaptive window length (CRITICAL FIX)
    # ------------------------------------------------------------------
    if window_length == 0:
        # Instead of "infinite accumulation",
        # use adaptive window proportional to sequence length.
        # Empirically best for LLaMA:
        window_length = max(min_window, min(6, curr_length))

    # If the sequence is shorter than the window → full mask
    if curr_length <= window_length:
        return torch.ones_like(past[0][0], device=device)

    # ------------------------------------------------------------------
    # 2. Build ones and zeros parts
    # ------------------------------------------------------------------
    ones_shape = past[0][0].shape[:-2] + (window_length,) + past[0][0].shape[-1:]
    zeros_shape = past[0][0].shape[:-2] + (curr_length - window_length,) + past[0][0].shape[-1:]

    ones = torch.ones(ones_shape, device=device)

    # ------------------------------------------------------------------
    # 3. Decay inside the window
    # ------------------------------------------------------------------
    if decay:
        t = torch.linspace(0, 1, window_length, device=device)

        if decay_type == "linear":
            dmask = t  # rises from 0 → 1

        elif decay_type == "exp":
            dmask = torch.exp(t * 3) / torch.exp(torch.tensor(3.0, device=device))

        elif decay_type == "gaussian":
            center = 1.0
            sigma = 0.35
            dmask = torch.exp(-0.5 * ((t - center) / sigma) ** 2)

        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")

        dmask = dmask.clamp(min=eps)
        ones = ones * dmask.view(1, 1, -1, 1)

    # ------------------------------------------------------------------
    # 4. Combine window and non-window segments
    # ------------------------------------------------------------------
    zeros = torch.zeros(zeros_shape, device=device)
    window_mask = torch.cat((ones, zeros), dim=-2)

    return window_mask

def add_func(L1: DynamicCache, L2: DynamicCache) -> DynamicCache:
    pkv1 = DynamicCache.from_legacy_cache(L1)
    pkv2 = DynamicCache.from_legacy_cache(L2)

    summed = [
        [ (m1 + m2) for m1, m2 in zip(layer1, layer2) ]
        for layer1, layer2 in zip(pkv1, pkv2)
    ]

    return DynamicCache.from_legacy_cache(summed)

def classifying_attribute(curr_unpert_past, curr_probs, curr_length, model, tokenizer, horizon_length, new_accumulated_hidden, classifier):
    delta = 0.7 # decay factor
    
    wte = model.resize_token_embeddings(len(tokenizer)) 
    for _ in range(horizon_length):
        inputs_embeds = torch.matmul(curr_probs, wte.weight.data)

        curr_output = model(past_key_values=curr_unpert_past,
                            inputs_embeds=inputs_embeds,
                            output_hidden_states=True
        )
        curr_unpert_past = curr_output.past_key_values
        curr_all_hidden = curr_output.hidden_states
        
        curr_hidden = curr_all_hidden[-1]
        
        predict_accumulated_hidden = delta * new_accumulated_hidden + torch.sum(
            curr_hidden, dim=1)
        predict_accumulated_hidden = predict_accumulated_hidden
    
    prediction = classifier(predict_accumulated_hidden /
                            (curr_length + 1 + horizon_length))
    
    return prediction
    

def perturb_hidden(
        past,
        model,
        last,
        context=None,
        encoder_attention_mask=None,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        strategy_prediction_model=None,
        target_strategy=None,
        strategy_classifier=None,
        nsp_classifier=None,
        emo_classifier=None,
        user_emotional_state=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        sample=True,
        gamma=1.5,
        emo_w=1,
        str_w=1,
        device='cuda',
        verbosity_level=REGULAR,
        tokenizer=None,
        strategy_tokenizer=None,
        context_t=None,
        last_response=None
):
    
    # Generate inital perturbed past
    grad_accumulator = [
        [
        (np.zeros(p.shape).astype("float32"))        
        for p in p_layer
        ]
        for p_layer in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    window_mask = compute_window_mask(past, window_length, decay, device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    
    log_logits_record = [] # For RSA
    
    iteration_stop = False
    
    if (loss_type == LossType.EMOTION or loss_type == LossType.ALL) and emo_w != 0:           
        # emotion pseudo-target            
        with torch.no_grad(): 
            context_output = model(context_t, output_hidden_states=True)
            context_all_hidden = context_output.hidden_states
    
            context_hidden = torch.mean(context_all_hidden[-1],dim=1)
            context_pred = emo_classifier(context_hidden)
    
    target_dtype = past[0][0].dtype
    
    for i in range(num_iterations):
        if iteration_stop:
            break
        
        if verbosity_level >= VERBOSE:
            print("\nIteration ", i + 1)
            
        curr_perturbation = [
            [
                torch.from_numpy(p_).type(target_dtype).to(device).requires_grad_()
                for p_ in p_layer
            ]
            for p_layer in grad_accumulator
        ]
        
        # Compute hidden using perturbed past
        perturbed_past = add_func(past, curr_perturbation) 
        
        _, _, curr_length, _ = curr_perturbation[0][0].shape   
        curr_output = model(last, past_key_values=perturbed_past,
                                          output_hidden_states=True)
        all_logits = curr_output.logits
        all_hidden = curr_output.hidden_states
        
        hidden = all_hidden[-1] #[1,1,1024]
        
        with torch.no_grad():
            new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1)      

        logits = all_logits[:, -1, :] #[1,1,50257]
        probs = F.softmax(logits, dim=-1)
        
        log_logits = F.log_softmax(logits, dim=-1)
        log_logits_record.append(log_logits.detach()) # for RSA Inference, Shared World
        
        if verbosity_level >= VERBOSE:
            if sample:
                next_token = torch.multinomial(probs.detach(), num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)           
            respon = torch.cat((last_response, next_token), dim=1) 
            print('respon(candidate):', tokenizer.decode(respon[0])) 

        loss = 0.0
        
        ### === calculating Kullback–Leibler Divergence loss === 
        kl_loss = 0.0
        
        KLD = nn.KLDivLoss(reduction="batchmean")
        target = F.softmax(unpert_logits[:, -1, :], dim=-1)
        kl_loss = KLD(log_logits, target)
        
        # # # weight kl_loss
        kl_loss = torch.mul(kl_loss, 0.01, out=None)
                    
        if verbosity_level >= VERY_VERBOSE:
            print('--------')
            print('kl_loss', kl_loss.data.cpu().numpy())
        
        # =========Attribute Models=============   
        curr_unpert_past = unpert_past
        curr_probs = torch.unsqueeze(probs, dim=1)
        
        ### ===engaging attribute model start=== 
        eng_loss = 0
        if loss_type == LossType.ENGAGEMENT: # TODO, not included for ALL fow now.
            #system   
            classification = classifying_attribute(curr_unpert_past,
                                                   curr_probs, curr_length,
                                                   model, tokenizer,
                                                   horizon_length, new_accumulated_hidden,
                                                   nsp_classifier)
    
            # user
            class_label = 0 # next sentence prediction postive target 
            label = torch.tensor(classification.shape[0] * [class_label],
                      device=device,
                      dtype=torch.long)
            
            ce_loss = torch.nn.CrossEntropyLoss()
            eng_loss = ce_loss(classification, label)

            # # # weight eng_loss
            # eng_loss = torch.mul(eng_loss, 2, out=None)
            
            if verbosity_level >= VERY_VERBOSE:
                print('--------')
                print('class_pred:{}'.format(torch.argmax(classification).item()))
                print(" pplm_eng_loss:", eng_loss.data.cpu().numpy())
                
        ### ===engaging attribute model end=== 
        
        ### ===strategy attribute model start=== 
        strategy_loss = 0
        if (loss_type == LossType.STRATEGY or loss_type == LossType.ALL) and target_strategy and str_w != 0:      

            strdict = DISCRIMINATOR_MODELS_PARAMS['Strategy']['class_vocab']
                                
            label_vector = [0.0] * len(strdict)
            for i, label_m in enumerate(strdict):
                for t_strat in target_strategy:
                    if SequenceMatcher(None, label_m, t_strat).ratio() > 0.7 :
                        label_vector[i] = target_strategy[t_strat]
                                       
            # strategy classification
            classification = classifying_attribute(curr_unpert_past,
                                               curr_probs, curr_length,
                                               model, tokenizer,
                                               horizon_length, new_accumulated_hidden,
                                               strategy_classifier)
            
            label = torch.tensor([label_vector], device=device, dtype=torch.float)
           
            bce_loss = torch.nn.BCEWithLogitsLoss()
            strategy_loss = bce_loss(classification, label)
            
            # # # weight strategy 
            strategy_loss = torch.mul(strategy_loss, str_w, out=None)

            if verbosity_level >= VERY_VERBOSE: 
                strategy_cls = list(strdict.keys())[list(strdict.values()).index(torch.argmax(classification).item())]
                print('\n---strategy attribute model info-----')
                print('str_w:', str_w)
                print('class_target:{},\n class_pred:{}\n'.format(target_strategy, strategy_cls))
                print("pplm_strategy_loss:", strategy_loss.data.cpu().numpy())

        ### ===strategy attribute model end=== 
        
        ### ===emotion attribute model start=== 
        emo_loss = 0
        if (loss_type == LossType.EMOTION or loss_type == LossType.ALL) and emo_w != 0:
            #system
            classification = classifying_attribute(curr_unpert_past,
                                               curr_probs, curr_length,
                                               model, tokenizer,
                                               horizon_length, new_accumulated_hidden,
                                               emo_classifier)
                        
            mse_loss = torch.nn.MSELoss()
            emo_loss = mse_loss(classification, context_pred)
             
            # # # weight emo_loss
            emo_loss = torch.mul(emo_loss, emo_w, out=None)
            
            if verbosity_level >= VERY_VERBOSE:
                emodict = DISCRIMINATOR_MODELS_PARAMS['Emotion_GO']['class_vocab']
                emo_target = list(emodict.keys())[list(emodict.values()).index(torch.argmax(context_pred).item())]
                emo_cls = list(emodict.keys())[list(emodict.values()).index(torch.argmax(classification).item())]  
                
                print('\n---emotion attribute model info-----')
                print('user_self-contagion:{}'.format(user_emotional_state))
                print('emo_w:{}'.format(emo_w))
                print('class_target:{}, class_pred:{}\n'.format(emo_target, emo_cls))
                print("pplm_emo_loss:", emo_loss.data.cpu().numpy())
                    
        # calculating total loss
        if loss_type == LossType.ENGAGEMENT:
            loss += eng_loss
        elif loss_type == LossType.STRATEGY:
            loss += strategy_loss
            loss += kl_loss
        elif loss_type == LossType.EMOTION:
            loss += emo_loss
            loss += kl_loss   
        elif loss_type == LossType.ALL:
            loss += strategy_loss
            loss += emo_loss
            loss += kl_loss
            # loss += eng_loss # TODO, not included for now.
        else:
            loss += kl_loss
                  
        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print('--------')
            print("total loss: ", loss.data.cpu().numpy())
            
        # === Early stopping if loss is too small (likely useless perturbation) ===
        SMALL_LOSS_THRESHOLD = 1e-6
        if loss.item() < SMALL_LOSS_THRESHOLD:
            if verbosity_level >= VERBOSE:
                print(f"🧊 Loss too small ({loss.item():.6f}) — skipping perturbation.")
            break  # or `return` if you're in a function
        
        # compute gradients
        loss.backward(retain_graph=True)

        # gradient checking
        # for p_layer in curr_perturbation:
        #     for p_ in p_layer:
        #         print(p_.grad , end=' ')
        #         break
        #     break
        # breakpoint()
        
        if grad_norms is not None:
            grad_norms = [
                    [
                    torch.max(grad, torch.norm(p_.grad * window_mask))
                    for grad, p_ in zip(grads, p_layer)
                    ]
                    for grads, p_layer in zip(grad_norms, curr_perturbation)
            ]
        else:       
            grad_norms = [
                    [
                    torch.norm(p_.grad * window_mask) + SMALL_CONST
                    for p_ in p_layer[:2]
                    ]
                    for p_layer in curr_perturbation
            ]
            
        # normalize gradients
        grad = [
                [
                -stepsize *
                (p_.grad * window_mask / grad ** gamma).data.cpu().numpy()
                for grad, p_ in zip(grads, p_layer[:2])
                ]
                for grads, p_layer in zip(grad_norms, curr_perturbation)
        ]
        
        # accumulate gradient
        grad_accumulator = add_func(grad, grad_accumulator)
        
        # reset gradients, just to make sure
        for p_layer in curr_perturbation:
            for p_ in p_layer[:2]:
                p_.grad.data.zero_()
                
        # removing past from the graph
        new_past = []
        for p_layer in past:
            new_past.append([])
            for p_ in p_layer:
                new_past[-1].append(p_.detach())
                
        past = new_past
    
    # apply the accumulated perturbations to the past 
    grad_accumulator = [
        [
            torch.from_numpy(p_).type(target_dtype).to(device).requires_grad_()
            for p_ in p_layer
        ]
        for p_layer in grad_accumulator
    ]
    
    pert_past = add_func(past, grad_accumulator)
    
    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter, log_logits_record


def full_text_generation(
        model,
        tokenizer,
        strategy_tokenizer=None,
        batch=None,
        max_length=100,
        min_length=10,
        num_samples=1,
        device="cuda",
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        sample=True,
        rsa=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        emo_w=1,
        str_w=1,
        verbosity_level=REGULAR,
        loss_type=None,
        strategy_prediction_model=None,
        strategy_classifier=None,
        nsp_classifier=None,
        emo_classifier=None,
        user_emotional_state=None,
        gold=False,
        joint=False,
        **kwargs
):
    
    
    if verbosity_level > REGULAR:
        # # # Generating the original responses without perturbation
        # unpert_gen_tok_text = user_prefix + original_response
        unpert_response, _, context = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            gold=gold,
            joint=joint,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            max_length=max_length,
            min_length=min_length,
            sample=sample,
            perturb=False, # without perturbation
            verbosity_level=verbosity_level
        )
    else:
        unpert_response = None
        context = None
    
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_responses = []
    losses_in_time = []
    
    # we first use last_response to perturb the current hidden, then use the perturbed hidden to generate the next word
    for i in range(num_samples):
        # use pert_sen_past to generate a complete sentence
        # here para_perturb = false, which means this function will use para_past = pert_sen_past  to generate a complete sentence without perturbation in word level
        # if para_perturb = true, this function will perturb in word level(original author design)
        
        if loss_type != 0 and (emo_w !=0 or str_w !=0):
            # # # Generating the responses with perturbation
            pert_response, loss_in_time, _ = generate_text_pplm(
                model=model,
                tokenizer=tokenizer,
                strategy_tokenizer=strategy_tokenizer,
                batch=batch,
                gold=gold,
                joint=joint,
                device=device,
                perturb=True, # with perturbation 
                strategy_prediction_model=strategy_prediction_model,
                strategy_classifier=strategy_classifier,
                nsp_classifier=nsp_classifier,
                emo_classifier=emo_classifier,
                user_emotional_state=user_emotional_state,
                loss_type=loss_type,
                max_length=max_length,
                min_length=min_length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample=sample,
                rsa=rsa,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                emo_w=emo_w,
                str_w=str_w,
                verbosity_level=verbosity_level,
                last_response=unpert_response
            )
        else:
            pert_response = unpert_response
            loss_in_time = []
            
        pert_responses.append(pert_response)
        losses_in_time.append(loss_in_time)

        # print('pert_gen_tok_text: {}'.format(pert_gen_tok_text))
        # print('pert_response: {}'.format(pert_response))
        
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    return context, unpert_response, pert_responses, losses_in_time

def generate_text_pplm(
        model,
        tokenizer,
        strategy_tokenizer=None,
        batch=None,
        gold=None,
        joint=None,
        past=None,
        device="cuda",
        perturb=True,
        strategy_prediction_model=None,
        strategy_classifier=None,
        nsp_classifier=None,
        emo_classifier=None,
        user_emotional_state=None,
        loss_type=0,
        max_length=100,
        min_length=10,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        sample=True,
        rsa=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        emo_w=1,
        str_w=1,
        verbosity_level=REGULAR,
        last_response=None,
):
    output_so_far = None
    system_response = None

    output_so_far = batch['input_ids']
    attention_mask = batch['attention_mask']

    context_t = batch['input_ids'].clone()
    context = tokenizer.decode(context_t[0]).split(tokenizer.eos_token)[-2].strip()
    context_t = torch.tensor(tokenizer.encode(context), device=device).unsqueeze(0)
        
    if len(tokenizer) > tokenizer.vocab_size:
        bad_words_ids = [i for i in range(tokenizer.vocab_size, len(tokenizer))]
        ones_mask = torch.ones(len(tokenizer)).to(device)
        ones_mask[bad_words_ids] = 0
        bad_words_mask = (ones_mask == 0)
       
    last = None
    grad_norms = None
    loss_in_time = []
    
    if verbosity_level >= VERBOSE:
        range_func = trange(max_length, ascii=True)
    else:
        range_func = range(max_length)
    
    if rsa:
        worldprior_initial = True
    
    outputs = model(output_so_far, attention_mask=attention_mask)
    
    # gold strategy 
    if gold:
        strat_id = batch['strat_id'].unsqueeze(-1) + len(tokenizer) - 8
        output_so_far = torch.cat([output_so_far, strat_id], dim=-1)
        last = strat_id
        
        # decode golden token
        golden_str = tokenizer.decode(strat_id[0])
        
        # --- assign soft weights ---
        w_gold = 1.0       # main gold strategy
        w_other1 = 0.5     # top-1 predicted strategy
        w_other2 = 0.5     # top-2 predicted strategy
        
        # start target_strategy dict
        target_strategy = {golden_str: w_gold}
        
        # --- get top-2 predicted strategies from model ---
        lm_logits = outputs['logits']
        encoded_info = {}
        model.predict_strategy(lm_logits, encoded_info)  # fills top predicted IDs
        
        if perturb:
            pred_strat_ids = encoded_info['pred_strat_id_top3'] + len(tokenizer) - 8
            # exclude golden_str
            for i, strat_id_pred in enumerate(pred_strat_ids[0]):
                s = tokenizer.decode([strat_id_pred])
                if s == golden_str:
                    continue
                if i == 0:
                    target_strategy[s] = w_other1
                elif i == 1:
                    target_strategy[s] = w_other2
                else:
                    break  # only add top 2
        
    # strategy prediction
    elif joint:
        lm_logits = outputs['logits']
        encoded_info = {}
        model.predict_strategy(lm_logits, encoded_info)  
        strat_id = encoded_info['pred_strat_id'].unsqueeze(-1) + len(tokenizer) - 8
        
        output_so_far = torch.cat([output_so_far, strat_id], dim=-1)
        last = strat_id
        
        # --- assign soft weights ---
        w_other1 = 1.0     # top-1 predicted strategy
        w_other2 = 1.0     # top-2 predicted strategy
        w_other3 = 1.0     # top-3 predicted strategy
        
        if (encoded_info['pred_strat_id_top1'] == batch['strat_id']).item():
            global strat_pred_acc
            strat_pred_acc += 1
        
        if perturb:
            target_strategy = {}
            pred_strat_ids = encoded_info['pred_strat_id_top3'] + len(tokenizer) - 8

            for i, strat_id_pred in enumerate(pred_strat_ids[0]):
                s = tokenizer.decode([strat_id_pred])
                # target_strategy[s] = 1.0  # full weight for predicted top strategies
                if i == 0:
                    target_strategy[s] = w_other1
                elif i == 1:
                    target_strategy[s] = w_other2
                else:
                    target_strategy[s] = w_other3
                
    # without strategy
    else:
        target_strategy = None
        last = output_so_far[:, -1:]
    
    for i in range_func:
        '''
        Get past/probs for current output, except for last word
        "past" are the precomputed key and value hidden states of the attention blocks
        Note that GPT takes 2 inputs: past + current_token
        '''

        # decoder_input_ids = torch.cat((torch.tensor([[tokenizer.bos_token_id]]).to(device), output_so_far), 1)
        # run model forward to obtain unperturbed past
        if past is None and output_so_far is not None:
            # last = decoder_input_ids[:, -1:]
            if output_so_far.shape[1] > 1:             
                output = model(output_so_far[:, :-1],
                                     output_hidden_states=True)
                past = output.past_key_values
                
        unpert_output = model(output_so_far, output_hidden_states=True)
        
        unpert_logits = unpert_output.logits
        unpert_past = unpert_output.past_key_values
        unpert_all_hidden = unpert_output.hidden_states
        
        unpert_last_hidden = unpert_all_hidden[-1]
        
        # check if we are above grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary (unperturb or perturb)
        if not perturb or num_iterations == 0:
            pert_past = past

        else:            
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            
            # shared world O initilization
            log_logits_record = torch.tensor([]) 
            
            if past is not None:
                pert_past, _, grad_norms, loss_this_iter, log_logits_record = perturb_hidden(
                    past,
                    model,
                    last,
                    encoder_attention_mask=attention_mask,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    strategy_prediction_model=strategy_prediction_model,
                    target_strategy=target_strategy,
                    strategy_classifier=strategy_classifier,
                    nsp_classifier=nsp_classifier,
                    emo_classifier=emo_classifier,
                    user_emotional_state=user_emotional_state,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    sample=sample,
                    gamma=gamma,
                    emo_w=emo_w,
                    str_w=str_w,
                    device=device,
                    verbosity_level=verbosity_level,
                    tokenizer=tokenizer,
                    strategy_tokenizer=strategy_tokenizer,
                    context_t=context_t,
                    last_response=last_response
                )

                log_logits_record = torch.cat(log_logits_record, 0) 
                
                loss_in_time.append(loss_this_iter)
                                
            else:
                pert_past = past
        
        # # # generating actual output token
        
        pert_output = model(last, past_key_values=pert_past, output_hidden_states=True)
        
        pert_logits = pert_output.logits
        past = pert_output.past_key_values

        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_logits_ = pert_logits.clone()
           
        # Fuse the modified model and original model
        if rsa and perturb:    
            ## S_0
            log_pert_probs = F.log_softmax(pert_logits, dim=-1) #[1,50257]
            log_unpert_probs = F.log_softmax(unpert_logits[:, -1, :], dim=-1)
            log_pert_probs = ((log_pert_probs * gm_scale) + (
                    log_unpert_probs * (1 - gm_scale)))  # + SMALL_CONST
                  
            log_score = torch.cat((log_pert_probs, log_logits_record.to(device)),0).unsqueeze(0) #S_0 [1,perturb_num,50257]
                      
            if worldprior_initial:
                worldpriors = _initialize_worldpriors_unigram(log_pert_probs.size(1))
                worldprior_initial = False  
                
            pert_probs, worldpriors = RSA_inference_llama(log_score, worldpriors, bad_words_mask, top_k, top_p)  
            # pert_probs, worldpriors = RSA_inference(log_score, worldpriors, bad_words_mask, top_k, top_p)    

                
        elif perturb:
            pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
            pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
            pert_probs = F.softmax(pert_logits, dim=-1)
            
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST   
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST
            
            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)
                
                       
        else:
            pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
            pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
            pert_probs = F.softmax(pert_logits, dim=-1)


        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)
                         
        if last.tolist()[0][0] == tokenizer.eos_token_id:
            if output_so_far.size(1)-2 <= min_length:
                pert_logits_[:, tokenizer.eos_token_id] = -float("inf")
                pert_logits = pert_logits_.masked_fill(bad_words_mask, float("-inf"))
                pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
                pert_probs = F.softmax(pert_logits, dim=-1) 
                
                if sample:
                    last = torch.multinomial(pert_probs, num_samples=1)
                else:
                    _, last = torch.topk(pert_probs, k=1, dim=-1)
            else:    
                # ***avoid system_response = None***
                if system_response is None:
                    system_response = context_t
                break

        if last.tolist()[0][0] <= len(tokenizer):
            #update system_response
            output_so_far = (
                last if output_so_far is None
                else torch.cat((output_so_far, last), dim=1)
            )
            #update system_response
            system_response = (
                last if system_response is None
                else torch.cat((system_response, last), dim=1)
            )
        else:
            print(last.tolist()[0][0])
            name = input('pause of word_id out of 50256: ')
            print('continue: ', name)
            break
        
        last_response = system_response
        if verbosity_level > REGULAR:
            decode_response = tokenizer.decode(system_response.tolist()[0])
            print('system_response(perturbed)--------------:')
            print(decode_response)
            print()
        

    return system_response, loss_in_time, context


def run_pplm_example(
        config_name=None,
        inputter_name=None,
        seed=42,
        load_checkpoint=None,
        fp16=False,
        max_input_length=256, 
        max_src_turn=None,
        max_decoder_input_length=50,
        max_knowledge_length=None,
        label_num=None,
        multi_knl=None,
        only_encode=None,
        only_generate=None,
        chinese=None,
        add_nlg_eval=None,
        infer_batch_size=1,
        infer_input_file=None,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        use_gpu=False,
        max_length=100,
        min_length=10,
        num_samples=1,
        stepsize=0.02,
        sample=True,
        rsa=False,
        page=False,
        gold=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        emo_weight=0.25,
        str_weight=1,
        no_cuda=False,
        verbosity='regular',
        out_dir=None,
        for_test_run=False,
        valid=False,
        attribute_type=None,
        joint=False,
        interact=False
):

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
    
    logging.info('initializing cuda...')
    _ = torch.tensor([1.], device=device)
    
    # set logger
    logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
    }
    
    # load pretrained model
    logger.info('Loading checkpoint: {}\n'.format(load_checkpoint))
    tokenizer, model = build_model(checkpoint=args.load_checkpoint, **names)
    model = deploy_model(model, args)
    
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Number of parameter = {}'.format(total_params))
    
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, opt_level="O1")
    
    model.eval()
    model.to(device)
    
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    
    if page:
        # load PAGE
        PAGE = torch.load('../PAGE/saves_final/model_0.pkl')
        PAGE.eval()
        PAGE.to(device)
        
        # Freeze PAGE
        for param in PAGE.parameters():
            param.requires_grad = False
        
        # load tokenizer for page
        tokenizer_page = AutoTokenizer.from_pretrained("roberta-base")

    inputter = inputters[args.inputter_name]()
    dataloader_kwargs = {
        'max_src_turn': args.max_src_turn,
        'max_input_length': args.max_input_length,
        'max_decoder_input_length': args.max_decoder_input_length,
        'max_knowledge_length': args.max_knowledge_length,
        'label_num': args.label_num,
        'multi_knl': args.multi_knl,
        'only_encode': args.only_encode,
        'infer_batch_size': args.infer_batch_size,
    }
        
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
    
    tokenizer.add_bos_token = True
             
    # # attribute model control
    emo_attribute_flag = False
    str_attribute_flag = False
    nsp_attribute_flag = False
    
    strategy_classifier = None
    strategy_tokenizer = None
    strategy_prediction_model = None
    nsp_classifier = None
    emo_classifier = None
    
    attribute_type_map = {
        "none": LossType.NONE,
        "engagement": LossType.ENGAGEMENT,
        "strategy": LossType.STRATEGY,
        "emotion": LossType.EMOTION,
        "all": LossType.ALL
    }
    loss_type = attribute_type_map.get(args.attribute_type.lower(), LossType.NONE)
    
    if attribute_type == 'all':
        print("***** All atribute model activated *****")
        emo_attribute_flag = True
        str_attribute_flag = True
    elif attribute_type == 'engagemnt': # TODO, not used for now.   
        print("***** Engagement atribute model activated *****")
        nsp_attribute_flag = True       
    elif attribute_type == 'strategy': 
        print("***** Empathetic strategy atribute model activated *****")
        str_attribute_flag = True
    elif attribute_type == 'emotion':
        print("***** Emotion atribute model activated *****")
        emo_attribute_flag = True
    else:
        print("***** No atribute model *****")
        num_iterations = 0
        emo_weight = 0
        str_weight = 0

    if rsa:
        print("***** Rational Speech Act activated *****")
        
    if page:
        print("***** Emotion Dynamic activated *****")
        
    if emo_attribute_flag:
        emo_classifier = get_classifier(
            'Emotion_GO',
            device
        )
    
    if str_attribute_flag:
        strategy_classifier = get_classifier(
            'Strategy',
            device
        )
    
    if nsp_attribute_flag:
        nsp_classifier = get_classifier(
            'Engagement',
            device
        )
        
    
    for infer_idx, infer_input_file in enumerate(args.infer_input_file):
        set_seed(args.seed)
        infer_dataloader = inputter.infer_dataloader(
            infer_input_file,
            tokenizer,
            **dataloader_kwargs
        )
    
    # Set output path 
    logger.info("Output dir: {}".format(out_dir))
    
    global file_log
    if rsa and page:
        out_name = '/DPPRSA_llama_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight)
    elif rsa:
        out_name = '/PPRSA_llama_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight)
    elif page:
        out_name = '/DPPLM_llama_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight)
    elif attribute_type == 'all':
        out_name = '/PPLM_llama_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight) 
    elif attribute_type == 'emotion':
        out_name = '/PPLM_llama_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight)
    elif attribute_type == 'strategy':
        out_name = '/PPLM_llama_' + attribute_type + '_x' + str(num_iterations) + '_sw' + str(str_weight)
    else:
        out_name = '/Llama-Joint-pplm'
        num_iterations = 0
    
    if gold:
        out_name += '_g' 
        joint = True
    elif joint:
        out_name += '_j'
        
    if valid:
        out_dir = 'valid'
        if not sample:
            out_dir += '/greedy/'
    
    if for_test_run:
        verbosity_level = VERY_VERBOSE
    elif out_dir != None:
        if not os.path.exists('output/{}'.format(out_dir)):
            logger.info("Create dir: {}".format(out_dir))
            os.makedirs('output/{}'.format(out_dir))
        
        file_pert = open('output/' + out_dir + '/' + out_name + '.txt', 'w+', encoding='utf-8')
        file_log = open('output/' + out_dir + '/' + out_name + '_log.txt', 'w+', encoding='utf-8')
    else:
        file_pert = open('output/' + out_name + '.txt', 'w+', encoding='utf-8')
        file_log = open('output/' + out_name + '_log.txt', 'w+', encoding='utf-8')
        
    # # # === begin time ====
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    sentence_count = 0
    neutral_count = 0
    self_count = 0
    inter_count = 0
        
    for batch, posts, references, sample_ids, histories, speaker_roles in infer_dataloader:   
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        
        if page:
            user_emotional_state, emo_w, str_w = get_emotional_state(histories, tokenizer_page,
                                                                     PAGE, speaker_roles, emo_weight, str_weight, device)
    
            if user_emotional_state == 'neutral':
                neutral_count += 1
                emo_w = emo_weight
                str_w = str_weight
            elif user_emotional_state == 'self-contagion':
                self_count += 1
            elif user_emotional_state == 'inter-personal':
                inter_count += 1

        else:
            user_emotional_state = 'neutral'
            emo_w = emo_weight 
            str_w = str_weight
         
        logging.disable(logging.WARNING)
         
        sentence_count += 1
        if sentence_count % 500 == 0 or sentence_count == 1:
            print("===" + str(sentence_count) + "===")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            if verbosity_level >= REGULAR:
                print("= Prefix of sentence =")
                print(posts)
                print()

        # generate unperturbed and perturbed texts 
        context, unpert_response, pert_responses, losses_in_time = full_text_generation(
            model=model,
            tokenizer=tokenizer,
            strategy_tokenizer=strategy_tokenizer,
            batch=batch,
            device=device,
            max_length=max_length,
            min_length=min_length,
            num_samples=num_samples,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample=sample,
            rsa=rsa,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            emo_w=emo_w,
            str_w=str_w,
            verbosity_level=verbosity_level,
            loss_type=loss_type,
            strategy_prediction_model=strategy_prediction_model,
            strategy_classifier=strategy_classifier,
            nsp_classifier=nsp_classifier,
            gold=gold,
            joint=joint,
            emo_classifier=emo_classifier,
            user_emotional_state=user_emotional_state,
        )

        decode = lambda x: _norm(tokenizer.decode(x))
        dec_sentence = decode(pert_responses[0][0])
        dec_sentence = dec_sentence.strip()
        dec_sentence = postprocess_response(dec_sentence)
        
        # untokenize unperturbed text 
        if verbosity_level > REGULAR or for_test_run:
            print(context)
            print("=" * 80)
            print("= Unperturbed generated text =")
            unpert_gen_text = decode(unpert_response.tolist()[0])
            unpert_gen_text = postprocess_response(unpert_gen_text)
            print(unpert_gen_text)
            print()

            for i, pert_res in enumerate(pert_responses):
                print("= Perturbed response {} =".format(i))
                pert_res_text = decode(pert_res.tolist()[0])
                pert_res_text = postprocess_response(pert_res_text)
                print(pert_res_text)
                print()                
            
            if page:
                print('\n= User Emotional State =')
                print(user_emotional_state)
            
            if emo_classifier is not None:               
                # og emotion
                unpert_respon_keys = classifying_emotion(unpert_gen_text, model, tokenizer, emo_classifier, device)
                # respon strategy
                respon_keys = classifying_emotion(dec_sentence, model, tokenizer, emo_classifier, device)
                print('\n= Emotion Prediction Result =')
                print('response_class:{} unpert_class:{}'.format(respon_keys, unpert_respon_keys))
            
            if strategy_classifier is not None:               
                # og strategy
                unpert_respon_keys = classifying_strategy(unpert_gen_text, model, tokenizer, strategy_classifier, device)
                # respon strategy
                respon_keys = classifying_strategy(dec_sentence, model, tokenizer, strategy_classifier, device)

                print('\n= Strategy Prediction Result =')
                print('gold_class:', tokenizer.decode(batch['strat_id'] + len(tokenizer) - 8))
                print('response_class:{} unpert_class:{}'.format(respon_keys, unpert_respon_keys))
                print()
            
            breakpoint()
                
        if not joint:
            dec_sentence = dec_sentence.split(']', 1)[-1].strip()
        if not for_test_run:
            file_pert.write(dec_sentence + '\n')
                    
    # # # === finish time ===
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    file_log.write("begin time: " + begin_time + "\t")
    file_log.write("finish time: " + finish_time + "\n")
    print(begin_time)
    print(finish_time)

    struct_time = time.strptime(begin_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_begin = int(time.mktime(struct_time)) # 轉成時間戳

    struct_time = time.strptime(finish_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_finish = int(time.mktime(struct_time)) # 轉成時間戳
    
    total_time = time_stamp_finish - time_stamp_begin
    
    if page:
        file_log.write('neutral:' + str(neutral_count)+' \n')
        file_log.write('self-contagion:' + str(self_count)+' \n')
        file_log.write('inter-personal:' + str(inter_count)+' \n')
    
    file_log.write('inter-total time(second):' + str(total_time) +' \n')
    file_log.write('strategy prediction accurracy:' + str(round(strat_pred_acc/sentence_count,2)) +' \n')

    print("total time(second): ", total_time)
    print('strategy prediction accurracy: ', round(strat_pred_acc/sentence_count,2))

    file_pert.close()
    file_log.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--inputter_name', type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_checkpoint", '-c', type=str, default=None)
    parser.add_argument("--fp16", type=boolean_string, default=False)
    
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_src_turn", type=int, default=None)
    parser.add_argument("--max_decoder_input_length", type=int, default=256)
    parser.add_argument("--max_knowledge_length", type=int, default=None)
    parser.add_argument('--label_num', type=int, default=None)
    parser.add_argument('--multi_knl', action='store_true', help='allow candidate knowledge items')
    
    parser.add_argument('--only_encode', action='store_true', help='only do encoding')
    parser.add_argument('--only_generate', action='store_true', help='do not conduct evaluations')
    parser.add_argument('--chinese', action='store_true', help='chinese language')
    parser.add_argument('--add_nlg_eval', action='store_true', help='add nlg-eval')
    
    parser.add_argument("--infer_batch_size", type=int, default=1)
    parser.add_argument('--infer_input_file', type=str, nargs='+', required=True)
    
    parser.add_argument("--use_gpu", action='store_true')
    
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=60)
    parser.add_argument("--num_samples", type=int, default=1)
    
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument("--sample", action="store_true")
    
    parser.add_argument("--attribute_type", type=str, default='None')
    parser.add_argument('--joint', action="store_true", help="Include strategy for generation")
    parser.add_argument(
        "--rsa", action="store_true",
        help="Activate Rational Speech Act for generation"
    )
    parser.add_argument(
        "--page", action="store_true",
        help="Activate PAGE for generation"
    )
    parser.add_argument(
        "--gold", action="store_true",
        help="Activate gold_strategy for generation"
    )
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    
    parser.add_argument("--emo_weight", type=float, default=1)
    parser.add_argument("--str_weight", type=float, default=1)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--for_test_run", action="store_true")
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--interact", action="store_true")
    
    args = parser.parse_args()
    
    run_pplm_example(**vars(args))
