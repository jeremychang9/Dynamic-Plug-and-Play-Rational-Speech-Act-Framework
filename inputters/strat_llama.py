# coding=utf-8

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH
import torch.nn.functional as F
import re


class Inputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features
        
        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader
        
        # valid
        self.valid_dataloader = DynamicBatchingLoader
        
        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
        self,
        input_ids,
        decoder_input_ids, labels,
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        
        if decoder_input_ids is not None:
            self.decoder_input_ids = decoder_input_ids
            self.decoder_input_length = len(decoder_input_ids)
        else:
            self.decoder_input_ids = None
            self.decoder_input_length = None
            
        self.labels = labels

        if self.decoder_input_length is not None:
            self.input_len = self.input_length + self.decoder_input_length
        else:
            self.input_len = self.input_length


def featurize(
    bos, eos,
    context, max_input_length,
    response, max_decoder_input_length,
    strat_id,
    infer=False,
):
        
    # Add strategy tokens and response
    response_ids = response + [eos] 
    
    max_response_len = max_input_length // 2
    if len(response_ids) > max_response_len:
        response_ids = response_ids[:max_response_len]
        
    max_context_length = max_input_length - len(response_ids) - 1  # -1 for bos
    max_context_length = max(0, max_context_length)  # ensure non-negative
         
    context_eos = [utterance + [eos] for utterance in context] 
    
    # Pop oldest utterances until the context fits
    total_len = sum(len(utt) for utt in context_eos)
    while context_eos and total_len > max_context_length:        
        if total_len - len(context_eos[0]) <= 0:
            over = total_len - max_context_length
            context_eos[0] = context_eos[0][over:]
            break
        else:  
            total_len -= len(context_eos[0])
            context_eos.pop(0)    
    
    flattened_context = [token for utt in context_eos for token in utt] 
    
    input_ids = [bos] + flattened_context + (response_ids if not infer else response_ids[0:3])
    
    labels = [-100] * (1 + len(flattened_context)) + response_ids

    decoder_input_ids = response_ids
    
    # if not infer:
    #     labels = [-100] * (1 + len(flattened_context)) + response_ids
    #     decoder_input_ids = response_ids
    # else:
    #     labels = None
    #     decoder_input_ids = None  # or decoder_input_ids = [strat_id] if needed for generation prompt
    
    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,
    )


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):  
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    
    dialog = data['dialog']
    inputs = []
    context = []
    speaker_role = []
    
    role_user = '<|start_header_id|>' + 'user' + '<|end_header_id|>'
    role_system = '<|start_header_id|>' + 'system' + '<|end_header_id|>'
    
    usr_flag = False
    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        
        speaker_role.append(dialog[i]['speaker'])
        
        if dialog[i]['speaker'] == 'usr':
            usr_flag = True
            text = role_user + text
        
        if dialog[i]['speaker'] == 'sys':
            strat_token = '[' + dialog[i]['strategy'] + ']'
            text = role_system + strat_token + text
            
        text = process(text)
        
        # get only the responses after user have spoken once before
        if i > 0 and dialog[i]['speaker'] == 'sys' and usr_flag: 
            res = {
                'context': context.copy(), # dialogue history list
                'response': text,
                'strat_id': process(strat_token),
                'speaker_role':speaker_role[-6:-1] # last 5 context speaker sequence
            }
            
            inputs.append(res)
            
        context = context + [text]

    return inputs


def convert_inputs_to_features(inputs, toker, infer=False, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')
    
    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
    
    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['context'], max_input_length,
            ipt['response'], max_decoder_input_length, ipt['strat_id'], infer
        )
        features.append(feat)

    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id or toker.eos_token_id
        bos = toker.bos_token_id or toker.cls_token_id
        eos = toker.eos_token_id or toker.sep_token_id
        
        assert pad is not None, "pad_token_id or eos_token_id must be set"
        assert bos is not None, "bos_token_id or cls_token_id must be set"
        assert eos is not None, "eos_token_id or sep_token_id must be set"
        
        input_tensors = [torch.tensor(f.input_ids, dtype=torch.long) for f in features]
        input_lengths = [f.input_length for f in features]
        
        if infer:
            # Left padding for inference
            max_len = max(input_lengths)
            input_ids = torch.stack([
                F.pad(t, (max_len - len(t), 0), value=pad) for t in input_tensors
            ])
            attention_mask = torch.stack([
                F.pad(torch.ones(len(t), dtype=torch.long), (max_len - len(t), 0), value=0)
                for t in input_tensors
            ])
        else:
            # Right padding for training/validation
            input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=pad)
            attention_mask = pad_sequence(
                [torch.ones(len(t), dtype=torch.long) for t in input_tensors],
                batch_first=True,
                padding_value=0,
            )
        
        # input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)
        
        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                              batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                              batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = None
            labels = None
        
        strat_id = torch.tensor([f.decoder_input_ids[3] for f in features], dtype=torch.long) - len(toker) + 8
        # strategy token located at response_id[3]
       
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'strat_id': strat_id,
        }
        
        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """
    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()
            
            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []
                    
            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch
                
        except StopIteration:
            pass
    
    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


# for inference
def prepare_infer_batch(features, toker, interact=None):
    res = FeatureDataset.collate(features, toker, True)
    res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_strat_id': 'pred_strat_id',
    }

    if interact is None and GOLDEN_TRUTH:
        other_res['cls_strat_id'] = res.get('strat_id')
    else:
        other_res['cls_strat_id'] = res.pop('strat_id')

    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()
    
    features = []
    sample_ids = []
    posts = []
    references = []
    histories = []
    speaker_roles = []
    
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, infer=True, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(toker.decode(ipt['context'][-1]))
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)
            for h in ipt['context'][-5:]:
                decoded = toker.decode(h, skip_special_tokens=True)
                decoded = re.sub(r"(system|user|assistant)\s*", "", decoded, count=1)
                histories.append(decoded)
            speaker_roles.append(ipt['speaker_role'])
            
            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids, histories, speaker_roles
                features = []
                sample_ids = []
                posts = []
                references = []
                histories = []
                speaker_roles = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids, histories, speaker_roles
        
def get_infer_batch_from_raw(raw_texts, toker, **kwargs):
    """
    raw_texts: list of str
    Returns a generator that yields batches similar to get_infer_batch,
    but with minimal fields (no references, no true speaker_roles).
    """
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    features = []
    sample_ids = []
    posts = []
    references = []
    histories = []
    speaker_roles = []

    for sample_id, raw_text in enumerate(raw_texts):
        # Wrap raw text as context-only input
        inputs = [{
            "context": [toker.encode(raw_text)],
            "response": [],  # no gold response
            "speaker_role": "user"  # default role since not provided
        }]
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)

        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(raw_text)              # raw input text
            references.append("")               # no reference
            sample_ids.append(sample_id)
            histories.append(raw_text)          # treat input as history
            speaker_roles.append("user")        # assign manually

            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids, histories, speaker_roles
                features, sample_ids, posts, references, histories, speaker_roles = [], [], [], [], [], []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids, histories, speaker_roles

