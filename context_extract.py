# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:20:00 2025

@author: OWNER
"""
from inputters import inputters
from utils.building_utils import boolean_string, build_model, deploy_model

dataloader_kwargs = {
    'max_src_turn': None,
    'max_input_length': 256,
    'max_decoder_input_length': 256,
    'max_knowledge_length': None,
    'label_num': None,
    'multi_knl': None,
    'only_encode': False,
    'infer_batch_size': 1,
}

infer_input_file = './_reformat/SUPPORTER/test.txt'
inputter_name = 'strat_pp'
config_name = 'strat'
load_checkpoint = 'DATA/strat_pp.strat/2024-04-21230454.3e-05.16.1gpu/epoch-2.bin'

inputter = inputters[inputter_name]()

names = {
    'inputter_name': inputter_name,
    'config_name': config_name,
}

toker, model = build_model(checkpoint=load_checkpoint, **names)

infer_dataloader = inputter.infer_dataloader(
    infer_input_file,
    toker,
    **dataloader_kwargs
)
    
all_posts = []

for batch, posts, references, sample_ids, histories, speaker_roles in infer_dataloader:
    for post, ids in zip(posts, sample_ids):
        # Remove prefix like "[Providing Suggestions]" and keep only the message
        if ']' in post:
            post = post.split(']', 1)[-1].strip()
        all_posts.append(str(ids) + ' ' + post)
    
# Save all processed posts to a text file, one per line
with open("CONTEXT_dialog_id.txt", "w", encoding="utf-8") as f:
    for post in all_posts:
        f.write(post + "\n")