# Dynamic-Plug-and-Play-Rational-Speech-Act-Framework

<img width="4044" height="2296" alt="figure1 2" src="https://github.com/user-attachments/assets/549214ed-169a-4137-94ec-627bc9c557a7" />

This is the PyTorch implementation of our work: **A Dynamic Plug-and-Play Rational Speech Act Framework for Emotional Support Dialogue System**. It is a integrated system incoperated two other papers:

- **PPRSA: A Plug-and-Play Language Model with Rational Speech Act Inference for Generating Empathetic and Engaging Dialogue Responses**, IEEE Transactions on Audio, Speech and Language Processing, 2025.
- **Applying Emotion-Cause Entailment for Help-Seeker Guidance in Emotional Support Conversations**, IEEE Transactions on Computational Social Systems, 2025.

The generated responses are located in the "generated_responses_txt" file.

## Abstract
Traditional supervised fine-tuning or pipeline frameworks, while effective for overall language quality, provide limited control over nuanced response attributes. Moreover, fine-tuning LLMs often requires large-scale retraining for each new attribute or domain, creating a trade-off between adaptability and computational efficiency—especially in scenarios that demand real-time interaction like emotional support conversation. To address these issues, an integrated framework called **Dynamic Plug-and-Play Rational Speech Acts (DPPRSA)** was developed. This framework integrates two core ideas: (1) a dynamic attribute gating mechanism informed by Emotion Cause Entailment (ECE), which controls when and how different attributes are activated to improve coordination among multiple attributes, and (2) a Rational Speech Act (RSA) inference is employed to improve the efficiency of the iterative process, leading to faster response generation (7.07 seconds per response on average) and improved response quality. This unified design enables the dialogue system to dynamically adapt to users' needs by generating responses with varying attribute weightings. In other words, it leverages the Plug-and-Play nature of the framework, plugging in only the proper attributes for each response, even when multiple attributes are available. 

## Implementation

### System Requirements
- Python 3.8
- Pytorch 2.0.1
- See environment.yml for details

### Environment setup
```
conda create -n DPPRSA python=3.8 -f environment.yml
conda activate DPPRSA
```

### Preprocess Training Data
1.Download the dataset from SUPPORTER: https://github.com/jfzhouyoo/Supporter/tree/master
2.Run the code to convert the json file to txt: /GenerationModel\_reformat\SUPPORTER/Json2txt.ipynb
3.cd /GenerationModel/
4.bash RUN/prepare_strat_llama.sh, the dataset will be placed in /GenerationModel/DATA/strat_llama.strat_llama

*Change the directories in the Json2txt.ipynb for your environment.
*The configuration of the model and the extension of special tokens is in /GenerationModel/CONFIG/strat_llama.json.
*The preprocessing of the dataset is in /GenerationModel/inputters/strat_llama.py

### Download Model (BlenderBot-small for Example)
1.Download the BlenderBot-small model: https://huggingface.co/facebook/blenderbot_small-90M
2.Place the model in the same directory.

You could use other models,but you need to maually change the configurations.

### Training The Generation Model (BlenderBot)
1.Train the model with strategy special tokens. The checkpoint will be saved in /GenerationModel/DATA/strat_pp.strat.
'''
bash RUN/train_strat.sh
'''
2.Change the file name of the checkpoint to your preference.

The generation model can be infered with: bash RUN/infer_strat.sh.
You can change GOLDEN_TRUTH in inputters/PARAMS.py to control whether use the golden strategy.

3.Infer with the generation model.
'''
bash RUN/infer_strat.sh
'''

### Training The Generation Model (Llama)
1.Download the model from: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2.Train the model with strategy special tokens. The checkpoint will be saved in /GenerationModel/DATA/strat_llama.strat_llama
'''
bash RUN/train_strat_llama.sh
'''
3.change the file name of the checkpoint to your preference.

The generation model can be infered with: bash RUN/infer_strat_llama.sh.
You can change GOLDEN_TRUTH in inputters/PARAMS.py to control whether use the golden strategy.

4.Infer with the generation model.
'''
bash RUN/infer_strat_llama.sh
'''

### Training The Empathy Attrubute Model
1.Go to the EmotionClassifier_head directory.
'''
cd /EmotionClassifier_head/
'''
2.Download the GoEmotion dataset: https://github.com/google-research/google-research/tree/master/goemotions/data
3.Run the code to preprocess the dataset:GoEmotion_preprocess.ipynb
4.Train the classifier.
'''
bash train_emo_classifier.sh
'''

*the attribute models are basically using generation model as encoder with classification head, you have to train the generation model first.
*change the settings in "train_emo_classifier.sh" based on your generation model.
*the trained attribute model's setting can be found in /output_llama/GO_classifier_head_meta.json
*the class_vocab dictionary is always different after you trained.

## Reference
```
@article{chang2025pprsa,
  title={PPRSA: A Plug-and-Play Language Model with Rational Speech Act Inference for Generating Empathetic and Engaging Dialogue Responses},
  author={Chang, Jeremy and Wu, Chung-Hsien},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2025},
  publisher={IEEE}
}
```
```
@article{chang2025applying,
  title={Applying Emotion-Cause Entailment for Help-Seeker Guidance in Emotional Support Conversations},
  author={Chang, Jeremy and Wu, Chung-Hsien},
  journal={IEEE Transactions on Computational Social Systems},
  year={2025},
  publisher={IEEE}
}
```
