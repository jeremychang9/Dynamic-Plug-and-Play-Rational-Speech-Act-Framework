# A Dynamic Plug-and-Play Rational Speech Act Framework for Emotional Support Dialogue System

<img width="4044" height="2296" alt="figure1 2" src="https://github.com/user-attachments/assets/549214ed-169a-4137-94ec-627bc9c557a7" />

This is the PyTorch implementation of our work: **A Dynamic Plug-and-Play Rational Speech Act Framework for Emotional Support Dialogue System**. It is a integrated system incoperated two other papers:

- **PPRSA: A Plug-and-Play Language Model with Rational Speech Act Inference for Generating Empathetic and Engaging Dialogue Responses**, IEEE Transactions on Audio, Speech and Language Processing, 2025.
- **Applying Emotion-Cause Entailment for Help-Seeker Guidance in Emotional Support Conversations**, IEEE Transactions on Computational Social Systems, 2025.

The generated responses are located in the "output" file.

The codes are highly inspired by: 
* https://github.com/thu-coai/Emotional-Support-Conversation/tree/main/codes_zcj
* https://github.com/skywalker023/focused-empathy

## Abstract
Traditional supervised fine-tuning or pipeline frameworks, while effective for overall language quality, provide limited control over nuanced response attributes. Moreover, fine-tuning LLMs often requires large-scale retraining for each new attribute or domain, creating a trade-off between adaptability and computational efficiency—especially in scenarios that demand real-time interaction like emotional support conversation. To address these issues, an integrated framework called **Dynamic Plug-and-Play Rational Speech Acts (DPPRSA)** was developed. This framework integrates two core ideas: (1) a dynamic attribute gating mechanism informed by Emotion Cause Entailment (ECE), which controls when and how different attributes are activated to improve coordination among multiple attributes, and (2) a Rational Speech Act (RSA) inference is employed to improve the efficiency of the iterative process, leading to faster response generation (7.07 seconds per response on average) and improved response quality. This unified design enables the dialogue system to dynamically adapt to users' needs by generating responses with varying attribute weightings. In other words, it leverages the Plug-and-Play nature of the framework, plugging in only the proper attributes for each response, even when multiple attributes are available. 

## Implementation

<img width="1887" height="1023" alt="DPPRSA_overview" src="https://github.com/user-attachments/assets/ee17768c-6512-4f02-a3a5-cd6a3bf540d4" />

Our current implementation supports only **BlenderBot-small** and **Llama-3.2-1B-Instruct**. While code for other models is preserved, it remains untested and unmaintained. You will need to handle the integration yourself if you wish to implement other models, such as DialoGPT and BlenderBot-base.

### System Requirements
- Python 3.8
- Pytorch 2.0.1
- See environment.yml for details

### Environment setup
```
conda create -n DPPRSA python=3.8 -f environment.yml
conda activate DPPRSA
```
### Preprocessing Training Data (BlenderBot)
1. Download the dataset from SUPPORTER: https://github.com/jfzhouyoo/Supporter/tree/master
2. Run the code to convert the json file to txt: /GenerationModel\_reformat\SUPPORTER/Json2txt.ipynb
3. Go to the GenerationModel directory.
```
cd /GenerationModel/
```
4. Preprocss the data, it will be placed in /GenerationModel/DATA/strat_pp.strat
```
bash RUN/prepare_strat_blenderbot.sh
```

* Change the directories in the Json2txt.ipynb for your environment.
* The configuration of the model and the extension of special tokens is in /GenerationModel/CONFIG/strat.json.
* The preprocessing of the dataset is in /GenerationModel/inputters/strat_pp.py

### Preprocessing Training Data (Llama)
1. Download the dataset from SUPPORTER: https://github.com/jfzhouyoo/Supporter/tree/master
2. Run the code to convert the json file to txt: /GenerationModel\_reformat\SUPPORTER/Json2txt.ipynb
3. Go to the GenerationModel directory.
```
cd /GenerationModel/
```
4. Preprocss the data, it will be placed in /GenerationModel/DATA/strat_llama.strat_llama
```
bash RUN/prepare_strat_llama.sh
```

* Change the directories in the Json2txt.ipynb for your environment.
* The configuration of the model and the extension of special tokens is in /GenerationModel/CONFIG/strat_llama.json.
* The preprocessing of the dataset is in /GenerationModel/inputters/strat_llama.py

### Downloading Model (BlenderBot-small for Example)
1. Download the BlenderBot-small model: https://huggingface.co/facebook/blenderbot_small-90M
2. Place the model in the same directory.

* You could use other models, but you need to maually change the configurations.

### Training The Generation Model (BlenderBot)
1. Train the model with strategy special tokens. The checkpoint will be saved in /GenerationModel/DATA/strat_pp.strat.
```
bash RUN/train_strat.sh
```
2. Change the file name of the checkpoint to your preference.

The generation model can be infered with: bash RUN/infer_strat.sh.
You can change GOLDEN_TRUTH in inputters/PARAMS.py to control whether use the golden strategy.

3. Infer with the generation model.
```
bash RUN/infer_strat.sh
```

### Training The Generation Model (Llama)
1. Download the model from: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Train the model with strategy special tokens. The checkpoint will be saved in /GenerationModel/DATA/strat_llama.strat_llama
```
bash RUN/train_strat_llama.sh
```
3. change the file name of the checkpoint to your preference.

The generation model can be infered with: bash RUN/infer_strat_llama.sh.
You can change GOLDEN_TRUTH in inputters/PARAMS.py to control whether use the golden strategy.

4. Infer with the generation model.
```
bash RUN/infer_strat_llama.sh
```

### Training The Empathy Attrubute Model
1. Go to the EmotionClassifier_head directory.
```
cd /EmotionClassifier_head/
```
2. Download the GoEmotion dataset: https://github.com/google-research/google-research/tree/master/goemotions/data
3. Run the code to preprocess the dataset:GoEmotion_preprocess.ipynb
4. Train the classifier.
```
bash train_emo_classifier.sh
```

* The attribute models are basically using generation model as encoder with classification head, you have to train the generation model first.
* Change the settings in "train_emo_classifier.sh" based on your generation model.
* The trained attribute model's setting can be found in /output_llama/GO_classifier_head_meta.json
* The class_vocab dictionary is always different after you trained.

### Training The Strategy Attrubute Model
1. Go to the StrategyClassifier_head directory.
```
cd /StrategyClassifier_head/
```
2. Train the classifier.
```
bash train_str_classifier.sh
```

* The attribute models are basically using generation model as encoder with classification head, you have to train the generation model first.
* Change the settings in train_str_classifier.shbased on your generation model.
* The trained attribute model's setting can be found in /output_llama/ESC_classifier_head_meta.json
* The class_vocab dictionary is always different after you trained.

### Training PAGE
1. Go to the PAGE directory.
```
cd /PAGE/
```
2. Train the ECE model.
```
python main.py
```

* You can download the original code from https://github.com/xiaojiegu/page
* Analyze ESConv with PAGE: bash analysis
* We found that the prediction of PAGE is inconsistant based on different version of transformers.
* encoder.py is the original code of PAGE, and encoder_emo_classifier.py is the updated version I applied an pretrained emotion classifier.

### Infering With DPPRSA (llama)
1. Go to the GenerationModel directory.
```
cd /GenerationModel/
```
2. Response Generation with DPPRSA and the attribute models.

  You should manually change the following commands in the shell file before running it.
* load_checkpoint: the location where you place your generation model.
* num_iterations: the perturbation time DPPRSA performs.
* emo_weight & str_weight: the base weighting of the attribute when neutral.
* joint: whether you use the strategy.
* gold: whether you use the golden strategy.
* page: whether you activate DAWG.
* rsa: whether you activate RSA inference.
* verbosity: you can see the detail of the generation.
* for_test_run: generate only one single response with detail.

  There are many commands remain redundant, they should be removed.

```
CUDA_VISIBLE_DEVICES=0 python3 run_DPPRSA_llama_master.py \
    --config_name strat_llama \
    --inputter_name strat_llama \
    --load_checkpoint DATA/strat_llama.strat_llama/2025-09-14141406.5e-05.8.2gpu/epoch-3.bin \
    --infer_input_file ./_reformat/SUPPORTER/test.txt \
    --verbosity quiet \
    --sample \
    --stepsize 0.01 \
    --attribute_type $type \
    --num_iterations 2 \
    --emo_weight 0.5 \
    --str_weight 0.5 \
    --joint \
    --page \
    --rsa
```

3. Manually change the settings of attribute models in  run_DPPRSA_blenderbot_master.py/run_DPPRSA_llama_master.py, the settings can refer to the meta file located at: StrategyClassifier_head/output_llama/ and EmotionClassifier_head/output_llama/

4. Strat generating responses with DPPRSA. 

The following command will run the full ablation study. (It will take a very very long time.)
```
bash RUN/infer_strat_dpprsa_llama.sh
```

### Infering With DPPRSA (blenderbot)
1. Go to the GenerationModel directory.
```
cd /GenerationModel/
```
2. Response Generation with DPPRSA and the attribute models.

Manually change the settings of attribute models in run_DPPRSA_llama_master.py, the settings can refer to the meta file located at: StrategyClassifier_head/output_llama/ and EmotionClassifier_head/output_llama/

This will run the full ablation study. (It will take a very very long time.)
Better run the python file itself.
```
bash RUN/infer_strat_dpprsa_llama.sh
```

### Experiment

* Perturbation time
```
bash RUN/exp_pplm_ptime.sh
```
* Weighting
```
bash RUN/exp_pplm_weight.sh
```

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
