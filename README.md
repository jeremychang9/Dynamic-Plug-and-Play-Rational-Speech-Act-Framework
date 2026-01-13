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
