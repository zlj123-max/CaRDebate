---
license: cc-by-4.0
language:
- en
pretty_name: HelpSteer2
size_categories:
- 10K<n<100K
tags:
  - human-feedback
---
# HelpSteer2: Open-source dataset for training top-performing reward models


HelpSteer2 is an open-source Helpfulness Dataset (CC-BY-4.0) that supports aligning models to become more helpful, factually correct and coherent, while being adjustable in terms of the complexity and verbosity of its responses.
This dataset has been created in partnership with [Scale  AI](https://scale.com/). 

When used to tune a [Llama 3.1 70B Instruct Model](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), we achieve 94.1% on RewardBench, which makes it the best Reward Model as of 1 Oct 2024.
This reward model is available on HuggingFace in both .nemo format at [Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) or HF-compatible format at [Llama-3.1-Nemotron-70B-Reward-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF)

Using this reward model for RLHF (specifically, REINFORCE), we were able to align a Llama-3.1-70B-Instruct model to reach [AlpacaEval 2 LC](https://tatsu-lab.github.io/alpaca_eval/) of 57.6, [Arena Hard](https://github.com/lmarena/arena-hard-auto) of 85.0 and [GPT-4-Turbo MT-Bench](https://github.com/lm-sys/FastChat/pull/3158) of 8.98, which are known to be predictive of [LMSys Chatbot Arena Elo](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) This Instruct model is available at [Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct) as .nemo model and [Llama-3.1-Nemotron-70B-Instruct-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) as a HF Transformers model.

As of 1 Oct 2024, this aligned model is #1 on all three automatic alignment benchmarks, edging out strong frontier models such as GPT-4o and Claude 3.5 Sonnet.

See details on HelpSteer2-Preference paper at [https://arxiv.org/abs/2410.01257](https://arxiv.org/abs/2410.01257) - as a preview, this model can correctly the question ```How many r in strawberry?``` without specialized prompting or additional reasoning tokens:

```
A sweet question!
Let’s count the “R”s in “strawberry”:
1. S
2. T
3. R
4. A
5. W
6. B
7. E
8. R
9. R
10. Y
There are **3 “R”s** in the word “strawberry”.
```

Reward Models was trained using the open-source [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner).

HelpSteer2 is a follow-up to the popular [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) dataset and we recommend using HelpSteer2 instead of HelpSteer.

HelpSteer2 Paper : [HelpSteer2: Open-source dataset for training top-performing reward models](http://arxiv.org/abs/2406.08673)


## RewardBench Primary Dataset LeaderBoard

As of 1 Oct 2024, Llama-3.1-Nemotron-70B-Reward performs best Overall on RewardBench as well as with strong performance in Chat, Safety and Reasoning categories among the models below.

 | Model  | Type of Data Used For Training |  Overall | Chat | Chat-Hard | Safety | Reasoning | 
|:-----------------------------|:----------------|:-----|:----------|:-------|:----------|:-----------------------|
| _**Llama-3.1-Nemotron-70B-Reward**_ |Permissive Licensed Data Only (CC-BY-4.0) | **94.1** | **97.5** | 85.7 | **95.1** | **98.1** | 
| Skywork-Reward-Gemma-2-27B | Includes GPT4 Generated Data| 93.8  |  95.8  |  **91.4**  |  91.9  |  96.1 |
| TextEval-Llama3.1-70B  | Not disclosed  | 93.5  |  94.1  |  90.1  |  93.2  |  96.4 |
| Skywork-Critic-Llama-3.1-70B  | Not fully disclosed |  93.3  |  96.6  |  87.9 |  93.1  |  95.5 |
| SFR-LLaMa-3.1-70B-Judge-r  | Not fully disclosed  | 92.7  |  96.9  |  84.8  |  91.6  |  97.6
  | Nemotron-4-340B-Reward  | Permissive Licensed Data Only (CC-BY-4.0) | 92.0  | 95.8 |   87.1 | 91.5  | 93.7 | 
  | ArmoRM-Llama3-8B-v0.1 | Includes GPT4 Generated Data |  90.8 | 96.9     | 76.8  | 92.2 | 97.3  | 
  | Cohere May 2024   | Not disclosed |   89.5  | 96.4     | 71.3      | 92.7 | 97.7  | 
  | Llama3-70B-SteerLM-RM  | Permissive Licensed Data Only (CC-BY-4.0) | 88.8  | 91.3 |   80.3 | 92.8  | 90.7 | 
  | Google Gemini Pro 1.5 | Not disclosed |  88.1 | 92.3  | 80.6 | 87.5  | 92.0  | 
  | GPT-4o-2024-08-06 |Not disclosed | 86.7 | 96.1 | 76.1 | 88.1 | 86.6 |
  | claude-3-5-sonnet-20240620 | Not disclosed | 84.2 | 96.4 | 74.0 | 81.6 | 84.7 | 
  | Meta-Llama-3.1-70B-Instruct | Not fully disclosed | 84.0 | 97.2 | 70.2 | 82.8 | 86.0 |

       
To better understand why Llama-3.1-Nemotron-70B-Reward does less well in the Chat-Hard category, we analyze the scores for each consistutent subset under the  Chat-Hard category. We find that on categories that uses human annotations as ground truth, Llama-3.1-Nemotron-70B-Reward performs similar to Skywork-Reward-Gemma-2-27B (<= 2.2% difference).
On the other hand, when GPT-4 annotations are used as Ground-Truth, Llama-3.1-Nemotron-70B-Reward trails substantially behind Skywork-Reward-Gemma-2-27B (by 10.8 to 19.2%). This suggests that Skywork-Reward-Gemma-2-27B can better modelling GPT-4 preferences (but not human-annotated preferences), likely contributed by the inclusion of GPT-4 annotated training data used to train it found in the [OffSetBias dataset](https://huggingface.co/datasets/NCSOFT/offsetbias) as part of the [Skywork-Reward-Preference-80k](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.1).
| Model  | Type of Data Used For Training |  Chat-Hard | LLMBar-Adversarial-Manual  | LLMBar-Adversarial-Neighbour | LLMBar-Natural | LLMBar-Adversarial-GPTInst | LLMBar-Adversarial-GPTOut | MT-Bench-Hard| 
|:-----------------------------|:----------------|:-----|:----------|:-------|:----------|:-----------------------|:-----------------------|:-----------------------|
|||| Human as Ground Truth | Human as Ground Truth | Human as Ground Truth | _GPT-4 as Ground Truth_ |_GPT-4 as Ground Truth_ | _GPT-4 as Ground Truth_ |
| Llama-3.1-Nemotron-70B-Reward |Permissive Licensed Data Only (CC-BY-4.0) | 85.7 | 76.1  |  88.8  |  95.0  |  87.0  | 72.3  |  75.7 
| Skywork-Reward-Gemma-2-27B | Includes GPT4 Generated Data |  91.4  |  78.3  |  89.6  |  96.0  |  97.8  |  91.5 | 86.5|


## Dataset Description

HelpSteer contains 21, 362 samples, each containing a prompt, a response as well as five human-annotated attributes of the response, each ranging between 0 and 4 where higher means better for each attribute. Consecutive samples (e.g. sample 1 with 2, 3 with 4 ...) share the same prompt so this can be used for preference pairs based on the helpfulness score (e.g. training DPO or Preference RM), in addition to training SteerLM Regression RM. 

About 29% of all prompts used are multi-turn. In this case, the prompt consists of all of the user turns and all but the last assistant turn, which is contained in the response field. This is done because the attribute values only assessed only for the last assistant turn.

For multi-turn prompts, the structure of prompts look like below. This is how we used it for training SteerLM Regression Reward Models, but you need to use it in other settings, please adjust as required.

```json
{user turn 1}
<extra_id_1>Assistant
{assistant turn 1}
<extra_id_1>User
{user turn 2}
```

These attributes are:

1. **Helpfulness**: Overall helpfulness of the response to the prompt.
2. **Correctness**: Inclusion of all pertinent facts without errors. 
3. **Coherence**: Consistency and clarity of expression. 
4. **Complexity**: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
5. **Verbosity**: Amount of detail included in the response, relative to what is asked for in the prompt.

An example looks like

```json
{
  "prompt": "c#", 
  "response": "C# is a high-level, object-oriented programming language developed by Microsoft as part of its .NET initiative. It was created as a modern alternative to Java and supports a variety of programming paradigms, including imperative, functional, and event-driven. C# is primarily used for Windows application development, but it can also be used for web, mobile, and game development. The language is designed to be safe, secure, and efficient, and it provides developers with a rich set of libraries and tools for building robust and scalable applications. C# is also widely used in the game development industry, particularly in the development of games for the Xbox 360 and Xbox One consoles.", 
  "helpfulness": 3, 
  "correctness": 4, 
  "coherence": 4, 
  "complexity": 2, 
  "verbosity": 1
}
```



## Preferences (NEW - 1 Oct 2024)

We release the preference annotations between the responses. 

While existing HelpSteer2 can be use for Bradley-Terry Modelling (by taking difference in helpfulness), we also collected dedicated annotation for which among two responses they prefer. We recommend using this set of annotation for Bradley-Terry Modelling because it accounts for the situation that one response can be preferred over the other even when they have the same helpfulness score (e.g. 4)

For each sample, we provide:

**Preference Strength** (one of the following - score: meaning)
   
-3: Response 1 is much better than Response 2

-2: Response 1 is better than Response 2

-1: Response 1 is slightly better than Response 2

0: Response 1 is about the same as Response 2

1: Response 2 is slightly better than Response 1

2: Response 2 is better than Response 1

3: Response 2 is much better than Response 1

**Human-written Preference Justification**

This has been post-processed by us into a Preference statement and a Preference elaboration.

**Three Most Similar Preferences**

In calculating overall preference, we only consider the most similar preferences in order to filter outliers.

**All Preferences Unprocessed**

We did not use these in our experiments but we release all preferences in case someone finds them helpful.

**Split**

For clarity, the ```train``` subset of the ```preference``` annotations only contains prompts and responses from the original HelpSteer2 ```train``` set, as is for the validation subset. This represents our best attempt at doing a data matched comparison of the two data collection approaches.


An example looks like

```json
{
    "split": "train",
    "prompt": "Define Signal Discuss its various properties with the help of diagram",
    "response_1": "A signal is a form of energy that is used to transmit information from one place to another. It can be in the form of sound, light, radio waves, or any other form of energy that can be detected by a sensor or receiver.\n\nThe properties of a signal can be described by its amplitude, frequency, and phase.\n\nAmplitude: The amplitude of a signal is the measure of its strength or intensity. It is usually measured in volts or watts.\n\nFrequency: The frequency of a signal is the measure of how many times it repeats itself in a given period of time. It is usually measured in Hertz (Hz).\n\nPhase: The phase of a signal is the measure of its position in time. It is usually measured in degrees or radians.\n\nA signal can be represented by a graph that shows its amplitude or intensity over time. This graph is called a waveform. The shape of the waveform can provide information about the type of signal and the information it carries.",
    "response_2": "A signal is a message that is conveyed from a sender to a receiver through a communication channel. The message can be in the form of a sound, light, electrical, or any other form that can be transmitted and received. The properties of a signal can vary depending on the type of signal and the communication channel used. Here are some common properties of signals:\n\nAmplitude: This is the maximum value that a signal can take. It represents the strength of the signal. For example, in an electrical signal, amplitude can be measured in volts.\n\nFrequency: This is the number of cycles or oscillations of a signal per unit time. It is measured in hertz (Hz). For example, a sound signal with a frequency of 20 Hz vibrates 20 times per second.\n\nWavelength: This is the distance between two successive crests or troughs of a signal. It is measured in meters (m). For example, in a light signal, wavelength can be measured in nanometers (nm).\n\nPhase: This is the position of a signal with respect to a reference signal. It is measured in degrees (\u00b0) or radians (rad). For example, in an electrical signal, phase can be measured in relation to a reference signal.\n\nEnvelope: This is the curve that bounds the amplitude of a signal. It can be used to represent the shape of a signal. For example, in an audio signal, the envelope can represent the shape of a musical note.\n\nNoise: This is any unwanted or random variation in a signal that can affect its quality. It can be caused by various sources such as interference, interference, or natural phenomena.\n\nHere is a diagram that represents the properties of a signal:\n\n\nAmplitude\nFrequency\nWavelength\nPhase\nEnvelope\nNoise\n\n\nThe diagram shows how each property can affect the shape and quality of a signal. The presence of noise can degrade the quality of a signal and make it difficult to interpret. The properties of a signal can be controlled and optimized to ensure that the message is conveyed accurately and efficiently.",
    "preference_strength": 1,
    "preference_statement": "@Response 2 is better than @Response 1 because it provides a comprehensive insightful explanation of signanal and its properties.",
    "preference_elaboration": "It is complete, clear and correct as it discuss all the the poperties of signal while @Response 1 only discusses  three properties of signal. It does not diuscuss important properties like noise, phase and envelope. @Response 2  follows all the instruction but @Response 1 does not follow all the instruction. For instance the instruction requires an explanation of signal and its properties with an aid of a diagram but @Response 1 does not provide the diagram.",
    "three_most_similar_preferences": [
        {
            "statement": "@Response 2 is better than @Response 1 because it provides a comprehensive insightful explanation of signanal and its properties.",
            "elaboration": "It is complete, clear and correct as it discuss all the the poperties of signal while @Response 1 only discusses  three properties of signal. It does not diuscuss important properties like noise, phase and envelope. @Response 2  follows all the instruction but @Response 1 does not follow all the instruction. For instance the instruction requires an explanation of signal and its properties with an aid of a diagram but @Response 1 does not provide the diagram.",
            "strength": 1
        },
        {
            "statement": "@Response 2 is slightly better than @Response 1.",
            "elaboration": "@Response 2 goes into detail about the different types of signals that can be used for transmittal. Providing these topics gives a full overview of Signal Discuss. That makes this prompt complete, extremely helpful, and it is well-written. This response uses a paragraph format which breaks up the change in topic. @Response 1 covers a signal in less detail. It leaves out wavelengths, noise, and envelop as a way to transmit information from one network to another. This is not necessarily bad, but it is not in full detail.",
            "strength": 1
        },
        {
            "statement": "@Response 2 is slightly better than @Response 1 because it includes the diagram as requested by the prompt, which @Response 1 does not.",
            "elaboration": "However, @Response 2 does have issues with **correctness**: irrelevant terms like \"envelope\" are typically properties of the diagram, not the signal. **Formatting** could also be improved for @Response 2. While the diagram is included, it does not display correctly and the word \"interference\" is erroneously repeated twice.",
            "strength": 1
        }
    ],
    "all_preferences_unprocessed": [
        {
            "strength": 1,
            "justification": "@Response 2 is better than @Response 1 because it provides a comprehensive insightful explanation of signanal and its properties. It is complete, clear and correct as it discuss all the the poperties of signal while @Response 1 only discusses  three properties of signal. It does not diuscuss important properties like noise, phase and envelope. @Response 2  follows all the instruction but @Response 1 does not follow all the instruction. For instance the instruction requires an explanation of signal and its properties with an aid of a diagram but @Response 1 does not provide the diagram."
        },
        {
            "strength": 1,
            "justification": "@Response 2 is slightly better than @Response 1. @Response 2 goes into detail about the different types of signals that can be used for transmittal. Providing these topics gives a full overview of Signal Discuss. That makes this prompt complete, extremely helpful, and it is well-written. This response uses a paragraph format which breaks up the change in topic. @Response 1 covers a signal in less detail. It leaves out wavelengths, noise, and envelop as a way to transmit information from one network to another. This is not necessarily bad, but it is not in full detail."
        },
        {
            "strength": 1,
            "justification": "@Response 2 is slightly better than @Response 1 because it includes the diagram as requested by the prompt, which @Response 1 does not. However, @Response 2 does have issues with **correctness**: irrelevant terms like \"envelope\" are typically properties of the diagram, not the signal. **Formatting** could also be improved for @Response 2. While the diagram is included, it does not display correctly and the word \"interference\" is erroneously repeated twice. Although @Response 1 is more succinct and the writing style flows better, it falls short in **instructions following** and therefore @Response 2 is slightly better."
        }
    ]
}
```

## Disagreements (NEW - 1 Oct 2024)

We release the individual annotations that were used to calculate the mean score (after filtering outliers) for the five HelpSteer attributes.

The schema is similar to HelpSteer2, except that each entry is a list of labels given by various annotators. We believe this resource support studies on how/why annotators disagree with one another as well as transparency on the data collection process.

Note: A small proportion of samples only annotations by 2 annotators - this is because one or more of the annotations were labelled as not-ratable by annotators or did not have a valid annotation.

Here's an example below:

```json
{
    "prompt": "c#",
    "response": "C# is a high-level, object-oriented programming language developed by Microsoft as part of its .NET initiative. It was created as a modern alternative to Java and supports a variety of programming paradigms, including imperative, functional, and event-driven. C# is primarily used for Windows application development, but it can also be used for web, mobile, and game development. The language is designed to be safe, secure, and efficient, and it provides developers with a rich set of libraries and tools for building robust and scalable applications. C# is also widely used in the game development industry, particularly in the development of games for the Xbox 360 and Xbox One consoles.",
    "helpfulness": [
        3,
        3,
        4
    ],
    "correctness": [
        3,
        4,
        4
    ],
    "coherence": [
        4,
        3,
        4
    ],
    "complexity": [
        2,
        2,
        2
    ],
    "verbosity": [
        2,
        1,
        1
    ]
}
```


## Using the Huggingface Datasets

```python
from datasets import load_dataset
ds = load_dataset("nvidia/HelpSteer2")
train = ds['train'] # len(train) = 20324 (95%)
val = ds['validation']     # len(val) = 1038 (5%)
preference = load_dataset("nvidia/HelpSteer2", data_dir="preference")['train'] # despite the name, this contains both train and val, which you can use split to distinguish
disagreements = load_dataset("nvidia/HelpSteer2", data_dir="disagreements")['train']
```


## Source

1. Prompts are collected based on mostly user-contributed ShareGPT prompts and with a small proportion (~5%) that are human generated by Scale AI. 
2. Responses are generated by early versions of a mix of 10 different inhouse LLMs (note: none from properitary LLM providers such as OpenAI). We generate 2 responses per prompts (each from a different model) using sampling techniques to give diverse yet reasonable responses.
3. Annotations of various attributes were done by Scale AI. Annotators rated each response on a Likert 5 scale (between 0 and 4) for each attribute (helpfulness, correctness, coherence, complexity and verbosity).

## Annotation methodology (short)	

1. We engaged a select group of contractors via Scale AI. These contractors were provided with comprehensive guidelines that defined each attribute and the criteria for every rating level, together with some annotated examples. These guidelines and examples are detailed in the Appendix of the accompanying paper.
2. The annotation process involved approximately 1000 U.S.-based human annotators. Candidates first underwent preliminary assignments, including assessments of English proficiency, to determine eligibility for working on the project. Subsequently, they participated in an introductory training course on the task which ended with a test that involved annotating 35 sample responses. This process ensured not only a thorough understanding of the task requirements but also the delivery of high-quality annotations.
3. Every sample was independently annotated by a minimum of three annotators and up to five annotators, if the initial annotators do not agree with each other sufficiently (2 points or less on helpfulness). The final annotations (mean of 3.41 annotators) were obtain by taking the mean of the three annotators who agree with each other most, rounded to the nearest integer. 
4. Post-annotations, Scale AI performed extensive quality assurance, with each annotation reaching a minimum of two human reviews in addition to automated checks. After receiving the annotations from Scale AI, we conducted our independent quality assurance to make sure that the quality of the annotations was up to our expectations. As a result, many annotations were filtered away to retain only 20, 324 samples. 


## Ethical statement	
Annotators for the dataset were contracted through Scale AI. Scale AI engages the Anker Methodology, GISC Impact Sourcing Standard, and UN Sustainable Development Goals to provide a fair and competitive pay. The specific pay is calculated based on many factors, including the specific project, the specialized skillset and expertise required, regional costs of living and then transparently listed on Scale AI platform. Scale AI also provides multiple channels for questions and support, including 24/7 support teams, community discussion channels with specially trained moderators, and a “speak up” hotline where contractors can report concerns anonymously. Worker concerns can be submitted to and are reviewed by our Remotasks support team, and pay disputes are reviewed by support specialists trained in this area. 


## Citation

If you find this dataset useful, please cite the following works

```bibtex
@misc{wang2024helpsteer2preferencecomplementingratingspreferences,
      title={HelpSteer2-Preference: Complementing Ratings with Preferences}, 
      author={Zhilin Wang and Alexander Bukharin and Olivier Delalleau and Daniel Egert and Gerald Shen and Jiaqi Zeng and Oleksii Kuchaiev and Yi Dong},
      year={2024},
      eprint={2410.01257},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01257}, 
}

@misc{wang2024helpsteer2,
      title={HelpSteer2: Open-source dataset for training top-performing reward models}, 
      author={Zhilin Wang and Yi Dong and Olivier Delalleau and Jiaqi Zeng and Gerald Shen and Daniel Egert and Jimmy J. Zhang and Makesh Narsimhan Sreedhar and Oleksii Kuchaiev},
      year={2024},
      eprint={2406.08673},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```