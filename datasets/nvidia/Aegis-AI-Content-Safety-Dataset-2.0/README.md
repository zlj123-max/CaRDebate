---
license: cc-by-4.0
language:
- en
pretty_name: Nemotron Content Safety Dataset V2
task_categories:
- text-classification
tags:
- safety
- content moderation
- LLM safety
- toxicity detection
- nemoguard
- aegis
- nemoguard
- nemotron
size_categories:
- 10K<n<100K

configs:
- config_name: default
  data_files:
  - split: train
    path: 
    - "train.json"
    - "refusals_train.json"
  - split: validation
    path: 
    - "validation.json"
    - "refusals_validation.json"
  - split: test
    path: "test.json"
  default: true
---

# 🛡️ Nemotron Content Safety Dataset V2

<!-- Provide a quick summary of the dataset. -->

The **Nemotron Content Safety Dataset V2**, formerly known as Aegis AI Content Safety Dataset 2.0, is comprised of `33,416` annotated interactions between humans and LLMs, split into `30,007` training samples, `1,445` validation samples,  and `1,964` test samples. This release is an extension of the previously published [Nemotron Content Safety Dataset V1](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0). 

To curate the dataset, we use the HuggingFace version of human preference data about harmlessness from [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf). We extract only the prompts, and elicit responses from [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1). Mistral v0.1 excels at instruction following, both harmful and unharmful, and thus generates high quality responses for the content moderation categories as compared to more recent models that have higher refusal rates for harmful instructions due to their safety alignment. 

The dataset adheres to a comprehensive and adaptable taxonomy for categorizing safety risks, structured into 12 top-level hazard categories with an extension to 9 fine-grained subcategories. The dataset leverages a hybrid data generation pipeline that combines human annotations at full dialog level with a multi-LLM "jury" system to assess the safety of responses.

## Dataset Details

### Dataset Description

Our curated [training dataset](../data/llama_training_ready/) consists of a mix collected or generated using the following data sources.
* Human-written prompts collected from the [Anthropic RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), [Do-Anything-Now DAN](https://github.com/verazuo/jailbreak_llms), and [AI-assisted Red-Teaming](https://www.kaggle.com/datasets/mpwolke/aart-ai-safety) datasets.
* LLM responses generated from [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
* Response safety labels generated using ensemble of 3 LLMs: [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1), [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct), and [Gemma-2-27B-it](https://huggingface.co/google/gemma-2-27b-it).
* We additionally mix-in refusal data generated using [Gemma-2-27B](https://build.nvidia.com/google/gemma-2-27b-it/modelcard.md) using custom prompt-engineered deflection strategies.
* Further data augmentation with [topic-following data](https://gitlab-master.nvidia.com/dialogue-research/topic_following/-/tree/main?ref_type=heads) generated using [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) is found to improve the robustness and adaptability of the model to custom categories. 

Additionally, for addressing the critical harm category of `Suicide and Self-Harm`, we have annotated about 358 samples from this [Suicide Detection dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?select=Suicide_Detection.csv) according to our taxonomy. We do not distribute these samples as part of this dataset. For more information on how to use these particular samples, see section on [Dataset Structure](#dataset-structure).  

**Data Collection Method**
* Hybrid: Human annotations with synthetic LLM augmentations 

**Labeling Method**
* Human - Overall content safety labels for the conversation
* Synthetic - Response safety label augmentation where needed

**Properties:**
The Aegis 2.0 dataset contains `25,007` user prompts, user prompts and LLM response single turn, user prompts and LLM response muliple turns. The validation set has `1,245` samples and test set has `1,964` samples from the above distribution. 

Additionally, the training dataset also contains `5,000` refusals data, where responses are synthetically generated using custom prompt-engineered deflection strategies. The validation set has `200` refusal samples and we do not augment the held out test set with any synthetic refusal data.

For training robust guard models that we release along with this dataset, we additionally use a specific type of data called [Topic-Following](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset) which has approximately ~8k samples that can be downloaded from this link.

- **Curated by:** NeMo Guardrails team, Nvidia.
- **Paper:** [AEGIS2.0: A Diverse AI Safety Dataset and
Risks Taxonomy for Alignment of LLM Guardrails](https://openreview.net/pdf?id=0MvGCv35wi)
- **Language(s) (NLP):** English (may contain small samples in other languages).
- **License:** CC-BY-4.0

## Uses

### Direct Use

1. To build LLM content safety moderation guardrails for LLMs, can be used to train both prompt and response moderation.

2. Can be used for alignment of general purpose LLMs towards safety along with carefully using the data for safe-unsafe preference pairs.


### Out-of-Scope Use

The data contain content that may be offensive or upsetting. Topics include, but are not limited to, discriminatory language and discussions of abuse, violence, self-harm, exploitation, and other potentially upsetting subject matter. Please only engage with the data in accordance with your own personal risk tolerance. The data are intended for research purposes, especially research that can make models less harmful. The views expressed in the data do not reflect the views of Nvidia or any of its employees. These data are not intended for training dialogue agents as this will likely lead to harmful model behavior.

## Dataset Structure

The main parition of the dataset now contains a minimal set of 9 columns. The column names and descriptions are as follows:

- id: the unique ID for the sample
- prompt: the first user turn. This will say "REDACTED" if an ID exists in the `reconstruction_id_if_redacted` column for redacted samples from the  [Suicide Detection dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?select=Suicide_Detection.csv). In this case The ID specified in the `reconstruction_id_if_redacted` column matches the sample in the Suicide dataset. You need to separately download those samples.
- response: the first assistant turn. `null` if prompt-only example. Sometimes can have an empty string '' instead of a `null`, in which case the response label would be present.
- prompt_label: binary safety label
- response_label; binary safety label. `null` if prompt-only example.
- prompt_label_source: always `human` annotated
- response_label_source: either `human`, `llm_jury`, or `refusal_data_augmentation`.
- violated_categories: comma-separated list of categories in order of frequency of annotations.


## Dataset Creation

### Annotation process

Quality Assurance (QA) is maintained by the leads of this project. Two to three times
per week, leads choose fifteen questions at random of every one hundred completed by
three annotators to reevaluate. This accounts for fifteen percent of the data analyzed
for three-way agreement at a minimum, often having at least twenty to thirty percent
analyzed to further ensure quality. These corrections are sent to each individual annotator
as audits, with brief explanations of why certain corrections were made with reference to
the project guidelines. Data sets are commonly broken into 2,000-4,000 text-based prompts
and delivered to annotators in batches of three to five. In the transitions between each
batch, Person In Charge (PIC) or the lead designate at least one full eight hour work day
for annotators to self-correct their categorization work. Many training sessions have been
held throughout the duration of this project for tips on best practices when self-correcting,
including filtering through key words in the data, referencing the regularly updated FAQ
sheet with example questions, and choosing completed prompts at random to reevaluate.
Annotators are instructed to only self-correct their own work and avoid looking at any
other annotations besides their own. Both Leads are also available at all times for further
questions or meetings to ensure consistent understanding of the material. Mandatory virtual
group training are held every two weeks or as needed depending on the circumstances of
the project. These meetings are led by leads and often utilize examples of commonly seen
discrepancies to present as learning opportunities.

#### Who are the annotators?

Throughout the three month time span of the Content Moderation Guardrails project, we
have averaged twelve annotators at any given time. Of these twelve, four annotators come
from Engineering backgrounds specializing in data analysis and collection, gaming, and
robotics. Eight annotators have a background in Creative Writing, with specialization in
linguistics, research and development, and other creative arts such as photography and
film. All annotators have been extensively trained in working with Large Language Models
(LLM), as well as other variations of Generative AI such as image retrieval or evaluations
of multi-turn conversations. All are capable of generating creative text-based output and
categorization work. Each of these twelve annotators resides in the United States, all from
various ethnic and religious backgrounds that allow for representation across race, age, and
social status.

## Content Safety Taxonomy

### Non-Unsafe Categories

- Safe
- Needs Caution

### Core Unsafe Categories

- Hate/Identity Hate
- Sexual
- Suicide and Self Harm
- Violence
- Guns/Illegal Weapons 
- Threat
- PII/Privacy
- Sexual Minor
- Criminal Planning/Confessions
- Harassment
- Controlled/Regulated substances 
- Profanity

### Fine-grained Unsafe Categories

- Illegal Activity
- Immoral/Unethical
- Unauthorized Advice
- Political/Misinformation/Conspiracy
- Fraud/Deception
- Copyright/Trademark/Plagiarism
- High Risk Gov. Decision Making
- Malware
- Manipulation

### Personal and Sensitive Information

This dataset comprises LLM responses generated from  [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1). We have carefully gone through the data and taken out anything that could have personal information in it. However, there is still a chance that some personal information might be left in the data. If you come across anything in the data that you think should not be made public, please let us know right away.

## Bias, Risks, and Limitations

- Safety and Moderation: This dataset is intended to be used for building content moderations systems, or aligning LLMs to generate safe responses. By the nature of the work, the dataset contains critically unsafe content and annotations for that content. Extreme care and caution should be exercised in referring and using this dataset.
- Legal Compliance: Users of this data are responsible for ensuring its appropriate use. The dataset should not be utilized in manners that conflict with legal and ethical standards.
- Non-Identification: Users of this data agree to not attempt to determine the identity of individuals in this dataset.

## Ethical Statement

The process in which the _Nemotron Content Safety Dataset V2_ creation abides by ethical data categorization work is based within the tooling of [Label Studio](http://label-studio.nvidia.com/user/login/), an open source data labeling tool often used for Nvidia's internal projects. This tooling technology allows for large sets of data to be analyzed by individual annotators without seeing the work of their peers. This is essential in preventing bias between annotators, as well as delivering prompts to each individual with variability so that no one annotator is completing similar tasks based
on how the data was initially arranged.

Due to the serious nature of this project, annotators were asked to join on a volunteer basis
based on their skill level, availability, and willingness to expose themselves to potentially
toxic content. Before work on this project began, all participants were asked to sign an
“_Adult Content Acknowledgement_” that coincides with the organization’s existing _AntiHarassment Policy_ and _Code of Conduct_. This was to ensure that all annotators be made  aware of the nature of this work, as well as the resources available to them should it affect their mental well-being. Regular 1:1 meetings were held between the leads assigned to this project and each annotator to make sure they are still comfortable with the material and are capable of continuing on with this type of work.

## Citation
```
@inproceedings{ghosh-etal-2025-aegis2,
    title = "{AEGIS}2.0: A Diverse {AI} Safety Dataset and Risks Taxonomy for Alignment of {LLM} Guardrails",
    author = "Ghosh, Shaona and Varshney, Prasoon and Sreedhar, Makesh Narsimhan and Padmakumar, Aishwarya and Rebedea, Traian and Varghese, Jibin Rajan and Parisien, Christopher",
    editor = "Chiruzzo, Luis and Ritter, Alan and Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.306/",
    doi = "10.18653/v1/2025.naacl-long.306",
    pages = "5992--6026",
    ISBN = "979-8-89176-189-6"
}
```

Dataset Card Authors and Contacts: <br>
Shaona Ghosh, Prasoon Varshney <br>
{shaonag,prasoonv}@nvidia.com