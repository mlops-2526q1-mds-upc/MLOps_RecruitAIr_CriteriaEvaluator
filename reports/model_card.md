---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for RecruitAIr_CriteriaEvaluator

This model evaluates the fit of an applicant profile to user-defined criteria. It compares applicant data (CV, resume text, skills, experience) against structured job requirements and outputs a suitability score.

## Model Details

### Model Description

RecruitAIr_CriteriaEvaluator is a Natural Language Processing (NLP) model designed to compare applicant information with job criteria extracted from job postings (via RecruitAIr_JobCriteriaExtractor). It generates a numerical score (0–1) for each applicant-criterion pair, indicating how well the applicant meets the requirement.
The model ensures explainability by providing criterion-specific scores, which can later be aggregated (weighted averages) into overall applicant suitability rankings.

- **Developed by:** Alfonso Brown (github: abrownglez (https://github.com/abrowng)), Tania González (github: taaniagonzaalez (https://github.com/taaniagonzaalez)), Virginia Nicosia (github: viiirgi(https://github.com/viiiiirgi)), Marc Parcerisa (github: AimboParce (https://github.com/AimbotParce)), Daniel Reverter (github: danirc2 (https://github.com/danirc2))
- **Funded by:** The RecruitAIr team
- **Shared by:** Alfonso Brown, Tania González, Virginia Nicosia, Marc Parcerisa, Daniel Reverter
- **Model type:** Machine Learning (LLM-based scoring)
- **Language(s) (NLP):** English
- **License:** apache-2.0
- **Finetuned From Model:** Qwen/Qwen3-0.6B (https://huggingface.co/Qwen/Qwen3-0.6B)

### Model Sources 

- **Repository:** git@github.com:mlops-2526q1-mds-upc/MLOps_RecruitAIr_CriteriaEvaluator.git

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

-   Evaluating applicant suitability against structured job requirements.
-   Producing criterion-level scores (e.g., Python proficiency = 0.9, AWS knowledge = 0.6).
-   Supporting recruitment dashboards that show recruiters how applicants compare to requirements.

### Downstream Use 

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

-   Feeding into the RecruitAIr ranking engine for applicant ranking and filtering.
-   Assisting recruiters in identifying top candidates faster.
-   Enabling feedback loops for recruiters to refine evaluation models.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

-   Automated hiring decisions without human oversight.
-   Use with job descriptions or applicant data in non-English languages (model is trained only on English).
-   Extraction of demographic or sensitive attributes (e.g., gender, age, ethnicity).

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

-   Bias in Applicant Data: eesumes are not standardized; applicants with better formatting or keyword optimization may receive higher scores.
-   False Negatives/Positives: the model may misinterpret skills if expressed in uncommon wording.
-   Context Limitations: cannot fully capture soft skills, teamwork or cultural fit.
-   Risk of Overreliance: recruiters should not rely solely on scores for decision making.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Recruiters should use this model as a decision support tool, not a final hiring filter. They should review the results, especially for borderline candidates, and refine weights or thresholds as needed.

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model is trained using preprocessed JSONL data derived from multiple sources:

    - Hugging Face: HF_RESUME_SCORE_DETAILS_REPO (custom dataset of resume–criteria scoring pairs)

    - Kaggle: 
        -batuhanmutlu/job-skill-set (https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)
        -surendra365/recruitement-dataset (https://www.kaggle.com/datasets/surendra365/recruitement-dataset)

Dataset card for job skills: https://github.com/mlops-2526q1-mds-upc/MLOps_RecruitAIr_CriteriaEvaluator/blob/main/reports/dataset_card_jobs.md

Dataset card for recruitment: https://github.com/mlops-2526q1-mds-upc/MLOps_RecruitAIr_CriteriaEvaluator/blob/main/reports/dataset_card_recruitment.md

Raw .json files are downloaded automatically using the Hugging Face Hub API, and preprocessed into a unified JSONL structure of which every line is:
{
      "resume": "...",
      "criteria": "leadership",
      "score": 2
}

This ensures standardized training data where each record captures a candidate’s resume, a job criterion, and its numerical evaluation score.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing
Scripts used:
    - download_raw_dataset.py: downloads datasets from Hugging Face & Kaggle
    - preprocess_jsons.py: extracts resume–criteria–score tuples
    -split_data.py: splits into train/validation/test JSONL files

Data preprocessing steps are handled by preprocess_jsons.py:
    - Load raw JSON files (match_X.json, mismatch_X.json)
    - Extract fields from "input" and "output" keys
    - Flatten criteria and score pairs from both macro_scores and micro_scores
    - Export to JSONL as preprocessed_cvs.jsonl
Then, split_data.py divides the dataset into:
    - Train: TRAIN_SPLIT (default 70%)
    - Validation: VALIDATION_SPLIT (default 15%)
    - Test: Remaining 15%

All splits are stored under:
data/processed/
  ├── train.jsonl
  ├── validation.jsonl
  └── test.jsonl

#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Evaluation is performed on the test split of preprocessed data (data/processed/test.jsonl).

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

The model achieves high correlation and low error on unseen resume–criteria pairs, demonstrating robust generalization for typical job-matching tasks.

#### Summary

The model achieves high correlation and low error on unseen resume–criteria pairs, demonstrating robust generalization for typical job-matching tasks.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Attention visualizations confirm the model primarily focuses on skill and experience tokens.

Tokens such as “years”, “experience”, “Python”, “AWS” receive highest attention weights for technical criteria.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA RTX 3060
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** Spain
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

A lightweight transfer-learning architecture extending Qwen3-0.6B with a small feed-forward grading head.
The head maps the final hidden state of the last token to a scalar suitability score via Sigmoid activation.
Loss function: MSE / BCE on normalized scores (0–1).
This design allows efficient training and fast inference.

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

GPU: NVIDIA GeForce RTX3060

#### Software

{{ software | default("[More Information Needed]", true)}}

## Model Card Authors
Alfonso Brown, Tania González, Virginia Nicosia, Marc Parcerisa, Daniel Reverter

## Model Card Contact
