
# Model Card for RecruitAIr_CriteriaEvaluator

This model evaluates the fit of an applicant profile to user-defined criteria. It compares applicant data (CV, resume text, skills, experience) against structured job requirements and outputs a suitability score.

## Model Details

### Model Description

is a transfer-learned model built on Qwen3-0.6B for resume-job matching. It takes as input a candidate’s resume and a specific job criterion and outputs a score between 0 and 1 representing the degree of match. The model uses a frozen Qwen backbone and a lightweight grading head for regression.

- **Developed by:** Alfonso Brown (github: abrownglez (https://github.com/abrowng)), Tania González (github: taaniagonzaalez (https://github.com/taaniagonzaalez)), Virginia Nicosia (github: viiirgi(https://github.com/viiiiirgi)), Marc Parcerisa (github: AimboParce (https://github.com/AimbotParce)), Daniel Reverter (github: danirc2 (https://github.com/danirc2))
- **Funded by:** The RecruitAIr team
- **Shared by:** Alfonso Brown, Tania González, Virginia Nicosia, Marc Parcerisa, Daniel Reverter
- **Model type:** Transformer based Causal Language Model with regression head
- **Language(s) (NLP):** English
- **License:** MIT
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

```python
import os
import torch
import mlflow
from tempfile import TemporaryDirectory
from recruitair.modeling.tokenize import ResumeAndCriteriaTokenizer

mlflow.set_tracking_uri("http://nattech.fib.upc.edu:40380/")

MODEL_NAME = "criteria-evaluation"
VERSION = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained model
model = mlflow.pytorch.load_model(model_uri=f"models:/{MODEL_NAME}/{VERSION}", map_location=device)
model.eval()

# Load pre-trained tokenizer
with TemporaryDirectory() as tmpdirname:
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{VERSION}", dst_path=tmpdirname)
    tokenizer = ResumeAndCriteriaTokenizer.from_pretrained(os.path.join(tmpdirname, "extra_files", "tokenizer"))

resume = "Senior Data Scientist with 5 years of experience in NLP and ML..."
criteria = "Experience with deep learning and Python"

encoded_input = tokenizer(resume, criteria).to(device)

with torch.no_grad():
    score = model(**encoded_input)
print(f"Compatibility score: {score.item():.3f}")
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was trained on structured resume–criteria pairs extracted and cleaned from the Job Skill Set, Recruitment, and Resume Score Details datasets. Each sample includes a resume, a hiring criterion, and a human-aligned score.

- Hugging Face: HF_RESUME_SCORE_DETAILS_REPO (custom dataset of resume–criteria scoring pairs)

- Kaggle: 
    -batuhanmutlu/job-skill-set (https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)
    -surendra365/recruitement-dataset (https://www.kaggle.com/datasets/surendra365/recruitement-dataset)

Dataset card for job skills: https://github.com/mlops-2526q1-mds-upc/MLOps_RecruitAIr_CriteriaEvaluator/blob/main/reports/dataset_card_jobs.md

Dataset card for recruitment: https://github.com/mlops-2526q1-mds-upc/MLOps_RecruitAIr_CriteriaEvaluator/blob/main/reports/dataset_card_recruitment.md

Dataset card for resume scores: https://github.com/mlops-2526q1-mds-upc/MLOps_RecruitAIr_CriteriaEvaluator/blob/main/reports/dataset_card_resume-score-details.md

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing
Data was cleaned, normalized, tokenized with EOS separation, and split into train/validation/test sets. Long resumes were truncated, and malformed entries were filtered.

#### Training Hyperparameters

- BAse model: Qwen/Qwen3-0.6B
- Training regime: bf16 mixed precision
- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 1e-4
- Batch size: 8
- Epochs: 1
- Loss Function: Mean Squared Error (MSELoss)
 <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

A held-out validation set of 1,000 annotated resume–criterion pairs across various industries (IT, Finance, HR).

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluation considered job domain, posting length, and skill diversity.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->
We plan to evaluate model quality using the Mean Squared Error (MSE) to measuree how close predicted scores are to human (ground truth) scores.
### Results

Fine-tuning completed successfully on the Qwen/Qwen3-0.6B base model. 
- Training Loss (MSE): 0.00054
- Validation Loss (MSE): 0.00358
- Epoch Duration: 2.59 s
- Validation Duration: 3.56 s
- Total Training Duration: 8.36 s
#### Summary

The model achieved very low MSE values on both training and validation, suggesting stable learning and no major overfitting.

## Model Examination 

<!-- Relevant interpretability work for the model goes here -->

Model interpretability and attention visualization are planned for later milestones to better understand how the model focuses on skill- and experience-related information.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions were estimated using CodeCarbon

- **Hardware Type:** NVIDIA RTX 3060 (1 GPU) + 32 CPU cores + 61.6 GB RAM
- **Duration:** ~7.16s
- **Execution Type:** Local workstation
- **Compute Region:** Spain (approx. 41.442 N, 2.171 E)
- **Electricity Usage:** ~0.00566 kWh total (CPU ≈ 0.00540 kWh, GPU ≈ 0.00024 kWh, RAM ≈ 0.00002 kWh)
- **Carbon Emitted:** ≈ 0.000985 kg CO₂ eq (≈ 0.99 g CO₂ eq)
- **Power Usage Effectiveness (PUE):** 1 

This experiment consumed roughly 5.66 Wh of electricity, producing less than 1 g of CO₂ eq: a minimal footprint.

## Technical Specifications

### Model Architecture and Objective

Base: Qwen3-0.6B transformer
Objective: Regression head predicting compatibility score between 0–1 based on resume–criterion text pairs.

### Compute Infrastructure

Framework: PyTorch + Transformers
Environment: CUDA 12.x, Python 3.10

#### Hardware
- GPU: NVIDIA GeForce RTX3060
- CPU: 32 cores
- RAM 61.6 GB 

#### Software
- Transformers
- PyTorch
- Scikit-learn
- Great Expectations
- Python 3.11
- ollama / langchain-ollama
- mlflow-genai
- pandas
- loguru
- typer
- tqdm

## Model Card Authors
Alfonso Brown, Tania González, Virginia Nicosia, Marc Parcerisa, Daniel Reverter

## Model Card Contact
For any doubt or question you can write to any member of the RecruitAIr team:
- Alfonso Brown: alfonso.brown@estudiantat.upc.edu
- Tania González: tania.gonzalez@estudiantat.upc.edu
- Virginia Nicosia: virginia.nicosia@estudiantat.upc.edu
- Marc Parcerisa: marc.parcerisa@estudiantat.upc.edu
- Daniel Reverter: daniel.reverter@estudiantat.upc.edu
