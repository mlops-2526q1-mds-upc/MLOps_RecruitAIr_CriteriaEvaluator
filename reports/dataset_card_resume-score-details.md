---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
card_data: {{card_data}}
---

# Dataset Card for Resume score details

<!-- Provide a quick summary of the dataset. -->

This dataset contains 1,031 samples of resumes and job descriptions (JDs) generated and assessed using GPT-4o. The primary goal of this dataset is to evaluate the alignment between resumes and job descriptions, aiding in the study of resume relevance, skill alignment, and job fit scoring based on predefined criteria.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

The dataset includes resumes matched with job descriptions, with the assessment and scoring details based on various matching criteria:

- 201 Mismatched JSONs: Resumes that are not relevant to the provided JD.
- 648 Matched JSONs: Resumes that are relevant and aligned with the JD.
- 142 Invalid JSONs: Cases where either the resume or JD is incomplete or invalid.
- 40 JSONs Missing Additional Info: Instances where additional input information was omitted.

- **Owned by:** NETSOL Technologies Inc.
- **Language(s) (NLP):** English
- **License:** Creative Commons license family

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://huggingface.co/datasets/netsol/resume-score-details
## Uses

<!-- Address questions around how the dataset is intended to be used. -->

For this use case, the dataset will serve as a collection of example applications matching or not several job offers. It will be used to train a model that identifies and selects the most suitable candidates for each job posting based on their application data.

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset is designed to support research in:

- AI-driven recruitment: Assessing resume-JD alignment and scoring accuracy.

- Job Matching Algorithms: Testing algorithms that rank or filter candidates based on job fit.

- Natural Language Processing (NLP): Analyzing how NLP can evaluate resume relevance based on custom criteria.
<!--### Out-of-Scope Use>

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->



## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
Each sample JSON file in the dataset includes the following keys:

- **input**:

**job_description**: Contains the full job description text.

**macro_dict**: A dictionary with macro-level criteria and their respective weighting.

**micro_dict**: A dictionary with micro-level criteria and their respective weighting.

**additional_info**: Extra requirements or preferences related to the JD.

**minimum_requirements**: List of fundamental qualifications for the role.

**resume**: Text of the resume as provided.

- **output**:

**justification**: Reasons for the scores assigned, based on specific criteria.
scores.

**macro_scores**: Scores for broader criteria (e.g., experience, strategic thinking).

**micro_scores**: Scores for detailed criteria (e.g., market research expertise).

**requirements**: Boolean indicators showing if key requirements are met.

**aggregated_scores**: Overall scores for macro and micro criteria.

**personal_info**: Extracted personal details (e.g., name, contact details, current position).

**valid_resume_and_jd**: Boolean indicating if both resume and JD are valid for evaluation.
details:

Resume Analysis: Detailed breakdown of education, certifications, skills, project history, and professional experience.

<!-- Motivation for the creation of this dataset. -->

This dataset is structured to resemble real-world hiring scenarios, making it useful for data science and HR analytics projects.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

Data is created by GPT-4o.

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->


<!---#### Personal and Sensitive Information>

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->



<!---## Bias, Risks, and Limitations>

<!-- This section is meant to convey both technical and sociotechnical limitations. -->


<!---### Recommendations>

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->



## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->


**APA:**

Dataset generated using GPT-4o by [rohan/netsol].

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

{{ glossary | default("[More Information Needed]", true)}}
