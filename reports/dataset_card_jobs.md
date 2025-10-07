---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
card_data: {{card_data}}
---

# Dataset Card for Job Skill

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

The Job Skill Set Dataset is designed for use in machine learning projects related to job matching, skill extraction, and natural language processing tasks. The dataset includes detailed information about job roles, descriptions, and associated skill sets, enabling developers and researchers to build and evaluate models for career recommendation systems, resume parsing, and skill inference.

- **Owned by:** [Batuhan Mutlu](https://www.kaggle.com/batuhanmutlu)
- **Language(s) (NLP):** English
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

This dataset is primarily used to sample a diverse range of job offers that serve as input during the model training and evaluation process. By including various job descriptions, roles, and requirements, the dataset helps create realistic scenarios that simulate how different types of job openings are matched with candidate applications. This enables the model to learn how to identify the most suitable applicants for each position, based on the specific demands and characteristics of each job offer.

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset is particularly useful for the following applications:

- Skill Extraction: Identifying and parsing skills from job descriptions.

- Job-Resume Matching: Matching job descriptions with potential candidate profiles.

- Recommendation Systems: Developing models that recommend jobs or training programs based on required skills.

- Natural Language Processing: Experimenting with text-based models in recruitment and career analytics.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->



## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

The dataset contains the following features:

- **job_id:** A unique identifier for each job posting.

- **category:** The category of the job, such as INFORMATION-TECHNOLOGY,BUSINESS-DEVELOPMENT,FINANCE,SALES or HR.

- **job_title:** The title of the job position.

- **job_description:** A detailed text description of the job, including responsibilities and qualifications.

- **job_skill_set:** A list of relevant skills(include hard and soft skills) associated with the job, extracted using RecAI APIs.

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

This dataset was initially sourced from the Kaggle dataset titled [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by Arshkon. The original job postings data has been enhanced by extracting skill sets using [RecAI API services](https://recai.tech/en). These APIs are designed for skill parsing, resume analysis, and other recruitment-related tasks.

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->


#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

{{ source_data_producers_section | default("[More Information Needed]", true)}}

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

{{ annotation_process_section | default("[More Information Needed]", true)}}

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

{{ who_are_annotators_section | default("[More Information Needed]", true)}}

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

{{ personal_and_sensitive_information | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

{{ bias_recommendations | default("Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations.", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

Mutlu, B. (2024). job-skill-set. Kaggle. https://www.kaggle.com/dsv/10201355
 DOI: 10.34740/KAGGLE/DSV/10201355

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Dataset Card Authors [optional]

{{ dataset_card_authors | default("[More Information Needed]", true)}}

## Dataset Card Contact

{{ dataset_card_contact | default("[More Information Needed]", true)}}