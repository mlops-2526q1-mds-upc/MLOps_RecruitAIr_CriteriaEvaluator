---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
card_data: {{card_data}}
---

# Dataset Card for Recruitment

<!-- Provide a quick summary of the dataset. -->

This dataset contains applicant details, resumes, job descriptions, and matching labels to assess how well a candidate fits a specific job role.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

In today’s competitive job market, companies receive numerous applications for each job posting, making it challenging to efficiently screen and shortlist candidates. This dataset is designed to facilitate research and development in resume screening, job matching, and recruitment analytics.

The dataset was compiled from synthetic and publicly available job application data. It is structured to resemble real-world hiring scenarios, making it useful for data science and HR analytics projects. The resumes and job descriptions are either anonymized, synthesized, or derived from publicly accessible recruitment data.

- **Owned by:** Surendra Kumar Nellore
- **Language(s) (NLP):** English
- **License:** [Database: Open Database, Contents: Database Contents](https://opendatacommons.org/licenses/dbcl/1-0/)

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://www.kaggle.com/datasets/surendra365/recruitement-dataset

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

For this use case, the dataset will serve as a collection of example applications submitted in response to a specific job opening. It will be used to train a model that identifies and selects the most suitable candidates for each job posting based on their application data.

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset is useful for:

- Building AI-powered resume-screening models to automate candidate selection.

- Developing job recommendation systems that suggest the best roles for applicants.

- Analyzing hiring trends & biases in recruitment based on age, gender, or ethnicity.

- Training NLP models for resume parsing and job description understanding.

<!--### Out-of-Scope Use>

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->



## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

**Job Applicant Name:** Full name of the applicant.

**Age:** Applicant’s age.

**Gender:** Applicant’s gender identity.

**Race:** Racial background of the applicant.

**Ethnicity:** Ethnic identity of the applicant.

**Resume:** Text content of the applicant’s resume, including skills, experience, and education.

**Job Roles** The job positions for which the applicant applied.

**Job Description:** A detailed description of the job role, including required skills, responsibilities, and qualifications.

**Best Match:** A label or score indicating how well the applicant matches the job role based on qualifications and experience.

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

This dataset is structured to resemble real-world hiring scenarios, making it useful for data science and HR analytics projects.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

The dataset was compiled from synthetic and publicly available job application data.

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->


### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->


#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->



## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->


### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->



## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Dataset Card Authors [optional]

{{ dataset_card_authors | default("[More Information Needed]", true)}}

## Dataset Card Contact

{{ dataset_card_contact | default("[More Information Needed]", true)}}