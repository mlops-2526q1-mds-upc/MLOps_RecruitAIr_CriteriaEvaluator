## Functional Requirements of the Criteria Evaluation AI System
- The AI system shall receive an applicant's CV and data, as well as a job requirement's description (criterion), and return a numerical score between 0 and 1 (the front-end shall convert it to 0-10 with a single decimal point) indicating how well the applicant matches that specific requirement.

## Non-Functional Requirements of the Criteria Evaluation AI System
- The AI system in charge of matching applicants to job requirements shall achieve a mean absolute error of at most 0.5 in predicting the suitability score for individual criteria, as validated against a benchmark dataset with human-assigned scores.
- The AI system shall ensure that the suitability scores are consistent and reliable, with a variance of less than 0.5 in repeated evaluations of the same applicant against the same criterion.
- The AI system shall process and return the suitability score for an applicant against a single job requirement within 3 seconds for 95% of requests.