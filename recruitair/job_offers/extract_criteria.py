import os

from langchain_ollama import ChatOllama
import mlflow

from .models import KeyCriteriaResponse

PROMPT_VERSION = os.getenv("CRITERIA_EXTRACTION_PROMPT_VERSION", "1")
if not PROMPT_VERSION.isdigit():
    raise ValueError("CRITERIA_EXTRACTION_PROMPT_VERSION must be a digit or not set")
prompt_template = mlflow.genai.load_prompt("job-offer-criteria-extraction", version=PROMPT_VERSION)


def extract_key_criteria_from_job_offer(job_offer_text: str) -> KeyCriteriaResponse:
    llm = ChatOllama(model="dolphin3", temperature=0)
    prompt = prompt_template.format(job_offer_text=job_offer_text)
    response = llm.with_structured_output(prompt_template.response_format, method="json_schema").invoke(prompt)
    return KeyCriteriaResponse.model_validate(response)


if __name__ == "__main__":
    sample_job_offer = """
    We are looking for a Software Engineer with experience in Python and machine
    learning. The ideal candidate should have at least 3 years of experience in
    software development, a strong understanding of algorithms and data structures,
    and the ability to work in a fast-paced environment. Familiarity with cloud
    platforms like AWS or GCP is a plus. Excellent communication skills and the
    ability to work in a team are essential.
    """
    criteria = extract_key_criteria_from_job_offer(sample_job_offer)
    for criterion in criteria.key_criteria:
        print(f"Title: {criterion.title}")
        print(f"Description: {criterion.description}")
        print(f"Importance: {criterion.importance}")
        print()
