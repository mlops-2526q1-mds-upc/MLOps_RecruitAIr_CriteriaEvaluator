"""Functions for downloading raw datasets."""

import os
from pathlib import Path
import shutil
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError
import kagglehub
from kagglehub import KaggleDatasetAdapter
from requests import RequestException

from recruitair.config.data_download_config import (
    HF_RESUME_SCORE_DETAILS_REPO,
    HF_RESUME_SCORE_DETAILS_REVISION,
    RAW_DATA_DIR,
)


def download_kaggle_dataset() -> None:
    """
    Download the dataset from Kaggle and save it to the 'data/raw' directory.
    """

    skill_set_file = f"{RAW_DATA_DIR}/job-skill-set.csv"
    recruitment_file = f"{RAW_DATA_DIR}/recruitment.csv"

    df_job_skill_set = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "batuhanmutlu/job-skill-set",
        "all_job_post.csv",
    )
    df_recruitment = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "surendra365/recruitement-dataset",
        "job_applicant_dataset.csv",
    )

    # Create the parent dataset if not exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    df_job_skill_set.to_csv(skill_set_file)
    df_recruitment.to_csv(recruitment_file)


def download_huggingface_dataset_jsons(
    repo_id: str,
    raw_data_dir: str | Path,
    target_subdir: Optional[str] = None,
    revision: str = "main",
) -> None:
    """
    Download all .json files from a HF dataset repo (repo_type='dataset') and copy them
    into raw_data_dir/target_subdir (creates if needed), also save the commit SHA1.
    """

    # destination directory
    dataset_name = target_subdir or repo_id.split("/")[-1]
    dest_dir = os.path.join(raw_data_dir, dataset_name)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Listing files in dataset {repo_id} (revision={revision})...")
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        raise RuntimeError(f"Failed to list files from {repo_id}: {e}") from e

    # filter JSONs
    json_paths = [p for p in all_files if p.lower().endswith(".json")]
    if not json_paths:
        print("No JSON files found in the dataset repo.")
        return

    print(f"Found {len(json_paths)} JSON files. Downloading...")
    downloaded = 0
    for rel_path in sorted(json_paths):
        downloaded = download_huggingface_dataset(
            repo_id=repo_id,
            rel_path=rel_path,
            revision=revision,
            dest_dir=dest_dir,
        )
        if downloaded:
            downloaded += 1
        if downloaded % 50 == 0:
            print(f"  - downloaded {downloaded}/{len(json_paths)}")

    # Save SHA1
    sha1_path = os.path.join(dest_dir, "sha1.txt")
    with open(sha1_path, "w", encoding="utf-8") as f:
        f.write(revision)
    print(f"SHA1 saved to {sha1_path}")

    print(f"Done. {downloaded}/{len(json_paths)} JSON files saved to: {dest_dir}")


def download_huggingface_dataset(
    repo_id: str,
    rel_path: str,
    revision: str,
    dest_dir: str | Path,
) -> bool:
    """Download a huggingface dataset and store it in a file."""

    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=rel_path, repo_type="dataset", revision=revision)
    except (HfHubHTTPError, ValueError, RequestException) as e:
        print(f"  - ERROR downloading {rel_path}: {e}")
        return False

    # copy to destination preserving filename
    filename = os.path.basename(rel_path)
    target_path = os.path.join(dest_dir, filename)

    try:
        shutil.copy(local_path, target_path)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        OSError,
        shutil.SameFileError,
    ) as e:
        print(f"  - ERROR copying {local_path} -> {target_path}: {e}")
        return False

    return True


if __name__ == "__main__":
    download_kaggle_dataset()
    download_huggingface_dataset_jsons(
        repo_id=HF_RESUME_SCORE_DETAILS_REPO,
        raw_data_dir=RAW_DATA_DIR,
        target_subdir="raw_jsons",
        revision=HF_RESUME_SCORE_DETAILS_REVISION,
    )
