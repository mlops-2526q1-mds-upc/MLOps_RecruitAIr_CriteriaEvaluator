import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import shutil
from huggingface_hub import list_repo_files, hf_hub_download
from typing import Optional
from recruitair.config.data_download_config import RAW_DATA_DIR, HF_RESUME_SCORE_DETAILS_REPO, HF_RESUME_SCORE_DETAILS_REVISION

def download_kaggle_dataset():
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

    df_job_skill_set.to_csv(skill_set_file)
    df_recruitment.to_csv(recruitment_file)

def download_huggingface_dataset_jsons(
    repo_id: str,
    raw_data_dir: str,
    target_subdir: Optional[str] = None,
    revision: str = "main"
):
    """
    Download all .json files from a HF dataset repo (repo_type='dataset') and copy them
    into raw_data_dir/target_subdir (creates if needed), also save the commit SHA1.
    """
    # destination directory
    dataset_name = (target_subdir or repo_id.split("/")[-1])
    dest_dir = os.path.join(raw_data_dir, dataset_name)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Listing files in dataset {repo_id} (revision={revision})...")
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        raise RuntimeError(f"Failed to list files from {repo_id}: {e}")

    # filter JSONs
    json_paths = [p for p in all_files if p.lower().endswith(".json")]
    if not json_paths:
        print("No JSON files found in the dataset repo.")
        return

    print(f"Found {len(json_paths)} JSON files. Downloading...")
    downloaded = 0
    for rel_path in sorted(json_paths):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                repo_type="dataset",
                revision=revision
            )
        except Exception as e:
            print(f"  - ERROR downloading {rel_path}: {e}")
            continue

        # copy to destination preserving filename
        filename = os.path.basename(rel_path)
        target_path = os.path.join(dest_dir, filename)

        try:
            shutil.copy(local_path, target_path)
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"  - downloaded {downloaded}/{len(json_paths)}")
        except Exception as e:
            print(f"  - ERROR copying {local_path} -> {target_path}: {e}")

    # Save SHA1
    sha1_path = os.path.join(dest_dir, "sha1.txt")
    with open(sha1_path, "w") as f:
        f.write(revision)
    print(f"SHA1 saved to {sha1_path}")

    print(f"Done. {downloaded}/{len(json_paths)} JSON files saved to: {dest_dir}")


if __name__ == "__main__":
    download_kaggle_dataset()
    download_huggingface_dataset_jsons(
        repo_id=HF_RESUME_SCORE_DETAILS_REPO,
        raw_data_dir=RAW_DATA_DIR,
        target_subdir="raw_jsons",
        revision=HF_RESUME_SCORE_DETAILS_REVISION
    )
