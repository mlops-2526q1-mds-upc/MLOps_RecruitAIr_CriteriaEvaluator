import os
from unittest.mock import patch, MagicMock, call, ANY

import pandas as pd
import pytest
from huggingface_hub.utils import HfHubHTTPError
from more_itertools.more import side_effect

from recruitair.data.download_raw_dataset import (
    download_kaggle_dataset,
    download_huggingface_dataset_jsons,
    download_huggingface_dataset,
)

@patch("recruitair.data.download_raw_dataset.os.makedirs")
@patch("recruitair.data.download_raw_dataset.kagglehub")
@patch("recruitair.data.download_raw_dataset.RAW_DATA_DIR", "mock/data/raw")
def test_download_kaggle_dataset(mock_kagglehub, mock_makedirs):
    """
    Test the download_kaggle_dataset function successfully downloads and saves data.
    """
    mock_df_skill = MagicMock(spec=pd.DataFrame)
    mock_df_recruit = MagicMock(spec=pd.DataFrame)

    mock_kagglehub.dataset_load.side_effect = [mock_df_skill, mock_df_recruit]
    download_kaggle_dataset()
    mock_makedirs.assert_called_once_with("mock/data/raw", exist_ok=True)

    mock_kagglehub.dataset_load.assert_has_calls([
        call(ANY, "batuhanmutlu/job-skill-set", "all_job_post.csv"),
        call(ANY, "surendra365/recruitement-dataset", "job_applicant_dataset.csv"),
    ])

    mock_df_skill.to_csv.assert_called_once_with("mock/data/raw/job-skill-set.csv")
    mock_df_recruit.to_csv.assert_called_once_with("mock/data/raw/recruitment.csv")


@patch("recruitair.data.download_raw_dataset.shutil.copy")
@patch("recruitair.data.download_raw_dataset.hf_hub_download")
def test_download_huggingface_dataset_success(mock_hf_download, mock_shutil_copy, tmp_path):
    """
    Test successful download and copy of a single Hugging Face dataset file.
    """
    repo_id, rel_path, revision, dest_dir = "test/repo", "data/file.json", "main", tmp_path
    dummy_local_path = tmp_path / "hf_cache_file.json"
    dummy_local_path.touch()
    mock_hf_download.return_value = str(dummy_local_path)

    result = download_huggingface_dataset(repo_id, rel_path, revision, dest_dir)

    assert result is True
    mock_hf_download.assert_called_once_with(
        repo_id=repo_id, filename=rel_path, repo_type="dataset", revision=revision
    )
    expected_target_path = os.path.join(dest_dir, "file.json")
    mock_shutil_copy.assert_called_once_with(str(dummy_local_path), expected_target_path)


@patch("recruitair.data.download_raw_dataset.shutil.copy")
@patch("recruitair.data.download_raw_dataset.hf_hub_download", side_effect=HfHubHTTPError("Download failed"))
def test_download_huggingface_dataset_download_error(mock_hf_download, mock_shutil_copy, capsys, tmp_path):
    """
    Test download_huggingface_dataset handles Hugging Face download errors.
    """
    repo_id, rel_path, revision, dest_dir = "test/repo", "data/file.json", "main", tmp_path

    result = download_huggingface_dataset(repo_id, rel_path, revision, dest_dir)

    assert result is False
    mock_hf_download.assert_called_once()
    mock_shutil_copy.assert_not_called()
    captured = capsys.readouterr()
    assert "ERROR downloading data/file.json: Download failed" in captured.out


@patch("recruitair.data.download_raw_dataset.hf_hub_download")
@patch("recruitair.data.download_raw_dataset.shutil.copy", side_effect=PermissionError("Copy failed"))
def test_download_huggingface_dataset_copy_error(mock_shutil_copy, mock_hf_download, capsys, tmp_path):
    """
    Test download_huggingface_dataset handles file copy errors.
    """
    repo_id, rel_path, revision, dest_dir = "test/repo", "data/file.json", "main", tmp_path
    mock_hf_download.return_value = "/fake/path/file.json"

    result = download_huggingface_dataset(repo_id, rel_path, revision, dest_dir)

    assert result is False
    mock_shutil_copy.assert_called_once()
    captured = capsys.readouterr()
    assert "ERROR copying /fake/path/file.json" in captured.out
    assert "Copy failed" in captured.out


@patch("builtins.open")
@patch("recruitair.data.download_raw_dataset.download_huggingface_dataset")
@patch("recruitair.data.download_raw_dataset.list_repo_files")
@patch("recruitair.data.download_raw_dataset.os.makedirs")
def test_download_huggingface_dataset_jsons_success(mock_makedirs, mock_list_files, mock_download_single, mock_open, tmp_path):
    """
    Test successful download of multiple JSON files from a Hugging Face repo.
    """
    repo_id, raw_data_dir, revision = "test/repo", tmp_path, "main"
    dest_dir = str(tmp_path / "repo")
    mock_list_files.return_value = ["data/file1.json", "README.md", "data/file2.JSON"]
    mock_download_single.return_value = True

    download_huggingface_dataset_jsons(repo_id=repo_id, raw_data_dir=raw_data_dir, revision=revision)

    mock_makedirs.assert_called_once_with(dest_dir, exist_ok=True)
    mock_list_files.assert_called_once_with(repo_id, repo_type="dataset", revision=revision)
    expected_calls = [
        call(repo_id=repo_id, rel_path="data/file1.json", revision=revision, dest_dir=str(dest_dir)),
        call(repo_id=repo_id, rel_path="data/file2.JSON", revision=revision, dest_dir=str(dest_dir)),
    ]
    mock_download_single.assert_has_calls(expected_calls, any_order=True)

    sha1_path = os.path.join(dest_dir, "sha1.txt")
    mock_open.assert_called_once_with(sha1_path, "w", encoding="utf-8")
    mock_open().__enter__().write.assert_called_once_with(revision)


@patch("recruitair.data.download_raw_dataset.list_repo_files", side_effect=Exception("API Error"))
def test_download_huggingface_dataset_jsons_list_files_error(tmp_path):
    """
    Test download_huggingface_dataset_jsons raises an error if listing files fails.
    """
    with pytest.raises(RuntimeError, match="Failed to list files from test/repo: API Error"):
        download_huggingface_dataset_jsons(repo_id="test/repo", raw_data_dir=tmp_path)


@patch("recruitair.data.download_raw_dataset.download_huggingface_dataset")
@patch("recruitair.data.download_raw_dataset.list_repo_files", return_value=["README.md", "data.csv"])
def test_download_huggingface_dataset_jsons_no_json_files(mock_list_files, mock_download_single, capsys, tmp_path):
    """
    Test download_huggingface_dataset_jsons handles repos with no JSON files.
    """
    # Act
    download_huggingface_dataset_jsons(repo_id="test/repo", raw_data_dir=tmp_path)

    # Assert
    mock_download_single.assert_not_called()
    captured = capsys.readouterr()
    assert "No JSON files found in the dataset repo." in captured.out
