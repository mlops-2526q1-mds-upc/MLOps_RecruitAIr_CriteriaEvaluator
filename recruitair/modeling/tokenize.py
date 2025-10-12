"""The tokenizer for Resumes and their key criterias."""

# pylint: disable=R0903
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer


class ResumeAndCriteriaTokenizer:
    """
    Manages tokenization of resumes and criterias for NLP models.
    """

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer):
        """
        Initialize the tokenizer for Resumes and their key criterias.
        """
        self.tokenizer = pretrained_tokenizer
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", None)
        if self.eos_token is None:
            raise ValueError("The tokenizer must have an EOS token.")

    def __call__(
        self,
        resumes: List[str] | str,
        criterias: List[str] | str,
        padding: bool = False,
        padding_side: str = "right",
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        """
        Tokenize resumes and criterias separately, then concatenate them with
        EOS in between and at the end.
        """
        if isinstance(resumes, str):
            resumes = [resumes]
        if isinstance(criterias, str):
            criterias = [criterias]
        if len(resumes) != len(criterias):
            raise ValueError("The number of resumes must match the number of criterias.")

        input_texts = [
            f"{resume}{self.eos_token}{criteria}{self.eos_token}"
            for resume, criteria in zip(resumes, criterias)
        ]
        encoded_inputs = self.tokenizer(
            input_texts, padding=padding, return_tensors=return_tensors, padding_side=padding_side
        )
        return encoded_inputs

    def save_pretrained(self, save_directory: str):
        """
        Save the tokenizer to a directory.
        """
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """
        Load a pretrained tokenizer from a model name or path.
        """
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(tokenizer)
