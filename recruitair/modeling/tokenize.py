"""The tokenizer for Resumes and their key criterias."""
# pylint: disable=R0903
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizer


class ResumeAndCriteriaTokenizer:
    """
    Manages tokenization of resumes and criterias for NLP models.
    """
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer):
        """
        Initialize the tokenizer for Resumes and their key criterias.
        """
        self.tokenizer = pretrained_tokenizer
        self.eos_token_id = pretrained_tokenizer.eos_token_id
        self.pad_token_id = pretrained_tokenizer.pad_token_id

    def __call__(
        self, resumes: List[str], criterias: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize resumes and criterias separately, then concatenate them with
        EOS in between and at the end.
        """
        # Join the resume and criteria tokens with "EOS" in between and at the end
        resume_tokens = list(
            map(lambda x: x + [self.eos_token_id], self.tokenizer(resumes)["input_ids"])
        )
        criteria_tokens = list(
            map(lambda x: x + [self.eos_token_id], self.tokenizer(criterias)["input_ids"])
        )
        input_tokens = [r + c for r, c in zip(resume_tokens, criteria_tokens)]
        # Pad sequences to the same length with padding on the left, to have the
        #  leftmost token be the final EOS
        padded_input_tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(t) for t in input_tokens],
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        # Compute attention mask (1 for real tokens, 0 for padding)
        attention_mask = (padded_input_tokens != self.pad_token_id).long()
        return padded_input_tokens, attention_mask
