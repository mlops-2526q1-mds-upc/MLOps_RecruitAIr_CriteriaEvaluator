from unittest.mock import MagicMock

import torch
import torch.nn as nn

from recruitair.modeling.custom_qwen import (
    CustomQwenModel,
    _GradingHead,
    customize_qwen_model,
    freeze_custom_qwen_backbone,
)


def test_grading_head_forward_pass():
    """
    Test the forward pass of the _GradingHead module.
    """
    batch_size, in_dim, hidden_dim = 4, 128, 64
    head = _GradingHead(in_dim, hidden_dim)
    dummy_input = torch.randn(batch_size, in_dim)

    output = head(dummy_input)

    assert output.shape == (batch_size, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_custom_qwen_model_forward_pass():
    """
    Test the forward pass of the complete CustomQwenModel.
    """
    batch_size, seq_len, hidden_size = 2, 10, 128

    mock_backbone = MagicMock(spec=nn.Module)
    mock_backbone_output = MagicMock()
    mock_backbone_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    mock_backbone.return_value = mock_backbone_output

    head = _GradingHead(in_dim=hidden_size)
    model = CustomQwenModel(backbone=mock_backbone, head=head)
    dummy_inputs = {"input_ids": torch.randint(0, 100, (batch_size, seq_len))}

    output = model(**dummy_inputs)

    mock_backbone.assert_called_once_with(**dummy_inputs)
    assert output.shape == (batch_size, 1)


def test_customize_qwen_model():
    """
    Test the customize_qwen_model function to ensure it correctly builds the custom model.
    """
    hidden_size = 768

    mock_original_model = MagicMock(spec=nn.Module)

    mock_backbone = MagicMock(spec=nn.Module)
    mock_backbone.config = MagicMock()
    mock_backbone.config.hidden_size = hidden_size
    mock_original_model.children.return_value = iter([mock_backbone])

    mock_backbone.to = MagicMock(return_value=mock_backbone)

    custom_model = customize_qwen_model(mock_original_model, hidden_dim=256)

    assert isinstance(custom_model, CustomQwenModel)
    assert custom_model.backbone is mock_backbone
    mock_backbone.to.assert_called_once_with(torch.float32)
    assert isinstance(custom_model.head, _GradingHead)
    head_in_features = custom_model.head.net[0].in_features
    assert head_in_features == hidden_size


def test_freeze_custom_qwen_backbone():
    """
    Test that the freeze_custom_qwen_backbone function correctly freezes parameters.
    """
    backbone = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))
    backbone.eval = MagicMock()

    head = _GradingHead(in_dim=30)

    model = CustomQwenModel(backbone=backbone, head=head)

    for param in model.backbone.parameters():
        assert param.requires_grad is True

    freeze_custom_qwen_backbone(model)

    for param in model.backbone.parameters():
        assert param.requires_grad is False

    for param in model.head.parameters():
        assert param.requires_grad is True

    model.backbone.eval.assert_called_once()
