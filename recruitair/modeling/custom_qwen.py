import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class _GradingHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # output between 0 and 1
        )

    def forward(self, x):
        return self.net(x)


class CustomQwenModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, **inputs):
        backbone_outputs = self.backbone(**inputs)
        last_hidden_state = backbone_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        # Instead of passing the entire last hidden state, we can just use the representation of the last token
        last_hidden_state = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        logits = self.head(last_hidden_state)  # (batch_size, 1)
        return logits


def customize_qwen_model(original_model: nn.Module, hidden_dim=512, dropout=0.5) -> CustomQwenModel:
    # Extract the backbone (all layers except the LM head)
    backbone = next(original_model.children())

    # Create the grading head
    in_dim = backbone.config.hidden_size
    head = _GradingHead(in_dim, hidden_dim=hidden_dim, dropout=dropout)

    # Combine backbone and head into a new model (ensure backbone is in float32)
    model = CustomQwenModel(backbone.to(torch.float32), head)

    return model


def freeze_custom_qwen_backbone(model: CustomQwenModel):
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.backbone.eval()
