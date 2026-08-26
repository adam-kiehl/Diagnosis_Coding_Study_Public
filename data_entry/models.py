"""
Module Name: models.py
Author: David Kott and Adam Kiehl
Date: Updated 8/12/24

Description:
    Model classes for fine-tuning foundational models.
    Fine-tuning performed in trainer.py.

Classes:
    GatorTronClassifier
        GatorTron (2021) model class.
"""

# Import packages
import torch.nn as nn
from transformers import AutoModel


# Define GatorTron model architecture
class GatorTronClassifier(nn.Module):
    """GatorTron (2021) classifier."""

    def __init__(self, hidden_size, num_labels):
        super(GatorTronClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.gator = AutoModel.from_pretrained(
            "UFNLP/gatortron-medium"
        ).requires_grad_(requires_grad=False)
        self.gator.encoder.layer[12:48].requires_grad_(requires_grad=True)
        self.pooler = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, text, attention_mask):
        """
        Define forward pass through model.

        Args:
            text (tensor): Tokenized input text.
            attention_mask (tensor): Self-attention mask.

        Returns:
            tensor: Raw logit outputs from model.
        """

        x = self.gator(text, attention_mask)
        x = x["last_hidden_state"][:, 0]
        x = self.pooler(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
