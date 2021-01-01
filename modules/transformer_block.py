import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(d_model, 3)

        self.ff = FeedForward(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)


    def forward(self, x, mask):
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask
        )

        x2 = x + self.dropout_1(output)

        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights

    

