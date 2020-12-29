import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    
    def forward(self, x):
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret