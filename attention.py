import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, d_model=300):
        super(Attention, self).__init__()

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.d_k = d_model

    
    def forward(self, q, k, v, mask):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask==0, -1e9)

        normlized_weights = F.softmax(weights, dim=-1)

        output = torch.matmul(normlized_weights, v)

        output = self.out(output)

        return output, normlized_weights