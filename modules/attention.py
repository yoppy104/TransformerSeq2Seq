import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *

class Attention(nn.Module):
    def __init__(self, d_model=300):
        super(Attention, self).__init__()

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)

        self.d_k = d_model

        # init parameters
        self.init_params()

    
    def forward(self, q, k, v, mask):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask==0, -1e9)
        weights = weights[0, :, :, :]

        normlized_weights = F.softmax(weights, dim=-1)

        output = torch.matmul(normlized_weights, v)


        output = self.out(output)

        return output, normlized_weights


    def init_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, num_layer=3, d_model=300, device=None):
        super(MultiHeadAttention, self).__init__()

        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.layers = nn.ModuleList(
            [Attention(d_model=d_model) for i in range(num_layer)]
            )

        self.out_layer = nn.Linear(d_model*num_layer, d_model, bias=False)


    def forward(self, q, k, v, mask):
        out = torch.zeros(q.size(0), q.size(1), 1).to(self.device, non_blocking=True)
        norm = torch.zeros(q.size(0), q.size(1), 1).to(self.device, non_blocking=True)
        for attn in self.layers:
            attn_out, normalized = attn(q, k, v, mask)
            attn_out = attn_out.squeeze()
            if attn_out.dim() < 3:
                attn_out = attn_out.unsqueeze(0)
            normalized = normalized.squeeze()
            if normalized.dim() < 3:
                normalized = normalized.unsqueeze(0)
            out = torch.cat([out, attn_out], dim=2)
            norm = torch.cat([norm, normalized], dim=2)
        out = out[:, :, 1:]
        norm = norm[:, :, 1:]

        out = self.out_layer(out)

        return out, norm
