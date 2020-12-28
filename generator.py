import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class Generator(nn.Module):
    def __init__(self, vocab_size, emb_net=None, embedding_dim=EMBEDDING_DIM, head_size=8, hidden_size=HIDDEN_DIM, num_layers_of_enc=6, num_layers_of_dec=6, dropout=0.5):
        super(Generator, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.emb_net = emb_net
        
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, head_size, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers_of_enc)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(embedding_dim, head_size, hidden_size, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers_of_dec)

        self.vec2out = nn.Linear(embedding_dim, vocab_size)

        self.init_weights()
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        if self.emb_net == None:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        else:
            self.encoder.weight = nn.Parameter(self.emb_net)
            self.encoder.requires_grad = False

        self.vec2out.bias.data.zero_()
        self.vec2out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        output = self.transformer_decoder(tgt, output)
        output = self.vec2out(output)

        return output