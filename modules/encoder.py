import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward
from positional_encoder import PositionalEncoder


class EncoderLayer(nn.Module):
    def __init__(self, d_model=300, device=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model=d_model, device=device)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn



class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx, n_layers, max_seq_len=256, device=None, d_model=300, emb_model=None, dropout=0.1):

        super(Encoder, self).__init__()

        if emb_model == None:
            self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        else:
            self.embeddings = nn.Embedding.from_pretrained(
                embeddings=emb_model, freeze=True, padding_idx=pad_idx
            )
            
        self.position_enc = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, device=device)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model=d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        
        enc_output = self.dropout(self.position_enc(self.embeddings(src_seq)))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
