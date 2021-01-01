import torch
import torch.nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward
from positional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    def __init__(self, d_model=300, device=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model=d_model, device=device)
        self.enc_attn = MultiHeadAttention(d_model=d_model, device=device)
        self.pos_ffn = PositionwiseFeedForward()

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx, n_layers, max_seq_len=256, device=None, d_model=300, emb_model=None, dropout=0.1):
        super(Decoder, self).__init__()

        if emb_model == None:
            self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        else:
            self.embeddings = nn.Embedding.from_pretrained(
                embeddings=emb_model, freeze=True, padding_idx=pad_idx
            )

        self.position_enc = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, device=device, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,