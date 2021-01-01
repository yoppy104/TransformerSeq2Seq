import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.encoder as enc
import modules.decoder as dec


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    batch_size, len_seq = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_seq, len_seq), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Generator(nn.Module):
    def __init__(self, vocab_size, max_src_len, max_tgt_len, emb_dim, pad_idx, device, n_enc_layer, n_dec_layer, emb_model, d_inner):
        super(Generator, self).__init__()

        self.pad_idx = pad_idx
        self.x_logit_scale = 1.

        self.encoder = enc.Encoder(vocab_size, emb_dim, pad_idx, n_enc_layer, \
                                    d_inner=d_inner, max_seq_len=max_src_len, device=device, d_model=emb_dim, emb_model=emb_model, dropout=0.1)

        self.decoder = dec.Decoder(vocab_size, emb_dim, pad_idx, n_dec_layer, \
                                    d_inner=d_inner, max_seq_len=max_tgt_len, device=device, d_model=emb_dim, emb_model=emb_model, dropout=0.1)

        self.out_layer = nn.Linear(emb_dim, vocab_size, bias=False)

        self.softmax = nn.Softmax(dim=2)

        for p in self.out_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    
    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.out_layer(dec_output) * self.x_logit_scale

        out = self.softmax(seq_logit)

        return out
