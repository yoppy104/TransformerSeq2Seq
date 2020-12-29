import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors, pad_idx=0):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True, padding_idx=pad_idx
        )
    
    def forward(self, x):
        return self.embeddings(x)