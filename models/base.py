# Base Model
import torch.nn as nn
import torch


class YamadaBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        word_embs = kwargs['word_embs']
        ent_embs = kwargs['ent_embs']
        W = kwargs['W']
        b = kwargs['b']

        self.args = kwargs['args']

        self.emb_dim = word_embs.shape[1]
        self.num_ent = ent_embs.shape[0]

        # Words
        self.word_embs = nn.Embedding(*word_embs.shape, padding_idx=0, sparse=True)
        self.word_embs.weight.data.copy_(torch.from_numpy(word_embs))
        self.word_embs.weight.requires_grad = False

        # Entities
        self.ent_embs = nn.Embedding(*ent_embs.shape, padding_idx=0, sparse=True)
        self.ent_embs.weight.data.copy_(torch.from_numpy(ent_embs))
        self.ent_embs.weight.requires_grad = False

        # Pre trained linear layer
        self.orig_linear = nn.Linear(word_embs.shape[1], ent_embs.shape[1])
        self.orig_linear.weight.data.copy_(torch.from_numpy(W))
        self.orig_linear.bias.data.copy_(torch.from_numpy(b))
        self.orig_linear.weight.requires_grad = True

        self.dp = nn.Dropout(self.args.dp)

    def forward(self, inputs):

        raise NotImplementedError

