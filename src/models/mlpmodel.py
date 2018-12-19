# Entity linking MLP.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.base import Base
from src.models.loss import Loss


class MLPModel(Loss, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        word_embs = kwargs['word_embs']
        ent_embs = kwargs['ent_embs']

        self.args = kwargs['args']

        self.emb_dim = word_embs.shape[1]
        self.num_ent = ent_embs.shape[0]

        # Words
        self.word_embs = nn.Embedding(*word_embs.shape, padding_idx=0, sparse=True)
        self.word_embs.weight.data.copy_(torch.from_numpy(word_embs) if isinstance(word_embs, np.ndarray) else word_embs)
        self.word_embs.weight.requires_grad = False

        # Entities
        self.ent_embs = nn.Embedding(*ent_embs.shape, padding_idx=0, sparse=True)
        self.ent_embs.weight.data.copy_(torch.from_numpy(ent_embs) if isinstance(ent_embs, np.ndarray) else ent_embs)
        self.ent_embs.weight.requires_grad = False

        # Pre trained linear layer
        self.orig_linear = nn.Linear(word_embs.shape[1], ent_embs.shape[1])

        # MLP Layers
        self.hidden = nn.Linear(6 + 2 * self.emb_dim, self.args.hidden_size)
        self.output = nn.Linear(self.args.hidden_size, 1)

        # Dropout
        self.dp = nn.Dropout(self.args.dp)

    def forward(self, input_dict):
        # Convert to tensor if input is numpy
        for k, v in input_dict.items():
            if isinstance(v, np.ndarray):
                input_dict[k] = torch.from_numpy(v)
        b, num_cand = input_dict['candidate_ids'].shape

        # Get the embeddings
        candidate_embs = self.ent_embs(input_dict['candidate_ids'])
        context_embs = self.word_embs(input_dict['context'])

        # Aggregate context
        context_embs = context_embs.mean(dim=len(context_embs.shape) - 2)

        # Normalize / Pass through linear layer / Unsqueeze
        context_embs = self.orig_linear(F.normalize(context_embs, dim=len(context_embs.shape) - 1))
        if len(context_embs.shape) == 1:
            context_embs = context_embs.unsqueeze(0)
        context_embs = context_embs.unsqueeze(1)
        context_embs = context_embs.expand(*candidate_embs.shape)

        # Dot product over last dimension
        dot_product = (context_embs * candidate_embs).sum(dim=2)

        # Unsqueeze in second dimension
        dot_product = dot_product.unsqueeze(dim=2)
        priors = input_dict['priors'].unsqueeze(dim=2)
        conditionals = input_dict['conditionals'].unsqueeze(dim=2)
        exact_match = input_dict['exact_match'].unsqueeze(dim=2)
        contains = input_dict['contains'].unsqueeze(dim=2)
        cand_cond_feature = input_dict['cand_cond_feature'].unsqueeze(dim=2)

        # Create input for mlp
        input = self.dp(torch.cat((context_embs,
                                   dot_product,
                                   candidate_embs,
                                   priors,
                                   conditionals,
                                   exact_match,
                                   contains,
                                   cand_cond_feature), dim=2))

        # Scores
        scores = self.output(F.relu(self.dp(self.hidden(input))))
        scores = scores.view(b, -1)

        return scores, context_embs, input

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)

