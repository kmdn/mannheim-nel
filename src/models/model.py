# Entity linking MLP.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import Base
from src.models.loss import Loss


class Model(Base, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hidden = nn.Linear(5 + 2 * self.emb_dim, self.args.hidden_size)
        self.output = nn.Linear(self.args.hidden_size, 1)

    def forward(self, input_dict):
        # Conver to tensor if input is numpy
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

        # Create input for mlp
        input = self.dp(torch.cat((context_embs,
                                   dot_product,
                                   candidate_embs,
                                   priors,
                                   conditionals,
                                   exact_match,
                                   contains), dim=2))

        # Scores
        scores = self.output(F.relu(self.dp(self.hidden(input))))
        scores = scores.view(b, -1)

        return scores, context_embs, input

    def loss(self, scores, labels):
        return self.cross_entropy(scores, labels)

