# Entity linking MLP.
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import YamadaBase
from models.loss import Loss


class Model(YamadaBase, Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hidden = nn.Linear(5 + 2 * self.emb_dim, self.args.hidden_size)
        self.output = nn.Linear(self.args.hidden_size, 1)

    def forward(self, input_dict):
        # Unpack

        candidate_ids = torch.from_numpy(input_dict['candidate_ids'])
        context = torch.from_numpy(input_dict['context'])
        priors = torch.from_numpy(input_dict['priors']).float()
        conditionals = torch.from_numpy(input_dict['conditionals']).float()
        exact_match = torch.from_numpy(input_dict['exact_match']).float()
        contains = torch.from_numpy(input_dict['contains']).float()
        b, num_cand = candidate_ids.shape

        # Get the embeddings
        candidate_embs = self.ent_embs(candidate_ids)
        context_embs = self.word_embs(context)

        # Aggregate context
        context_embs = context_embs.mean(dim=len(context_embs.shape) - 2)

        # Normalize / Pass through linear layer / Unsqueeze
        context_embs = self.orig_linear(F.normalize(context_embs, dim=len(context_embs.shape) - 1))
        context_embs = context_embs.expand(*candidate_embs.shape)

        # Dot product over last dimension
        dot_product = (context_embs * candidate_embs).sum(dim=2)

        # Unsqueeze in second dimension
        dot_product = dot_product.unsqueeze(dim=2)
        priors = priors.unsqueeze(dim=2)
        conditionals = conditionals.unsqueeze(dim=2)
        exact_match = exact_match.unsqueeze(dim=2)
        contains = contains.unsqueeze(dim=2)

        # Create input for mlp
        context_embs = context_embs.expand(-1, num_cand, -1)
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

