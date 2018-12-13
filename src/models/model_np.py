import numpy as np
from src.utils import normalize, relu
from flask import Flask

import torch
DATA_PATH = '/home/rohitalyosha/Student_Job/mannheim-nel/data/profile/'

app = Flask(__name__)
np.set_printoptions(threshold=10**6)


class NEL(object):

    def __init__(self, word_embs=None, ent_embs=None, params=None):
        self.word_embs = word_embs
        self.ent_embs = ent_embs
        self.orig_W = params['orig_linear.weight'].T  # Transpose herein
        self.orig_b = params['orig_linear.bias']
        self.hidden_W = torch.from_numpy(params['hidden.weight'].T).float()  # Transpose here
        self.hidden_b = torch.from_numpy(params['hidden.bias']).float()
        self.output_W = params['output.weight'].T  # Transpose here
        self.output_b = params['output.bias']

    def __call__(self, inputs):
        # Unpack
        context = inputs['context']
        cands = inputs['candidate_ids']
        priors = inputs['priors']
        conditionals = inputs['conditionals']
        exact_match = inputs['exact_match']
        contains = inputs['contains']

        # Get the embeddings
        candidate_embs = self.ent_embs[cands]
        context_embs = self.word_embs[context]

        # Aggregate context
        context_embs = normalize(context_embs.mean(axis=0))

        # Normalize / Pass through linear layer
        context_embs = (context_embs @ self.orig_W) + self.orig_b

        # Expand Context
        context_embs = np.tile(context_embs, reps=(*cands.shape, 1))

        # Dot product over last dimension
        dot_product = (context_embs * candidate_embs).sum(axis=2)

        # Create input for mlp
        input = (context_embs, dot_product[:, :, None], candidate_embs, priors[:, :, None],
                 conditionals[:, :, None], exact_match[:, :, None], contains[:, :, None])
        input = torch.from_numpy(np.concatenate(input, axis=2)).float()

        # Scores
        scores = ((input @ self.hidden_W) + self.hidden_b).numpy()
        scores = relu(scores)
        scores = np.squeeze((scores @ self.output_W) + self.output_b)

        return scores
