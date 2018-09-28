import numpy as np
from utils import normalize, relu


class NEL(object):

    def __init__(self, params=None):
        self.word_embs = params['embeddings_word.weight']
        self.ent_embs = params['embeddings_ent.weight']
        self.orig_W = params['orig_linear.weight'].T  # Transpose here
        self.orig_b = params['orig_linear.bias']
        self.hidden_W = params['hidden.weight'].T  # Transpose here
        self.hidden_b = params['hidden.bias']
        self.output_W = params['output.weight'].T  # Transpose here
        self.output_b = params['output.bias']

    def __call__(self, inputs):
        # Unpack
        context, candidate_ids, priors, conditionals, exact_match, contains = inputs

        # Get the embeddings
        candidate_embs = self.ent_embs[candidate_ids]
        context_embs = self.word_embs[context]

        # Aggregate context
        context_embs = context_embs.mean(axis=0)

        # Normalize / Pass through linear layer
        context_embs = normalize(self.orig_W @ context_embs + self.orig_b)

        # Expand Context
        context_embs = np.tile(context_embs, reps=(*candidate_ids.shape, 1))

        # Dot product over last dimension
        dot_product = (context_embs * candidate_embs).sum(axis=2)

        # Create input for mlp
        input = (context_embs, dot_product[:, :, None], candidate_embs, priors[:, :, None],
                 conditionals[:, :, None], exact_match[:, :, None], contains[:, :, None])
        input = np.concatenate(input, axis=2)

        # Scores
        scores = input @ self.hidden_W + self.hidden_b
        scores = relu(scores)
        scores = scores @ self.output_W + self.output_b

        return np.squeeze(scores)
