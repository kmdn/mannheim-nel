# This module implements dataloader for the yamada model

import torch.utils.data
import numpy as np
from more_itertools import unique_everseen

from src.utils.tokenizer import RegexpTokenizer
from src.utils.utils import reverse_dict, equalize_len, get_normalised_forms, normalise_form
import random

from logging import getLogger

logger = getLogger(__name__)


class Dataset(object):

    def __init__(self,
                 dicts=None,
                 id2context=None,
                 examples=None,
                 args=None,
                 cand_type='necounts',
                 data_type=None,
                 coref=False):
        super().__init__()

        self.args = args
        self.data_type = data_type
        self.word_tokenizer = RegexpTokenizer()

        # Dicts
        self.ent2id = dicts['ent_dict']
        self.len_ent = len(self.ent2id)
        self.id2ent = reverse_dict(self.ent2id)
        self.word_dict = dicts['word_dict']
        self.max_ent = len(self.ent2id)
        self.str_prior = dicts['str_prior']
        self.str_cond = dicts['str_cond']
        self.redirects = dicts['redirects']
        self.disamb = dicts['disamb']
        self.str_necounts = dicts['str_necounts']
        self.ent_strs = list(self.str_prior.keys())

        # Candidates
        self.num_candidates = self.args.num_candidates
        self.num_cand_gen = int(self.num_candidates * self.args.prop_gen_candidates)
        self.cand_type = cand_type

        # If coref, then there is a special format for which cands have been precomputed.
        # For file name checkout load_data function in utils file.
        self.coref = coref

        # Training data and context
        self.examples = examples
        self.id2context = id2context

        logger.info(f'Generating processed id2context')
        self.processed_id2context = {doc_id: self._init_context(doc_id) for doc_id in self.id2context.keys()}
        logger.info("Generated.")

    def _init_context(self, doc_id):
        """Initialize numpy array that will hold all context word tokens. Also return mentions"""

        context = self.id2context[doc_id]
        try:
            if isinstance(context, str):
                context = [self.word_dict.get(token, 0) for token in self.word_tokenizer.tokenize(context)]

            elif isinstance(context, tuple) and isinstance(context[0], str):
                context = [self.word_dict.get(token.lower(), 0) for token in context]
            context = np.array(equalize_len(context, self.args.max_context_size, pad=0))
        except Exception as e:
            print(e, context)

        assert np.any(context), ('CONTEXT IS ALL ZERO', self.id2context[doc_id])

        return context

    def _gen_features(self, mention_str, cand_strs):

        # Initialize
        exact = np.zeros(self.num_candidates).astype(np.float32)
        contains = np.zeros(self.num_candidates).astype(np.float32)
        priors = np.zeros(self.num_candidates).astype(np.float32)
        conditionals = np.zeros(self.num_candidates).astype(np.float32)

        # Populate
        for cand_idx, cand_str in enumerate(cand_strs):
            if mention_str == cand_str or mention_str in cand_str:
                exact[cand_idx] = 1
            if cand_str.startswith(mention_str) or cand_str.endswith(mention_str):
                contains[cand_idx] = 1

            priors[cand_idx] = self.str_prior.get(cand_str, 0)
            nf = normalise_form(mention_str)
            if nf in self.str_cond:
                conditionals[cand_idx] = self.str_cond[nf].get(cand_str, 0)
            else:
                conditionals[cand_idx] = 0

        return {'exact_match': exact,
                'contains': contains,
                'priors': priors,
                'conditionals': conditionals}

    def _get_cands(self, ent_str, cand_gen_strs):

        cand_gen_strs = list(unique_everseen(cand_gen_strs[:self.num_cand_gen]))
        if ent_str in cand_gen_strs:
            not_in_cand = False
            cand_gen_strs.remove(ent_str)
        else:
            not_in_cand = True

        len_rand = self.num_candidates - len(cand_gen_strs) - 1
        if len_rand >= 0:
            cand_strs = cand_gen_strs + random.sample(self.ent_strs, len_rand)
        else:
            cand_strs = cand_gen_strs[:-1]
        label = random.randint(0, self.args.num_candidates - 1)
        cand_strs.insert(label, ent_str)
        cand_ids = np.array([self.ent2id.get(cand_str, 0) for cand_str in cand_strs], dtype=np.int64)

        return cand_ids, cand_strs, not_in_cand, label

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        try:
            doc_id, mention_str, ent_str, cand_gen_strs = self.examples[index]
        except Exception as e:
            print(e)
            example_list = self.examples[index].split('||')
            doc_id, mention_str, ent_str = example_list[:3]
            cand_gen_strs = example_list[3:]

        ent_str = self.redirects.get(ent_str, ent_str)
        cand_ids, cand_strs, not_in_cand, label = self._get_cands(ent_str, cand_gen_strs)

        try:
            context = self.processed_id2context[doc_id]
        except KeyError:
            context = self.processed_id2context[str(doc_id)]
        features_dict = self._gen_features(mention_str, cand_strs)

        output = {'candidate_ids': cand_ids,
                  'not_in_cand': not_in_cand,
                  'context': context,
                  'cand_strs': cand_strs,
                  'ent_strs': ent_str,
                  'label': label,
                  **features_dict}

        return output

    def __len__(self):
        return len(self.examples)

    def get_loader(self,
                   batch_size=1,
                   shuffle=False,
                   sampler=None,
                   pin_memory=True,
                   drop_last=True,
                   num_workers=4):

        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)
