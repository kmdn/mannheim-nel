# This module implements dataloader for the MLP model

import torch.utils.data
import numpy as np
from more_itertools import unique_everseen

from src.utils.tokenizer import RegexpTokenizer
from src.utils.utils import reverse_dict, equalize_len
from src.pipeline.features import FeatureGenerator
import random

from logging import getLogger

logger = getLogger(__name__)


class Dataset(object):

    def __init__(self,
                 file_stores=None,
                 id2context=None,
                 examples=None,
                 args=None,
                 data_type=None):
        super().__init__()

        self.args = args
        self.data_type = data_type
        self.word_tokenizer = RegexpTokenizer()

        # Dicts
        self.ent2id = file_stores['ent_dict']
        self.len_ent = len(self.ent2id)
        self.id2ent = reverse_dict(self.ent2id)
        self.word_dict = file_stores['word_dict']
        self.max_ent = len(self.ent2id)
        self.str_prior = file_stores['str_prior']
        self.redirects = file_stores['redirects']
        self.ent_strs = list(self.str_prior.keys())

        # Features
        self.feature_generator = FeatureGenerator(**file_stores)

        # Candidates
        self.num_candidates = self.args.num_candidates
        self.num_cand_gen = int(self.num_candidates * self.args.prop_gen_candidates)

        # Training data and context
        self.examples = examples
        self.id2context = id2context

        logger.info(f'Generating processed id2context')
        self.processed_id2context = {doc_id: self._init_context(doc_id) for doc_id in self.id2context.keys()}
        logger.info("Generated.")

    def _init_context(self, doc_id):
        """Initialize numpy array that will hold all context word tokens."""

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

    def _get_cands(self, ent_str, cand_gen_strs, cand_cond_feature):

        cand_gen_strs = cand_gen_strs[:self.num_cand_gen]
        cand_cond_gold = 0
        if ent_str in cand_gen_strs:
            not_in_cand = False
            index = cand_gen_strs.index(ent_str)
            cand_gen_strs.remove(ent_str)
            cand_cond_gold = cand_cond_feature.pop(index)
        else:
            not_in_cand = True

        len_rand = self.num_candidates - len(cand_gen_strs) - 1
        if len_rand >= 0:
            cand_strs = cand_gen_strs + random.sample(self.ent_strs, len_rand)
        else:
            cand_strs = cand_gen_strs[:-1]
        label = random.randint(0, self.args.num_candidates - 1)
        cand_strs.insert(label, ent_str)
        cand_cond_feature.insert(label, cand_cond_gold)
        cand_ids = np.array([self.ent2id.get(cand_str, 0) for cand_str in cand_strs], dtype=np.int64)

        return cand_ids, cand_strs, not_in_cand, label, cand_cond_feature

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        example_list = self.examples[index].split('||')
        doc_id, mention_str, ent_str = example_list[:3]

        cand_feature_list = example_list[3:]
        cand_gen_strs, cand_cond_feature = zip(*[cand_feature.split('@@') for cand_feature in cand_feature_list])
        cand_cond_feature = np.array([float(feature) for feature in equalize_len(list(cand_cond_feature), self.args.num_candidates)])

        ent_str = self.redirects.get(ent_str, ent_str)
        cand_ids, cand_strs, not_in_cand, label, cand_cond_feature = self._get_cands(ent_str, cand_gen_strs, cand_cond_feature)

        try:
            context = self.processed_id2context[doc_id]
        except KeyError:
            context = self.processed_id2context[str(doc_id)]

        exact_match, contains = self.feature_generator.get_string_feats(mention_str, cand_strs)
        priors, conditionals, _ = self.feature_generator.get_stat_feats(mention_str, cand_strs)

        output = {'candidate_ids': cand_ids,
                  'not_in_cand': not_in_cand,
                  'context': context,
                  'cand_strs': cand_strs,
                  'ent_strs': ent_str,
                  'label': label,
                  'cand_cond_feature': np.array(cand_cond_feature, dtype=np.float32),
                  'exact_match': np.array(exact_match, dtype=np.float32),
                  'contains': np.array(contains, dtype=np.float32),
                  'priors': np.array(priors, dtype=np.float32),
                  'conditionals': np.array(conditionals, dtype=np.float32)}

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
