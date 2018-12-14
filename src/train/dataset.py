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
                 ent_prior=None,
                 ent_conditional=None,
                 ent_dict=None,
                 word_dict=None,
                 data=None,
                 args=None,
                 cand_rand=False,
                 cand_type='necounts',
                 data_type=None,
                 split=None,
                 necounts=None,
                 redirects=None,
                 dis_dict=None,
                 coref=False):
        super().__init__()

        self.args = args
        self.num_candidates = self.args.num_candidates
        self.num_cand_gen = int(self.num_candidates * self.args.prop_gen_candidates)
        self.ent2id = ent_dict
        self.len_ent = len(self.ent2id)
        self.id2ent = reverse_dict(self.ent2id)
        self.word_dict = word_dict
        self.max_ent = len(self.ent2id)
        self.ent_prior = ent_prior
        self.ent_conditional = ent_conditional
        self.ent_strs = list(self.ent_prior.keys())
        self.data_type = data_type
        self.split = split
        self.word_tokenizer = RegexpTokenizer()
        # If coref, then there is a special format for which cands have been precomputed.
        # For file name checkout load_data function in utils file.
        self.coref = coref

        self.redirects = redirects
        self.dis_dict = dis_dict

        self.cand_rand = cand_rand
        self.cand_type = cand_type
        if self.cand_rand:
            self.num_candidates = 10 ** 6

        self.necounts = necounts

        id2context, examples = data
        self.examples = examples
        self.id2context = id2context

        logger.info(f'Generating processed id2context')
        self.processed_id2context = {}
        for doc_id in self.id2context.keys():
            self.processed_id2context[doc_id] = self._init_context(doc_id)
        logger.info("Generated.")

    def add_dismb_cands(self, cands, mention):
        mention_title = mention.title().replace(' ', '_')
        cands.append(mention_title)
        if mention_title != self.redirects.get(mention_title, mention_title):
            cands.append(self.redirects[mention_title])
        mention_disamb = mention_title + '_(disambiguation)'

        if mention_title in self.dis_dict:
            cands.extend(self.dis_dict[mention_title])
        if mention_disamb in self.dis_dict:
            cands.extend(self.dis_dict[mention_disamb])

        return cands

    def _gen_cands(self, ent_str, mention):
        cand_gen_strs = self.add_dismb_cands([], mention)
        nfs = get_normalised_forms(mention)
        for nf in nfs:
            if nf in self.necounts:
                cand_gen_strs.extend(self.necounts[nf])

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

    def _init_context(self, doc_id):
        """Initialize numpy array that will hold all context word tokens. Also return mentions"""

        context = self.id2context[doc_id]
        context = context[5:] if self.args.ignore_init else context
        if isinstance(context, str):
            context = [self.word_dict.get(token.text, 0) for token in self.word_tokenizer.tokenize(context)]
        elif isinstance(context, tuple) and isinstance(context[0], str):
            context = [self.word_dict.get(token.lower(), 0) for token in context]
        context = np.array(equalize_len(context, self.args.max_context_size, pad=0))

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

            priors[cand_idx] = self.ent_prior.get(cand_str, 0)
            nf = normalise_form(mention_str)
            if nf in self.ent_conditional:
                conditionals[cand_idx] = self.ent_conditional[nf].get(cand_str, 0)
            else:
                conditionals[cand_idx] = 0

        return {'exact_match': exact,
                'contains': contains,
                'priors': priors,
                'conditionals': conditionals}

    def _gen_pershina_cands(self, doc_id, ent_str, mention_str):
        try:
            cand_strs = self.docid2candidates[doc_id][mention_str]
        except KeyError as K:
            print(K, doc_id)
            cand_strs = []
        cand_strs = equalize_len(cand_strs, self.args.num_candidates, pad='')
        if ent_str == cand_strs[0]:
            not_in_cand = 0
        else:
            not_in_cand = 1

        label = random.randint(0, self.args.num_candidates - 1)
        cand_strs = cand_strs[1:]
        cand_strs.insert(label, ent_str)
        cand_ids = np.array([self.ent2id.get(cand_str, 0) for cand_str in cand_strs], dtype=np.int64)

        return cand_ids, cand_strs, not_in_cand, label

    def _get_coref_cands(self, ent_str, cand_gen_strs):

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

        if self.coref:
            doc_id, mention_str, ent_str, cand_gen_strs = self.examples[index]
            ent_str = self.redirects.get(ent_str, ent_str)
            cand_ids, cand_strs, not_in_cand, label = self._get_coref_cands(ent_str, cand_gen_strs)
        else:
            doc_id, (mention_str, ent_str, _, _ ) = self.examples[index]
            ent_str = self.redirects.get(ent_str, ent_str)
            if self.cand_type == 'necounts':
                cand_ids, cand_strs, not_in_cand, label = self._gen_cands(ent_str, mention_str)
            else:
                cand_ids, cand_strs, not_in_cand, label = self._gen_pershina_cands(doc_id, ent_str, mention_str)

        try:
            context = self.processed_id2context[doc_id]
        except KeyError:
            context = self.processed_id2context[str(doc_id)]
        features_dict = self._gen_features(mention_str, cand_strs)

        output = {'cand_ids': cand_ids,
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
                   num_workers=4
                   ):

        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)
