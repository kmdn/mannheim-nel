from utils import reverse_dict, normalise_form, get_normalised_forms, equalize_len
import spacy
import numpy as np

import sys
from tokenizer import RegexpTokenizer


class PreProcessor(object):

    def __init__(self,
                 yamada_model=None,
                 necounts=None,
                 ent_priors=None,
                 ent_conditionals=None,
                 max_context=None,
                 max_cands=None,
                 filter_out=None):

        self.ent2id = yamada_model['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.ent_priors = ent_priors
        self.ent_conditionals = ent_conditionals
        self.nlp = spacy.load('en')
        self.word_dict = yamada_model['word_dict']
        self.tokenizer = RegexpTokenizer()
        self.necounts = necounts
        self.filter_out = filter_out
        self.max_context = max_context
        self.max_cands = max_cands

    def _get_string_feats(self, mention_str, candidates):
        exact_match = []
        contains = []

        for candidate in candidates:
            ent_str = self.id2ent.get(candidate, '')
            if mention_str == ent_str or mention_str in ent_str:
                exact_match.append(1)
            else:
                exact_match.append(0)

            if ent_str.startswith(mention_str) or ent_str.endswith(mention_str):
                contains.append(1)
            else:
                contains.append(0)

        return exact_match, contains

    def _get_stat_feats(self, mention_str, candidates):
        priors = []
        conditionals = []

        for candidate in candidates:
            priors.append(self.ent_priors.get(candidate, 0))
            nf = normalise_form(mention_str)
            if nf in self.ent_conditionals:
                conditionals.append(self.ent_conditionals[nf].get(candidate, 0))
            else:
                conditionals.append(0)

        if len(priors) == 0 or len(conditionals) == 0:
            print(mention_str, candidates)

        return priors, conditionals

    def _get_mentions(self, text):
        doc = self.nlp(text)
        strings = [ent.text for ent in doc.ents if not ent.label_ in self.filter_out]
        spans = [(ent.start_char, ent.end_char) for ent in doc.ents if not ent.label_ in self.filter_out]

        return strings, spans

    def _get_candidates(self, mentions):
        res = []
        for mention in mentions:
            nfs = get_normalised_forms(mention)
            candidate_ids = []
            for nf in nfs:
                if nf in self.necounts:
                    candidate_ids.extend(self.necounts[nf])

            res.append(equalize_len(candidate_ids, self.max_cands))

        return res

    def _get_context_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = [self.word_dict.get(token, 0) for token in tokens][:self.max_context]

        return token_ids

    def process(self, text):
        context_tokens = self._get_context_tokens(text)
        all_mentions, all_mention_spans = self._get_mentions(text)
        all_candidates = self._get_candidates(all_mentions)

        all_exact_match = []
        all_contains = []
        all_priors = []
        all_conditionals = []

        for men_idx, (mention, candidates) in enumerate(zip(all_mentions, all_candidates)):
            exact_match, contains = self._get_string_feats(mention, candidates)
            all_exact_match.append(exact_match)
            all_contains.append(contains)

            priors, conditionals = self._get_stat_feats(mention, candidates)
            all_priors.append(priors)
            all_conditionals.append(conditionals)

        return ((np.array(context_tokens),
                 np.array(all_candidates),
                 np.array(all_priors),
                 np.array(all_conditionals),
                 np.array(all_exact_match),
                 np.array(all_contains)),
                all_mentions,
                all_mention_spans)
