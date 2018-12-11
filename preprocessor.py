from utils import reverse_dict, normalise_form
import numpy as np


class PreProcessor(object):

    def __init__(self, **kwargs):

        self.ent2id = kwargs['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.str_prior = kwargs['str_prior']
        self.str_cond = kwargs['str_cond']

    @staticmethod
    def _get_string_feats(mention_str, candidate_strs):
        exact_match = []
        contains = []

        for candidate_str in candidate_strs:
            if mention_str == candidate_str or mention_str in candidate_str:
                exact_match.append(1)
            else:
                exact_match.append(0)

            if candidate_str.startswith(mention_str) or candidate_str.endswith(mention_str):
                contains.append(1)
            else:
                contains.append(0)

        return exact_match, contains

    def _get_stat_feats(self, mention_str, candidates):
        priors = []
        conditionals = []

        for candidate in candidates:
            priors.append(self.str_prior.get(candidate, 0))
            nf = normalise_form(mention_str)
            if nf in self.str_cond:
                conditionals.append(self.str_cond[nf].get(candidate, 0))
            else:
                conditionals.append(0)

        if len(priors) == 0 or len(conditionals) == 0:
            print(mention_str, candidates)

        return priors, conditionals

    def process(self, doc):
        context_tokens = doc.get_context_tokens()
        doc.gen_cands()

        all_exact_match = []
        all_contains = []
        all_priors = []
        all_conditionals = []
        all_candidate_strs = []
        all_candidate_ids = []

        for men_idx, mention in enumerate(doc.mentions):
            all_candidate_strs.append(mention.cands)
            all_candidate_ids.append([self.ent2id.get(cand, 0) for cand in mention.cands])

            exact_match, contains = self._get_string_feats(mention.text, mention.cands)
            all_exact_match.append(exact_match)
            all_contains.append(contains)

            priors, conditionals = self._get_stat_feats(mention.text, mention.cands)
            all_priors.append(priors)
            all_conditionals.append(conditionals)

        ret = {'context': np.array(context_tokens, dtype=np.int64),
               'candidate_ids': np.array(all_candidate_ids, dtype=np.int64),
               'candidate_strs': np.array(all_candidate_strs),
               'priors': np.array(all_priors),
               'conditionals': np.array(all_conditionals),
               'exact_match': np.array(all_exact_match),
               'contains': np.array(all_contains)}

        return ret
