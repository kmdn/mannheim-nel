from src.utils.utils import reverse_dict, normalise_form, equalize_len
import numpy as np


class FeatureGenerator(object):

    def __init__(self, **kwargs):

        self.ent2id = kwargs['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.str_prior = kwargs['str_prior']
        self.str_cond = kwargs['str_cond']

    @staticmethod
    def get_string_feats(mention_str, candidate_strs):
        exact_match = [1 if mention_str in candidate_str else 0 for candidate_str in candidate_strs]
        contains = [1 if candidate_str.startswith(mention_str) or candidate_str.endswith(mention_str) else 0
                    for candidate_str in candidate_strs]

        return exact_match, contains

    def get_stat_feats(self, mention_str, candidates):
        nf = normalise_form(mention_str)
        priors = [self.str_prior.get(candidate, 0) for candidate in candidates]

        if nf in self.str_cond:
            conditionals = np.array([self.str_cond[nf].get(candidate, 0) for candidate in candidates], dtype=np.float32)
        else:
            conditionals = np.zeros(len(candidates))

        cand_cond_dict = dict(zip(candidates, conditionals))

        return priors, conditionals, cand_cond_dict

    def process(self, doc):
        context_tokens = doc.get_context_tokens()
        doc.gen_cands()

        all_exact_match = []
        all_contains = []
        all_priors = []
        all_conditionals = []
        all_candidate_strs = []
        all_candidate_ids = []
        all_cand_cond_dict = {}

        max_cand = max([len(mention.cands) for mention in doc.mentions])

        for men_idx, mention in enumerate(doc.mentions):
            mention.cands = equalize_len(mention.cands, max_cand, pad='')
            all_candidate_strs.append(mention.cands)
            all_candidate_ids.append([self.ent2id.get(cand, 0) for cand in mention.cands])

            exact_match, contains = self.get_string_feats(mention.text, mention.cands)
            all_exact_match.append(exact_match)
            all_contains.append(contains)

            priors, conditionals, cand_cond_dict = self.get_stat_feats(mention.text,
                                                                       mention.cands)
            for cand, cond in cand_cond_dict.items():
                all_cand_cond_dict[cand] = max(all_cand_cond_dict.get(cand, 0), cond)
            all_priors.append(priors)
            all_conditionals.append(conditionals)

        cand_cond_feature = [[all_cand_cond_dict[cand] for cand in cand_list]
                                 for cand_list in all_candidate_strs]

        ret = {'context': np.array(context_tokens, dtype=np.int64),
               'candidate_ids': np.array(all_candidate_ids, dtype=np.int64),
               'candidate_strs': np.array(all_candidate_strs),
               'priors': np.array(all_priors, dtype=np.float32),
               'conditionals': np.array(all_conditionals, dtype=np.float32),
               'exact_match': np.array(all_exact_match, dtype=np.float32),
               'contains': np.array(all_contains, dtype=np.float32),
               'cand_cond_feature': np.array(cand_cond_feature, dtype=np.float32)}

        # print(ret['candidate_strs'][:, :10])
        # print(ret['cand_cond_feature'][:, :10])

        return ret
