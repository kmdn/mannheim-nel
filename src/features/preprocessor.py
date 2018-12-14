from src.utils.utils import reverse_dict, normalise_form
import numpy as np


class PreProcessor(object):

    def __init__(self, **kwargs):

        self.ent2id = kwargs['ent_dict']
        self.id2ent = reverse_dict(self.ent2id)
        self.str_prior = kwargs['str_prior']
        self.str_cond = kwargs['str_cond']

    @staticmethod
    def _get_string_feats(mention_str, candidate_strs):
        exact_match = [1 if mention_str in candidate_str else 0 for candidate_str in candidate_strs]
        contains = [1 if candidate_str.startswith(mention_str) or candidate_str.endswith(mention_str) else 0
                    for candidate_str in candidate_strs]

        return exact_match, contains

    def _get_stat_feats(self, mention_str, candidates):
        nf = normalise_form(mention_str)
        priors = [self.str_prior.get(candidate, 0) for candidate in candidates]

        if nf in self.str_cond:
            conditionals = [self.str_cond[nf].get(candidate, 0) for candidate in candidates]
        else:
            conditionals = np.zeros(len(candidates))

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

        # examples_str = str(doc.doc_id) + '\n'

        for men_idx, mention in enumerate(doc.mentions):
            all_candidate_strs.append(mention.cands)
            # examples_str += '||'.join([mention.text] + [mention.ent] + mention.cands) + '\n'
            all_candidate_ids.append([self.ent2id.get(cand, 0) for cand in mention.cands])

            exact_match, contains = self._get_string_feats(mention.text, mention.cands)
            all_exact_match.append(exact_match)
            all_contains.append(contains)

            priors, conditionals = self._get_stat_feats(mention.text, mention.cands)
            all_priors.append(priors)
            all_conditionals.append(conditionals)

        # with open(f'/home/rohitalyosha/Student_Job/mannheim-nel/data/cands/{doc.doc_id}', 'w') as f:
        #     f.write(examples_str)

        ret = {'context': np.array(context_tokens, dtype=np.int64),
               'candidate_ids': np.array(all_candidate_ids, dtype=np.int64),
               'candidate_strs': np.array(all_candidate_strs),
               'priors': np.array(all_priors, dtype=np.float32),
               'conditionals': np.array(all_conditionals, dtype=np.float32),
               'exact_match': np.array(all_exact_match, dtype=np.float32),
               'contains': np.array(all_contains, dtype=np.float32)}

        return ret
