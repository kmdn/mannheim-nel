# Candidate Generator
from more_itertools import unique_everseen
from src.utils.utils import get_normalised_forms, equalize_len


class NelCandidateGenerator:

    def __init__(self,
                 max_cands=100,
                 file_stores=None):
        self.max_cands = max_cands
        self.disamb = file_stores['disamb']
        self.str_necounts = file_stores['str_necounts']
        self.rd = file_stores['redirects']

    def gen_cands(self, mention_text, cluster_mention_text):
        cands = []
        nfs = set()
        nfs.update(get_normalised_forms(mention_text))
        nfs.update(get_normalised_forms(cluster_mention_text))
        [cands.extend(self.str_necounts.get(nf, [])) for nf in nfs]
        [cands.extend(self._add_dismb_cands(nf)) for nf in nfs]
        cands = list(unique_everseen(cands))[:self.max_cands]

        return cands

    def _add_dismb_cands(self, mention):
        res = []
        mention_title = mention.title().replace(' ', '_')
        res.append(mention_title)
        if mention_title != self.rd.get(mention_title, mention_title):
            res.append(self.rd[mention_title])
        mention_disamb = mention_title + '_(disambiguation)'

        if mention_title in self.disamb:
            res.extend(self.disamb[mention_title])
        if mention_disamb in self.disamb:
            res.extend(self.disamb[mention_disamb])

        return res
