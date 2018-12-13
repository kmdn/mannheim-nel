# Candidate Generator
from more_itertools import unique_everseen
from src.utils.utils import get_normalised_forms, equalize_len


class CandidateGenerator:

    def __init__(self, max_cands=100, dis_dict=None, necounts=None, rd=None):
        self.max_cands = max_cands
        self.dis_dict = dis_dict
        self.necounts = necounts
        self.rd = rd

    def gen_cands(self, mention_text, cluster_mention_text):
        cands = []
        nfs = set()
        nfs.update(get_normalised_forms(mention_text))
        nfs.update(get_normalised_forms(cluster_mention_text))
        [cands.extend(self.necounts.get(nf, [])) for nf in nfs]
        [cands.extend(self.add_dismb_cands(nf)) for nf in nfs]

        return equalize_len(list(unique_everseen(cands)), self.max_cands, pad='')

    def add_dismb_cands(self, mention):
        res = []
        mention_title = mention.title().replace(' ', '_')
        res.append(mention_title)
        if mention_title != self.rd.get(mention_title, mention_title):
            res.append(self.rd[mention_title])
        mention_disamb = mention_title + '_(disambiguation)'

        if mention_title in self.dis_dict:
            res.extend(self.dis_dict[mention_title])
        if mention_disamb in self.dis_dict:
            res.extend(self.dis_dict[mention_disamb])

        return res
