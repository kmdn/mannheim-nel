from utils import get_normalised_forms, equalize_len
from more_itertools import unique_everseen
MAX_CANDS = 100


class Mention:

    def __init__(self,
                 text,
                 span,
                 cluster_mention=None,
                 file_stores=None):

        self.text = text
        self.begin, self.end = span
        self.cluster_mention = cluster_mention
        self.necounts = file_stores['str_necounts']
        self.rd = file_stores['redirects']
        self.dis_dict = file_stores['disamb']
        self.cands = []
        self.cand_ids = []

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

    def gen_cands(self):
        nfs = set()
        nfs.update(get_normalised_forms(self.text))
        nfs.update(get_normalised_forms(self.cluster_mention))
        [self.cands.extend(self.necounts.get(nf, [])) for nf in nfs]
        [self.cands.extend(self.add_dismb_cands(nf)) for nf in nfs]

        self.cands = equalize_len(list(unique_everseen(self.cands)), MAX_CANDS, pad='')
