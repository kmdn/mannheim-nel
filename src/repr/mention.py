# Mention Representation


class Mention:

    def __init__(self,
                 text_span,
                 cluster_mention=None,
                 file_stores=None,
                 candidate_generator=None):

        self.text = text_span[0]
        self.begin, self.end = text_span[1]
        self.cluster_mention = cluster_mention
        self.necounts = file_stores['str_necounts']
        self.rd = file_stores['redirects']
        self.dis_dict = file_stores['disamb']
        self.candidate_generator = candidate_generator
        self.cands = []

    def gen_cands(self):
        self.cands = self.candidate_generator.gen_cands(self.text, self.cluster_mention)
