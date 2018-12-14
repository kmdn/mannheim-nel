# Doc Representation

from src.repr.mention import Mention
from src.utils.tokenizer import RegexpTokenizer

MAX_CONTEXT = 200


class Doc:

    def __init__(self, text,
                 text_spans=None,
                 file_stores=None,
                 doc_id=None,
                 detector=None,
                 coref_resolver=None,
                 candidate_generator=None):
        self.text = text
        self.tokenizer = RegexpTokenizer()
        self.word_dict = file_stores['word_dict']
        self.doc_id = doc_id

        if not text_spans:
            text_spans = detector.detect(text)
        self.mentions = [Mention(text_span,
                                 file_stores=file_stores,
                                 candidate_generator=candidate_generator)
                         for text_span in text_spans]

        # Performs coref resolution and assigns cluster_mention to each mention in self.mentions
        coref_resolver.coref(self.mentions)

    def gen_cands(self):
        for mention in self.mentions:
            mention.gen_cands()

    def get_context_tokens(self):
        tokens = self.tokenizer.tokenize(self.text)
        token_ids = [self.word_dict.get(token, 0) for token in tokens][:MAX_CONTEXT]

        return token_ids
