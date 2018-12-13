from src.repr.mention import Mention
import spacy
from src.utils.tokenizer import RegexpTokenizer

nlp = spacy.load('en')
ENT_FILTER = {'CARDINAL', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL'}
MAX_CONTEXT = 200


class Doc:

    def __init__(self, text, text_spans=None, file_stores=None, doc_id=None):
        self.text = text
        self.tokenizer = RegexpTokenizer()
        self.word_dict = file_stores['word_dict']
        self.doc_id = doc_id

        if not text_spans:
            text_spans = self._get_mentions()
        self.mentions = [Mention(text_span, file_stores=file_stores) for text_span in text_spans]

        self.assign_clusters()

    def _get_mentions(self):
        spacy_doc = nlp(self.text)
        text_spans = [(ent.text, (ent.start_char, ent.end_char))
                     for ent in spacy_doc.ents if not ent.label_ in ENT_FILTER and len(ent.text) > 2]
        # text_spans.extend([(token.text, (token.idx, token.idx + len(token.text)))
        #                  for token in spacy_doc if token.text.isupper() and len(token.text) > 2])

        # text_spans = sorted(text_spans, key=lambda m: m[1][0], reverse=True)

        return text_spans

    def assign_clusters(self):
        unchained_mentions = sorted(self.mentions, key=lambda m: m.begin, reverse=True)

        while unchained_mentions:
            mention = unchained_mentions.pop(0)

            potential_antecedents = [(m.text, m) for m in unchained_mentions]  # if m.tag == mention.tag
            chain = [mention]

            likely_acronym = False

            if mention.text.upper() == mention.text:
                # check if our mention is an acronym of a previous mention
                for a, m in potential_antecedents:
                    if (''.join(p[0] for p in a.split(' ') if p).upper() == mention.text) or \
                            (''.join(p[0] for p in a.split(' ') if p and p[0].isupper()).upper() == mention.text):
                        chain.insert(0, m)
                        unchained_mentions.remove(m)
                        likely_acronym = True
                potential_antecedents = [(m.text, m) for m in unchained_mentions]

            last = None
            longest_mention = mention
            while last != longest_mention and potential_antecedents:
                # check if we are a prefix/suffix of a preceding mention
                n = longest_mention.text.lower()
                for a, m in potential_antecedents:
                    na = a.lower()
                    if (likely_acronym and mention.text == a) or \
                            (not likely_acronym and (
                                    na.startswith(n) or na.endswith(n) or n.startswith(na) or n.endswith(na))):
                        chain.insert(0, m)
                        unchained_mentions.remove(m)

                last = longest_mention
                longest_mention = sorted(chain, key=lambda m: len(m.text), reverse=True)[0]
                potential_antecedents = [(m.text, m) for m in unchained_mentions]  # if m.tag == mention.tag

            for mention in chain:
                mention.cluster_mention = longest_mention.text

    def gen_cands(self):
        for mention in self.mentions:
            mention.gen_cands()

    def get_context_tokens(self):
        tokens = self.tokenizer.tokenize(self.text)
        token_ids = [self.word_dict.get(token, 0) for token in tokens][:MAX_CONTEXT]

        return token_ids
