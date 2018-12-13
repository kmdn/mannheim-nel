from mention import Mention
import spacy
from tokenizer import RegexpTokenizer

nlp = spacy.load('en')
ENT_FILTER = {'CARDINAL', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL'}
MAX_CONTEXT = 200


class Doc:

    def __init__(self, text, mentions=None, spans=None, file_stores=None, doc_id=None):
        self.text = text
        self.tokenizer = RegexpTokenizer()
        self.word_dict = file_stores['word_dict']
        self.doc_id = doc_id

        if not mentions or not spans:
            mentions, spans = self._get_mentions()
        self.mention_strings = mentions
        self.mention_spans = spans
        self.mentions = [Mention(text, span, file_stores=file_stores) for text, span in zip(mentions, spans)]

        self.assign_clusters()

    def _get_mentions(self):
        spacy_doc = nlp(self.text)
        strings = [ent.text for ent in spacy_doc.ents if not ent.label_ in ENT_FILTER]
        strings.extend([token for token in spacy_doc if token.text.isupper()])
        spans = [(ent.start_char, ent.end_char) for ent in spacy_doc.ents if not ent.label_ in ENT_FILTER]

        return strings, spans

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
