# Mention detection
import spacy

# Don't detect these kind of entities
ENT_FILTER = {'CARDINAL', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL'}


class SpacyDetector:

    def __init__(self):
        self.nlp = spacy.load('en')

    def detect(self, text):
        spacy_doc = self.nlp(text)
        text_spans = [(ent.text, (ent.start_char, ent.end_char))
                      for ent in spacy_doc.ents if not ent.label_ in ENT_FILTER and len(ent.text) > 2]

        # Heuristic to add all caps of length greater than two as entities
        # text_spans.extend([(token.text, (token.idx, token.idx + len(token.text)))
        #                   for token in spacy_doc if token.text.isupper() and len(token.text) > 2])
        #
        # text_spans = sorted(text_spans, key=lambda m: m[1][0], reverse=False)

        return text_spans
