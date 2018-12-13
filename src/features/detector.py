# Mention detection
import spacy
nlp = spacy.load('en')


class Detector:

    @staticmethod
    def spacy_detector(text):
        spacy_doc = nlp(text.text)
        text_spans = [(ent.text, (ent.start_char, ent.end_char))
                      for ent in spacy_doc.ents if not ent.label_ in ENT_FILTER and len(ent.text) > 2]
        text_spans.extend([(token.text, (token.idx, token.idx + len(token.text)))
                          for token in spacy_doc if token.text.isupper() and len(token.text) > 2])

        text_spans = sorted(text_spans, key=lambda m: m[1][0], reverse=True)

        return text_spans
