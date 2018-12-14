# Coreference Resolution


class HeuresticCorefResolver:

    @staticmethod
    def coref(mentions):
        unchained_mentions = sorted(mentions, key=lambda m: m.begin, reverse=True)

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
