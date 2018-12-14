import re


class RegexpTokenizer(object):
    __slots__ = ('_rule', 'lower')

    def __init__(self, rule=r"[\w\d]+"):
        self._rule = re.compile(rule, re.UNICODE)

    def tokenize(self, text):
        return [text[o.start():o.end()].lower() for o in self._rule.finditer(text)]

