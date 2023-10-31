import re
from sacremoses.corpus import Perluniprops
from sacremoses.corpus import NonbreakingPrefixes
perluniprops = Perluniprops()
nonbreaking_prefixes = NonbreakingPrefixes()


class MosesSentTokenizer(object):
    """
    This is a Python port of the Moses Tokenizer from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/ems/support/split-sentences.perl
    """
    raise NotImplementedError
    '\n    # Perl Unicode Properties character sets.\n    IsPi = str("".join(perluniprops.chars("IsPi")))\n    IsUpper = str("".join(perluniprops.chars("IsUpper")))\n    IsPf = str("".join(perluniprops.chars("IsPf")))\n    Punctuation = str("".join(perluniprops.chars("Punctuation")))\n    CJK = str("".join(perluniprops.chars("CJK")))\n    CJKSymbols = str("".join(perluniprops.chars("CJKSymbols")))\n    IsAlnum = str("".join(perluniprops.chars("IsAlnum")))\n\n    # Remove ASCII junk.\n    DEDUPLICATE_SPACE = r"\\s+", r" "\n\n    # Non-period end of sentence markers (?!) followed by sentence starters.\n    NONPERIOD_UPPER = r"([?!]) +([\\\'\\"\\(\\[\\¿\\¡\\p{startpunct}]*[\\p{upper}])".format(startpunct=IsPi, upper=IsUpper), r"\\1\\n\\2"\n\n    # Multi-dots followed by sentence starters.\n    MULTDOT_UPPER = r"(\\.[\\.]+) +([\\\'\\"\\(\\[\\¿\\¡\\p{startpunct}]*[\\p{upper}])".format(startpunct=IsPi, upper=IsUpper), r"\\1\\n\\2"\n\n    # Add breaks for sentences that end with some sort of punctuation\n    # inside a quote or parenthetical and are followed by a possible\n    # sentence starter punctuation and upper case.\n    QUOTES_UPPER = r"([?!\\.][\\ ]*[\\\'\\"\\)\\]\\p{endpunct}]+) +([\\\'\\"\\(\\[\\¿\\¡\\p{startpunct}]*[\\ ]*[\\p{upper}])".format(endpunct=IsPf, startpunct=IsPi, upper=IsUpper), r"\\1\\n\\2"\n\n    # Add breaks for sentences that end with some sort of punctuation,\n    # and are followed by a sentence starter punctuation and upper case.\n    ENDPUNCT_UPPER = r"([?!\\.]) +([\\\'\\"\\(\\[\\¿\\¡\\p{startpunct}]+[\\ ]*[\\p{upper}])".format(startpunct=IsPi, upper=IsUpper), r"\\1\\n\\2"\n\n    IS_EOS = r"([\\p{alphanum}\\.\\-]*)([\\\'\\"\\)\\]\\%\\p{endpunct}]*)(\\.+)$".format(alphanum=IsAlnum, endpunct=IsPf)\n\n\n    def __init__(self, lang="en", custom_nonbreaking_prefixes_file=None):\n        # Load custom nonbreaking prefixes file.\n        if custom_nonbreaking_prefixes_file:\n            self.NONBREAKING_PREFIXES  = []\n            with open(custom_nonbreaking_prefixes_file, \'r\') as fin:\n                for line in fin:\n                    line = line.strip()\n                    if line and not line.startswith("#"):\n                        if line not in self.NONBREAKING_PREFIXES:\n                            self.NONBREAKING_PREFIXES.append(line)\n\n        detokenized_text = ""\n        tokens = text.split()\n        # Iterate through every token till the last 2nd token.\n        for i, token in enumerate(iter(tokens[:-1])):\n            if re.search(IS_EOS, token):\n                pass\n    '


