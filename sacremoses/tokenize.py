import re
from sacremoses.corpus import Perluniprops
from sacremoses.corpus import NonbreakingPrefixes
from sacremoses.util import is_cjk
from sacremoses.indic import VIRAMAS, NUKTAS
perluniprops = Perluniprops()
nonbreaking_prefixes = NonbreakingPrefixes()


class MosesTokenizer(object):
    """
    This is a Python port of the Moses Tokenizer from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
    """
    IsN = str(''.join(perluniprops.chars('IsN')))
    IsAlnum = str(''.join(perluniprops.chars('IsAlnum')) + ''.join(VIRAMAS) + ''.join(NUKTAS))
    IsSc = str(''.join(perluniprops.chars('IsSc')))
    IsSo = str(''.join(perluniprops.chars('IsSo')))
    IsAlpha = str(''.join(perluniprops.chars('IsAlpha')) + ''.join(VIRAMAS) + ''.join(NUKTAS))
    IsLower = str(''.join(perluniprops.chars('IsLower')))
    DEDUPLICATE_SPACE = (re.compile('\\s+'), ' ')
    ASCII_JUNK = (re.compile('[\\000-\\037]'), '')
    MID_STRIP = (' +', ' ')
    LEFT_STRIP = ('^ ', '')
    RIGHT_STRIP = (' $', '')
    PAD_NOT_ISALNUM = (re.compile("([^{}\\s\\.'\\`\\,\\-])".format(IsAlnum)), ' \\1 ')
    AGGRESSIVE_HYPHEN_SPLIT = (re.compile('([{alphanum}])\\-(?=[{alphanum}])'.format(alphanum=IsAlnum)), '\\1 @-@ ')
    REPLACE_DOT_WITH_LITERALSTRING_1 = (re.compile('\\.([\\.]+)'), ' DOTMULTI\x01')
    REPLACE_DOT_WITH_LITERALSTRING_2 = (re.compile('DOTMULTI\\.([^\\.])'), 'DOTDOTMULTI \x01')
    REPLACE_DOT_WITH_LITERALSTRING_3 = (re.compile('DOTMULTI\\.'), 'DOTDOTMULTI')
    COMMA_SEPARATE_1 = (re.compile('([^{}])[,]'.format(IsN)), '\\1 , ')
    COMMA_SEPARATE_2 = (re.compile('[,]([^{}])'.format(IsN)), ' , \\1')
    COMMA_SEPARATE_3 = (re.compile('([{}])[,]$'.format(IsN)), '\\1 , ')
    DIRECTIONAL_QUOTE_1 = (re.compile('^``'), '`` ')
    DIRECTIONAL_QUOTE_2 = (re.compile('^"'), '`` ')
    DIRECTIONAL_QUOTE_3 = (re.compile('^`([^`])'), '` \\1')
    DIRECTIONAL_QUOTE_4 = (re.compile("^'"), '`  ')
    DIRECTIONAL_QUOTE_5 = (re.compile('([ ([{<])"'), '\\1 `` ')
    DIRECTIONAL_QUOTE_6 = (re.compile('([ ([{<])``'), '\\1 `` ')
    DIRECTIONAL_QUOTE_7 = (re.compile('([ ([{<])`([^`])'), '\\1 ` \\2')
    DIRECTIONAL_QUOTE_8 = (re.compile("([ ([{<])'"), '\\1 ` ')
    REPLACE_ELLIPSIS = (re.compile('\\.\\.\\.'), ' _ELLIPSIS_ ')
    RESTORE_ELLIPSIS = (re.compile('_ELLIPSIS_'), '\\.\\.\\.')
    COMMA_1 = (re.compile('([^{numbers}])[,]([^{numbers}])'.format(numbers=IsN)), '\\1 , \\2')
    COMMA_2 = (re.compile('([{numbers}])[,]([^{numbers}])'.format(numbers=IsN)), '\\1 , \\2')
    COMMA_3 = (re.compile('([^{numbers}])[,]([{numbers}])'.format(numbers=IsN)), '\\1 , \\2')
    SYMBOLS = (re.compile('([;:@#\\$%&{}{}])'.format(IsSc, IsSo)), ' \\1 ')
    INTRATOKEN_SLASHES = ('([{alphanum}])\\/([{alphanum}])'.format(alphanum=IsAlnum), '$1 \\@\\/\\@ $2')
    FINAL_PERIOD = (re.compile('([^.])([.])([\\]\\)}>"\']*) ?$'), '\\1 \\2\\3')
    PAD_QUESTION_EXCLAMATION_MARK = (re.compile('([?!])'), ' \\1 ')
    PAD_PARENTHESIS = (re.compile('([\\]\\[\\(\\){}<>])'), ' \\1 ')
    CONVERT_PARENTHESIS_1 = (re.compile('\\('), '-LRB-')
    CONVERT_PARENTHESIS_2 = (re.compile('\\)'), '-RRB-')
    CONVERT_PARENTHESIS_3 = (re.compile('\\['), '-LSB-')
    CONVERT_PARENTHESIS_4 = (re.compile('\\]'), '-RSB-')
    CONVERT_PARENTHESIS_5 = (re.compile('\\{'), '-LCB-')
    CONVERT_PARENTHESIS_6 = (re.compile('\\}'), '-RCB-')
    PAD_DOUBLE_DASHES = (re.compile('--'), ' -- ')
    PAD_START_OF_STR = (re.compile('^'), ' ')
    PAD_END_OF_STR = (re.compile('$'), ' ')
    CONVERT_DOUBLE_TO_SINGLE_QUOTES = (re.compile('"'), " '' ")
    HANDLES_SINGLE_QUOTES = (re.compile("([^'])' "), "\\1 ' ")
    APOSTROPHE = (re.compile("([^'])'"), "\\1 ' ")
    CONTRACTION_1 = (re.compile("'([sSmMdD]) "), " '\\1 ")
    CONTRACTION_2 = (re.compile("'ll "), " 'll ")
    CONTRACTION_3 = (re.compile("'re "), " 're ")
    CONTRACTION_4 = (re.compile("'ve "), " 've ")
    CONTRACTION_5 = (re.compile("n't "), " n't ")
    CONTRACTION_6 = (re.compile("'LL "), " 'LL ")
    CONTRACTION_7 = (re.compile("'RE "), " 'RE ")
    CONTRACTION_8 = (re.compile("'VE "), " 'VE ")
    CONTRACTION_9 = (re.compile("N'T "), " N'T ")
    CONTRACTION_10 = (re.compile(' ([Cc])annot '), ' \\1an not ')
    CONTRACTION_11 = (re.compile(" ([Dd])'ye "), " \\1' ye ")
    CONTRACTION_12 = (re.compile(' ([Gg])imme '), ' \\1im me ')
    CONTRACTION_13 = (re.compile(' ([Gg])onna '), ' \\1on na ')
    CONTRACTION_14 = (re.compile(' ([Gg])otta '), ' \\1ot ta ')
    CONTRACTION_15 = (re.compile(' ([Ll])emme '), ' \\1em me ')
    CONTRACTION_16 = (re.compile(" ([Mm])ore'n "), " \\1ore 'n ")
    CONTRACTION_17 = (re.compile(" '([Tt])is "), " '\\1 is ")
    CONTRACTION_18 = (re.compile(" '([Tt])was "), " '\\1 was ")
    CONTRACTION_19 = (re.compile(' ([Ww])anna '), ' \\1an na ')
    CLEAN_EXTRA_SPACE_1 = (re.compile('  *'), ' ')
    CLEAN_EXTRA_SPACE_2 = (re.compile('^ *'), '')
    CLEAN_EXTRA_SPACE_3 = (re.compile(' *$'), '')
    ESCAPE_AMPERSAND = (re.compile('&'), '&amp;')
    ESCAPE_PIPE = (re.compile('\\|'), '&#124;')
    ESCAPE_LEFT_ANGLE_BRACKET = (re.compile('<'), '&lt;')
    ESCAPE_RIGHT_ANGLE_BRACKET = (re.compile('>'), '&gt;')
    ESCAPE_SINGLE_QUOTE = (re.compile("\\'"), '&apos;')
    ESCAPE_DOUBLE_QUOTE = (re.compile('\\"'), '&quot;')
    ESCAPE_LEFT_SQUARE_BRACKET = (re.compile('\\['), '&#91;')
    ESCAPE_RIGHT_SQUARE_BRACKET = (re.compile(']'), '&#93;')
    EN_SPECIFIC_1 = (re.compile("([^{alpha}])[']([^{alpha}])".format(alpha=IsAlpha)), "\\1 ' \\2")
    EN_SPECIFIC_2 = (re.compile("([^{alpha}{isn}])[']([{alpha}])".format(alpha=IsAlpha, isn=IsN)), "\\1 ' \\2")
    EN_SPECIFIC_3 = (re.compile("([{alpha}])[']([^{alpha}])".format(alpha=IsAlpha)), "\\1 ' \\2")
    EN_SPECIFIC_4 = (re.compile("([{alpha}])[']([{alpha}])".format(alpha=IsAlpha)), "\\1 '\\2")
    EN_SPECIFIC_5 = (re.compile("([{isn}])[']([s])".format(isn=IsN)), "\\1 '\\2")
    ENGLISH_SPECIFIC_APOSTROPHE = [EN_SPECIFIC_1, EN_SPECIFIC_2, EN_SPECIFIC_3, EN_SPECIFIC_4, EN_SPECIFIC_5]
    FR_IT_SPECIFIC_1 = (re.compile("([^{alpha}])[']([^{alpha}])".format(alpha=IsAlpha)), "\\1 ' \\2")
    FR_IT_SPECIFIC_2 = (re.compile("([^{alpha}])[']([{alpha}])".format(alpha=IsAlpha)), "\\1 ' \\2")
    FR_IT_SPECIFIC_3 = (re.compile("([{alpha}])[']([^{alpha}])".format(alpha=IsAlpha)), "\\1 ' \\2")
    FR_IT_SPECIFIC_4 = (re.compile("([{alpha}])[']([{alpha}])".format(alpha=IsAlpha)), "\\1' \\2")
    FR_IT_SPECIFIC_APOSTROPHE = [FR_IT_SPECIFIC_1, FR_IT_SPECIFIC_2, FR_IT_SPECIFIC_3, FR_IT_SPECIFIC_4]
    NON_SPECIFIC_APOSTROPHE = (re.compile("\\'"), " ' ")
    TRAILING_DOT_APOSTROPHE = (re.compile("\\.' ?$"), " . ' ")
    BASIC_PROTECTED_PATTERN_1 = '<\\/?\\S+\\/?>'
    BASIC_PROTECTED_PATTERN_2 = '<\\S+( [a-zA-Z0-9]+\\="?[^"]")+ ?\\/?>'
    BASIC_PROTECTED_PATTERN_3 = "<\\S+( [a-zA-Z0-9]+\\='?[^']')+ ?\\/?>"
    BASIC_PROTECTED_PATTERN_4 = '[\\w\\-\\_\\.]+\\@([\\w\\-\\_]+\\.)+[a-zA-Z]{2,}'
    BASIC_PROTECTED_PATTERN_5 = '(http[s]?|ftp):\\/\\/[^:\\/\\s]+(\\/\\w+)*\\/[\\w\\-\\.]+'
    MOSES_PENN_REGEXES_1 = [DEDUPLICATE_SPACE, ASCII_JUNK, DIRECTIONAL_QUOTE_1, DIRECTIONAL_QUOTE_2, DIRECTIONAL_QUOTE_3, DIRECTIONAL_QUOTE_4, DIRECTIONAL_QUOTE_5, DIRECTIONAL_QUOTE_6, DIRECTIONAL_QUOTE_7, DIRECTIONAL_QUOTE_8, REPLACE_ELLIPSIS, COMMA_1, COMMA_2, COMMA_3, SYMBOLS, INTRATOKEN_SLASHES, FINAL_PERIOD, PAD_QUESTION_EXCLAMATION_MARK, PAD_PARENTHESIS, CONVERT_PARENTHESIS_1, CONVERT_PARENTHESIS_2, CONVERT_PARENTHESIS_3, CONVERT_PARENTHESIS_4, CONVERT_PARENTHESIS_5, CONVERT_PARENTHESIS_6, PAD_DOUBLE_DASHES, PAD_START_OF_STR, PAD_END_OF_STR, CONVERT_DOUBLE_TO_SINGLE_QUOTES, HANDLES_SINGLE_QUOTES, APOSTROPHE, CONTRACTION_1, CONTRACTION_2, CONTRACTION_3, CONTRACTION_4, CONTRACTION_5, CONTRACTION_6, CONTRACTION_7, CONTRACTION_8, CONTRACTION_9, CONTRACTION_10, CONTRACTION_11, CONTRACTION_12, CONTRACTION_13, CONTRACTION_14, CONTRACTION_15, CONTRACTION_16, CONTRACTION_17, CONTRACTION_18, CONTRACTION_19]
    MOSES_PENN_REGEXES_2 = [RESTORE_ELLIPSIS, CLEAN_EXTRA_SPACE_1, CLEAN_EXTRA_SPACE_2, CLEAN_EXTRA_SPACE_3, ESCAPE_AMPERSAND, ESCAPE_PIPE, ESCAPE_LEFT_ANGLE_BRACKET, ESCAPE_RIGHT_ANGLE_BRACKET, ESCAPE_SINGLE_QUOTE, ESCAPE_DOUBLE_QUOTE]
    MOSES_ESCAPE_XML_REGEXES = [ESCAPE_AMPERSAND, ESCAPE_PIPE, ESCAPE_LEFT_ANGLE_BRACKET, ESCAPE_RIGHT_ANGLE_BRACKET, ESCAPE_SINGLE_QUOTE, ESCAPE_DOUBLE_QUOTE, ESCAPE_LEFT_SQUARE_BRACKET, ESCAPE_RIGHT_SQUARE_BRACKET]
    BASIC_PROTECTED_PATTERNS = [BASIC_PROTECTED_PATTERN_1, BASIC_PROTECTED_PATTERN_2, BASIC_PROTECTED_PATTERN_3, BASIC_PROTECTED_PATTERN_4, BASIC_PROTECTED_PATTERN_5]
    WEB_PROTECTED_PATTERNS = ['((https?|ftp|rsync)://|www\\.)[^ ]*', '[\\w\\-\\_\\.]+\\@([\\w\\-\\_]+\\.)+[a-zA-Z]{2,}', '@[a-zA-Z0-9_]+', '#[a-zA-Z0-9_]+']
    
    def __init__(self, lang='en', custom_nonbreaking_prefixes_file=None):
        super(MosesTokenizer, self).__init__()
        self.lang = lang
        self.NONBREAKING_PREFIXES = [_nbp.strip() for _nbp in nonbreaking_prefixes.words(lang)]
        if custom_nonbreaking_prefixes_file:
            self.NONBREAKING_PREFIXES = []
            with open(custom_nonbreaking_prefixes_file, 'r') as fin:
                for line in fin:
                    line = line.strip()
                    if (line and not line.startswith('#')):
                        if line not in self.NONBREAKING_PREFIXES:
                            self.NONBREAKING_PREFIXES.append(line)
        self.NUMERIC_ONLY_PREFIXES = [w.rpartition(' ')[0] for w in self.NONBREAKING_PREFIXES if self.has_numeric_only(w)]
        if self.lang in ['zh', 'ja', 'ko', 'cjk']:
            cjk_chars = ''
            if self.lang in ['ko', 'cjk']:
                cjk_chars += str(''.join(perluniprops.chars('Hangul')))
            if self.lang in ['zh', 'cjk']:
                cjk_chars += str(''.join(perluniprops.chars('Han')))
            if self.lang in ['ja', 'cjk']:
                cjk_chars += str(''.join(perluniprops.chars('Hiragana')))
                cjk_chars += str(''.join(perluniprops.chars('Katakana')))
                cjk_chars += str(''.join(perluniprops.chars('Han')))
            self.IsAlpha += cjk_chars
            self.IsAlnum += cjk_chars
            self.PAD_NOT_ISALNUM = (re.compile("([^{}\\s\\.'\\`\\,\\-])".format(self.IsAlnum)), ' \\1 ')
            self.AGGRESSIVE_HYPHEN_SPLIT = (re.compile('([{alphanum}])\\-(?=[{alphanum}])'.format(alphanum=self.IsAlnum)), '\\1 @-@ ')
            self.INTRATOKEN_SLASHES = (re.compile('([{alphanum}])\\/([{alphanum}])'.format(alphanum=self.IsAlnum)), '$1 \\@\\/\\@ $2')
    
    def replace_multidots(self, text):
        text = re.sub('\\.([\\.]+)', ' DOTMULTI\\1', text)
        dotmulti = re.compile('DOTMULTI\\.')
        while dotmulti.search(text):
            text = re.sub('DOTMULTI\\.([^\\.])', 'DOTDOTMULTI \\1', text)
            text = dotmulti.sub('DOTDOTMULTI', text)
        return text
    
    def restore_multidots(self, text):
        dotmulti = re.compile('DOTDOTMULTI')
        while dotmulti.search(text):
            text = dotmulti.sub('DOTMULTI.', text)
        return re.sub('DOTMULTI', '.', text)
    
    def islower(self, text):
        return not set(text).difference(set(self.IsLower))
    
    def isanyalpha(self, text):
        return any(set(text).intersection(set(self.IsAlpha)))
    
    def has_numeric_only(self, text):
        return bool(re.search('[\\s]+(\\#NUMERIC_ONLY\\#)', text))
    
    def handles_nonbreaking_prefixes(self, text):
        tokens = text.split()
        num_tokens = len(tokens)
        for (i, token) in enumerate(tokens):
            token_ends_with_period = re.search('^(\\S+)\\.$', token)
            if token_ends_with_period:
                prefix = token_ends_with_period.group(1)
                if (('.' in prefix and self.isanyalpha(prefix)) or (prefix in self.NONBREAKING_PREFIXES and prefix not in self.NUMERIC_ONLY_PREFIXES) or (i != num_tokens - 1 and tokens[i + 1] and self.islower(tokens[i + 1][0]))):
                    pass
                elif (prefix in self.NUMERIC_ONLY_PREFIXES and i + 1 < num_tokens and re.search('^[0-9]+', tokens[i + 1])):
                    pass
                else:
                    tokens[i] = prefix + ' .'
        return ' '.join(tokens)
    
    def escape_xml(self, text):
        for (regexp, substitution) in self.MOSES_ESCAPE_XML_REGEXES:
            text = regexp.sub(substitution, text)
        return text
    
    def penn_tokenize(self, text, return_str=False):
        """
        This is a Python port of the Penn treebank tokenizer adapted by the Moses
        machine translation community.
        """
        text = str(text)
        for (regexp, substitution) in self.MOSES_PENN_REGEXES_1:
            text = regexp.sub(substitution, text)
        text = self.handles_nonbreaking_prefixes(text)
        for (regexp, substitution) in self.MOSES_PENN_REGEXES_2:
            text = regexp.sub(substitution, text)
        return (text if return_str else text.split())
    
    def tokenize(self, text, aggressive_dash_splits=False, return_str=False, escape=True, protected_patterns=None):
        """
        Python port of the Moses tokenizer.

            :param tokens: A single string, i.e. sentence text.
            :type tokens: str
            :param aggressive_dash_splits: Option to trigger dash split rules .
            :type aggressive_dash_splits: bool
        """
        text = str(text)
        for (regexp, substitution) in [self.DEDUPLICATE_SPACE, self.ASCII_JUNK]:
            text = regexp.sub(substitution, text)
        if protected_patterns:
            protected_patterns = [re.compile(p, re.IGNORECASE) for p in protected_patterns]
            protected_tokens = [match.group() for protected_pattern in protected_patterns for match in protected_pattern.finditer(text)]
            assert len(protected_tokens) <= 1000
            for (i, token) in sorted(enumerate(protected_tokens), key=lambda pair: len(pair[1]), reverse=True):
                substituition = 'THISISPROTECTED' + str(i).zfill(3)
                text = text.replace(token, substituition)
        text = text.strip()
        '\n        # For Finnish and Swedish, seperate out all "other" special characters.\n        if self.lang in ["fi", "sv"]:\n            # In Finnish and Swedish, the colon can be used inside words\n            # as an apostrophe-like character:\n            # USA:n, 20:een, EU:ssa, USA:s, S:t\n            regexp, substitution = self.FI_SV_COLON_APOSTROPHE\n            text = regexp.sub(substitution, text)\n            # If a colon is not immediately followed by lower-case characters,\n            # separate it out anyway.\n            regexp, substitution = self.FI_SV_COLON_NO_LOWER_FOLLOW\n            text = regexp.sub(substitution, text)\n        else:\n        '
        (regexp, substitution) = self.PAD_NOT_ISALNUM
        text = regexp.sub(substitution, text)
        if aggressive_dash_splits:
            (regexp, substitution) = self.AGGRESSIVE_HYPHEN_SPLIT
            text = regexp.sub(substitution, text)
        text = self.replace_multidots(text)
        for (regexp, substitution) in [self.COMMA_SEPARATE_1, self.COMMA_SEPARATE_2, self.COMMA_SEPARATE_3]:
            text = regexp.sub(substitution, text)
        if self.lang == 'en':
            for (regexp, substitution) in self.ENGLISH_SPECIFIC_APOSTROPHE:
                text = regexp.sub(substitution, text)
        elif self.lang in ['fr', 'it']:
            for (regexp, substitution) in self.FR_IT_SPECIFIC_APOSTROPHE:
                text = regexp.sub(substitution, text)
        else:
            (regexp, substitution) = self.NON_SPECIFIC_APOSTROPHE
            text = regexp.sub(substitution, text)
        text = self.handles_nonbreaking_prefixes(text)
        (regexp, substitution) = self.DEDUPLICATE_SPACE
        text = regexp.sub(substitution, text).strip()
        (regexp, substituition) = self.TRAILING_DOT_APOSTROPHE
        text = regexp.sub(substituition, text)
        if protected_patterns:
            for (i, token) in enumerate(protected_tokens):
                substituition = 'THISISPROTECTED' + str(i).zfill(3)
                text = text.replace(substituition, token)
        text = self.restore_multidots(text)
        if escape:
            text = self.escape_xml(text)
        return (text if return_str else text.split())



class MosesDetokenizer(object):
    """
    This is a Python port of the Moses Detokenizer from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl

    """
    IsAlnum = str(''.join(perluniprops.chars('IsAlnum')))
    IsAlpha = str(''.join(perluniprops.chars('IsAlpha')))
    IsSc = str(''.join(perluniprops.chars('IsSc')))
    AGGRESSIVE_HYPHEN_SPLIT = (re.compile(' \\@\\-\\@ '), '-')
    ONE_SPACE = (re.compile(' {2,}'), ' ')
    UNESCAPE_FACTOR_SEPARATOR = (re.compile('&#124;'), '|')
    UNESCAPE_LEFT_ANGLE_BRACKET = (re.compile('&lt;'), '<')
    UNESCAPE_RIGHT_ANGLE_BRACKET = (re.compile('&gt;'), '>')
    UNESCAPE_DOUBLE_QUOTE = (re.compile('&quot;'), '"')
    UNESCAPE_SINGLE_QUOTE = (re.compile('&apos;'), "'")
    UNESCAPE_SYNTAX_NONTERMINAL_LEFT = (re.compile('&#91;'), '[')
    UNESCAPE_SYNTAX_NONTERMINAL_RIGHT = (re.compile('&#93;'), ']')
    UNESCAPE_AMPERSAND = (re.compile('&amp;'), '&')
    UNESCAPE_FACTOR_SEPARATOR_LEGACY = (re.compile('&bar;'), '|')
    UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY = (re.compile('&bra;'), '[')
    UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY = (re.compile('&ket;'), ']')
    MOSES_UNESCAPE_XML_REGEXES = [UNESCAPE_FACTOR_SEPARATOR_LEGACY, UNESCAPE_FACTOR_SEPARATOR, UNESCAPE_LEFT_ANGLE_BRACKET, UNESCAPE_RIGHT_ANGLE_BRACKET, UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY, UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY, UNESCAPE_DOUBLE_QUOTE, UNESCAPE_SINGLE_QUOTE, UNESCAPE_SYNTAX_NONTERMINAL_LEFT, UNESCAPE_SYNTAX_NONTERMINAL_RIGHT, UNESCAPE_AMPERSAND]
    FINNISH_MORPHSET_1 = ['N', 'n', 'A', 'a', 'Ä', 'ä', 'ssa', 'Ssa', 'ssä', 'Ssä', 'sta', 'stä', 'Sta', 'Stä', 'hun', 'Hun', 'hyn', 'Hyn', 'han', 'Han', 'hän', 'Hän', 'hön', 'Hön', 'un', 'Un', 'yn', 'Yn', 'an', 'An', 'än', 'Än', 'ön', 'Ön', 'seen', 'Seen', 'lla', 'Lla', 'llä', 'Llä', 'lta', 'Lta', 'ltä', 'Ltä', 'lle', 'Lle', 'ksi', 'Ksi', 'kse', 'Kse', 'tta', 'Tta', 'ine', 'Ine']
    FINNISH_MORPHSET_2 = ['ni', 'si', 'mme', 'nne', 'nsa']
    FINNISH_MORPHSET_3 = ['ko', 'kö', 'han', 'hän', 'pa', 'pä', 'kaan', 'kään', 'kin']
    FINNISH_REGEX = re.compile('^({})({})?({})$'.format('|'.join(FINNISH_MORPHSET_1), '|'.join(FINNISH_MORPHSET_2), '|'.join(FINNISH_MORPHSET_3)))
    IS_CURRENCY_SYMBOL = re.compile('^[{}\\(\\[\\{{\\¿\\¡]+$'.format(IsSc))
    IS_ENGLISH_CONTRACTION = re.compile("^['][{}]".format(IsAlpha))
    IS_FRENCH_CONRTACTION = re.compile("[{}][']$".format(IsAlpha))
    STARTS_WITH_ALPHA = re.compile('^[{}]'.format(IsAlpha))
    IS_PUNCT = re.compile('^[\\,\\.\\?\\!\\:\\;\\\\\\%\\}\\]\\)]+$')
    IS_OPEN_QUOTE = re.compile('^[\\\'\\"„“`]+$')
    
    def __init__(self, lang='en'):
        super(MosesDetokenizer, self).__init__()
        self.lang = lang
    
    def unescape_xml(self, text):
        for (regexp, substitution) in self.MOSES_UNESCAPE_XML_REGEXES:
            text = regexp.sub(substitution, text)
        return text
    
    def tokenize(self, tokens, return_str=True, unescape=True):
        """
        Python port of the Moses detokenizer.
        :param tokens: A list of strings, i.e. tokenized text.
        :type tokens: list(str)
        :return: str
        """
        text = ' {} '.format(' '.join(tokens))
        text = str(text)
        (regexp, substitution) = self.AGGRESSIVE_HYPHEN_SPLIT
        text = regexp.sub(substitution, text)
        if unescape:
            text = self.unescape_xml(text)
        quote_counts = {"'": 0, '"': 0, '``': 0, '`': 0, "''": 0}
        prepend_space = ' '
        detokenized_text = ''
        tokens = text.split()
        for (i, token) in enumerate(iter(tokens)):
            if (is_cjk(token[0]) and self.lang != 'ko'):
                if (i > 0 and is_cjk(tokens[i - 1][-1])):
                    detokenized_text += token
                else:
                    detokenized_text += prepend_space + token
                prepend_space = ' '
            elif self.IS_CURRENCY_SYMBOL.search(token):
                detokenized_text += prepend_space + token
                prepend_space = ''
            elif self.IS_PUNCT.search(token):
                if (self.lang == 'fr' and re.search('^[\\?\\!\\:\\;\\\\\\%]$', token)):
                    detokenized_text += ' '
                detokenized_text += token
                prepend_space = ' '
            elif (self.lang == 'en' and i > 0 and self.IS_ENGLISH_CONTRACTION.search(token)):
                detokenized_text += token
                prepend_space = ' '
            elif (self.lang == 'cs' and i > 1 and re.search('^[0-9]+$', tokens[-2]) and re.search('^[.,]$', tokens[-1]) and re.search('^[0-9]+$', token)):
                detokenized_text += token
                prepend_space = ' '
            elif (self.lang in ['fr', 'it', 'ga'] and i <= len(tokens) - 2 and self.IS_FRENCH_CONRTACTION.search(token) and self.STARTS_WITH_ALPHA.search(tokens[i + 1])):
                detokenized_text += prepend_space + token
                prepend_space = ''
            elif (self.lang == 'cs' and i <= len(tokens) - 3 and self.IS_FRENCH_CONRTACTION.search(token) and re.search('^[-–]$', tokens[i + 1]) and re.search('^li$|^mail.*', tokens[i + 2], re.IGNORECASE)):
                detokenized_text += prepend_space + token + tokens[i + 1]
                next(tokens, None)
                prepend_space = ''
            elif self.IS_OPEN_QUOTE.search(token):
                normalized_quo = token
                if re.search('^[„“”]+$', token):
                    normalized_quo = '"'
                quote_counts[normalized_quo] = quote_counts.get(normalized_quo, 0)
                if (self.lang == 'cs' and token == '„'):
                    quote_counts[normalized_quo] = 0
                if (self.lang == 'cs' and token == '“'):
                    quote_counts[normalized_quo] = 1
                if quote_counts[normalized_quo] % 2 == 0:
                    if (self.lang == 'en' and token == "'" and i > 0 and re.search('[s]$', tokens[i - 1])):
                        detokenized_text += token
                        prepend_space = ' '
                    else:
                        detokenized_text += prepend_space + token
                        prepend_space = ''
                        quote_counts[normalized_quo] += 1
                else:
                    detokenized_text += token
                    prepend_space = ' '
                    quote_counts[normalized_quo] += 1
            elif (self.lang == 'fi' and re.search(':$', tokens[i - 1]) and self.FINNISH_REGEX.search(token)):
                detokenized_text += prepend_space + token
                prepend_space = ' '
            else:
                detokenized_text += prepend_space + token
                prepend_space = ' '
        (regexp, substitution) = self.ONE_SPACE
        detokenized_text = regexp.sub(substitution, detokenized_text)
        detokenized_text = detokenized_text.strip()
        return (detokenized_text if return_str else detokenized_text.split())
    
    def detokenize(self, tokens, return_str=True, unescape=True):
        """Duck-typing the abstract *tokenize()*."""
        return self.tokenize(tokens, return_str, unescape)

__all__ = ['MosesTokenizer', 'MosesDetokenizer']

