import re
import regex
from itertools import chain


class MosesPunctNormalizer:
    """
    This is a Python port of the Moses punctuation normalizer from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/normalize-punctuation.perl
    """
    EXTRA_WHITESPACE = [('\\r', ''), ('\\(', ' ('), ('\\)', ') '), (' +', ' '), ('\\) ([.!:?;,])', ')\\g<1>'), ('\\( ', '('), (' \\)', ')'), ('(\\d) %', '\\g<1>%'), (' :', ':'), (' ;', ';')]
    NORMALIZE_UNICODE_IF_NOT_PENN = [('`', "'"), ("''", ' " ')]
    NORMALIZE_UNICODE = [('„', '"'), ('“', '"'), ('”', '"'), ('–', '-'), ('—', ' - '), (' +', ' '), ('´', "'"), ('([a-zA-Z])‘([a-zA-Z])', "\\g<1>'\\g<2>"), ('([a-zA-Z])’([a-zA-Z])', "\\g<1>'\\g<2>"), ('‘', "'"), ('‚', "'"), ('’', "'"), ("''", '"'), ('´´', '"'), ('…', '...')]
    FRENCH_QUOTES = [('\xa0«\xa0', '"'), ('«\xa0', '"'), ('«', '"'), ('\xa0»\xa0', '"'), ('\xa0»', '"'), ('»', '"')]
    HANDLE_PSEUDO_SPACES = [('\xa0%', '%'), ('nº\xa0', 'nº '), ('\xa0:', ':'), ('\xa0ºC', ' ºC'), ('\xa0cm', ' cm'), ('\xa0\\?', '?'), ('\xa0\\!', '!'), ('\xa0;', ';'), (',\xa0', ', '), (' +', ' ')]
    EN_QUOTATION_FOLLOWED_BY_COMMA = [('"([,.]+)', '\\g<1>"')]
    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [(',"', '",'), ('(\\.+)"(\\s*[^<])', '"\\g<1>\\g<2>')]
    DE_ES_CZ_CS_FR = [('(\\d)\xa0(\\d)', '\\g<1>,\\g<2>')]
    OTHER = [('(\\d)\xa0(\\d)', '\\g<1>.\\g<2>')]
    REPLACE_UNICODE_PUNCTUATION = [('，', ','), ('。\\s*', '. '), ('、', ','), ('”', '"'), ('“', '"'), ('∶', ':'), ('：', ':'), ('？', '?'), ('《', '"'), ('》', '"'), ('）', ')'), ('！', '!'), ('（', '('), ('；', ';'), ('」', '"'), ('「', '"'), ('０', '0'), ('１', '1'), ('２', '2'), ('３', '3'), ('４', '4'), ('５', '5'), ('６', '6'), ('７', '7'), ('８', '8'), ('９', '9'), ('．\\s*', '. '), ('～', '~'), ('’', "'"), ('…', '...'), ('━', '-'), ('〈', '<'), ('〉', '>'), ('【', '['), ('】', ']'), ('％', '%')]
    
    def __init__(self, lang='en', penn=True, norm_quote_commas=True, norm_numbers=True, pre_replace_unicode_punct=False, post_remove_control_chars=False, perl_parity=False):
        """
        :param language: The two-letter language code.
        :type lang: str
        :param penn: Normalize Penn Treebank style quotations.
        :type penn: bool
        :param norm_quote_commas: Normalize quotations and commas
        :type norm_quote_commas: bool
        :param norm_numbers: Normalize numbers
        :type norm_numbers: bool
        :param perl_parity: exact parity with perl script
        :type perl_parity: bool
        """
        if perl_parity:
            self.NORMALIZE_UNICODE[11] = ('’', '"')
            self.FRENCH_QUOTES[0] = ('\xa0«\xa0', ' "')
            self.FRENCH_QUOTES[3] = ('\xa0»\xa0', '" ')
        self.substitutions = [self.EXTRA_WHITESPACE, self.NORMALIZE_UNICODE, self.FRENCH_QUOTES, self.HANDLE_PSEUDO_SPACES]
        if penn:
            self.substitutions.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)
        if norm_quote_commas:
            if lang == 'en':
                self.substitutions.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ['de', 'es', 'fr']:
                self.substitutions.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)
        if norm_numbers:
            if lang in ['de', 'es', 'cz', 'cs', 'fr']:
                self.substitutions.append(self.DE_ES_CZ_CS_FR)
            else:
                self.substitutions.append(self.OTHER)
        self.substitutions = list(chain(*self.substitutions))
        self.pre_replace_unicode_punct = pre_replace_unicode_punct
        self.post_remove_control_chars = post_remove_control_chars
    
    def normalize(self, text):
        """
        Returns a string with normalized punctuation.
        """
        if self.pre_replace_unicode_punct:
            text = self.replace_unicode_punct(text)
        for (regexp, substitution) in self.substitutions:
            text = re.sub(regexp, substitution, str(text))
        if self.post_remove_control_chars:
            text = self.remove_control_chars(text)
        return text.strip()
    
    def replace_unicode_punct(self, text):
        for (regexp, substitution) in self.REPLACE_UNICODE_PUNCTUATION:
            text = re.sub(regexp, substitution, str(text))
        return text
    
    def remove_control_chars(self, text):
        return regex.sub('\\p{C}', '', text)


