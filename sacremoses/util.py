from itertools import tee, zip_longest
from xml.sax.saxutils import escape, unescape
from joblib import Parallel, delayed
from tqdm import tqdm


class CJKChars(object):
    """
    An object that enumerates the code points of the CJK characters as listed on
    http://en.wikipedia.org/wiki/Basic_Multilingual_Plane#Basic_Multilingual_Plane
    """
    Hangul_Jamo = (4352, 4607)
    CJK_Radicals = (11904, 42191)
    Phags_Pa = (43072, 43135)
    Hangul_Syllables = (44032, 55215)
    CJK_Compatibility_Ideographs = (63744, 64255)
    CJK_Compatibility_Forms = (65072, 65103)
    Katakana_Hangul_Halfwidth = (65381, 65500)
    Ideographic_Symbols_And_Punctuation = (94176, 94207)
    Tangut = (94208, 101119)
    Kana_Supplement = (110592, 110895)
    Nushu = (110960, 111359)
    Supplementary_Ideographic_Plane = (131072, 196607)
    ranges = [Hangul_Jamo, CJK_Radicals, Phags_Pa, Hangul_Syllables, CJK_Compatibility_Ideographs, CJK_Compatibility_Forms, Katakana_Hangul_Halfwidth, Tangut, Kana_Supplement, Nushu, Supplementary_Ideographic_Plane]

_CJKChars_ranges = CJKChars().ranges

def is_cjk(character):
    """
    This checks for CJK character.

        >>> CJKChars().ranges
        [(4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215), (63744, 64255), (65072, 65103), (65381, 65500), (94208, 101119), (110592, 110895), (110960, 111359), (131072, 196607)]
        >>> is_cjk('㏾')
        True
        >>> is_cjk('﹟')
        False

    :param character: The character that needs to be checked.
    :type character: char
    :return: bool
    """
    char = ord(character)
    for (start, end) in _CJKChars_ranges:
        if char < end:
            return char > start
    return False

def xml_escape(text):
    """
    This function transforms the input text into an "escaped" version suitable
    for well-formed XML formatting.
    Note that the default xml.sax.saxutils.escape() function don't escape
    some characters that Moses does so we have to manually add them to the
    entities dictionary.

        >>> input_str = ''')| & < > ' " ] ['''
        >>> expected_output =  ''')| &amp; &lt; &gt; ' " ] ['''
        >>> escape(input_str) == expected_output
        True
        >>> xml_escape(input_str)
        ')&#124; &amp; &lt; &gt; &apos; &quot; &#93; &#91;'

    :param text: The text that needs to be escaped.
    :type text: str
    :rtype: str
    """
    return escape(text, entities={"'": '&apos;', '"': '&quot;', '|': '&#124;', '[': '&#91;', ']': '&#93;'})

def xml_unescape(text):
    """
    This function transforms the "escaped" version suitable
    for well-formed XML formatting into humanly-readable string.
    Note that the default xml.sax.saxutils.unescape() function don't unescape
    some characters that Moses does so we have to manually add them to the
    entities dictionary.

        >>> from xml.sax.saxutils import unescape
        >>> s = ')&#124; &amp; &lt; &gt; &apos; &quot; &#93; &#91;'
        >>> expected = ''')| & < > ' " ] ['''
        >>> xml_unescape(s) == expected
        True

    :param text: The text that needs to be unescaped.
    :type text: str
    :rtype: str
    """
    return unescape(text, entities={'&apos;': "'", '&quot;': '"', '&#124;': '|', '&#91;': '[', '&#93;': ']'})

def pairwise(iterable):
    """
    From https://docs.python.org/3/library/itertools.html#recipes
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.util.pairwise', 'pairwise(iterable)', {'tee': tee, 'iterable': iterable}, 1)

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks
    from https://stackoverflow.com/a/16789869/610569
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def parallelize_preprocess(func, iterator, processes, progress_bar=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.util.parallelize_preprocess', 'parallelize_preprocess(func, iterator, processes, progress_bar=False)', {'tqdm': tqdm, 'Parallel': Parallel, 'delayed': delayed, 'func': func, 'iterator': iterator, 'processes': processes, 'progress_bar': progress_bar}, 1)

