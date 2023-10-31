import importlib
from codecs import IncrementalDecoder
from collections import Counter
from functools import lru_cache
from typing import Counter as TypeCounter, Dict, List, Optional, Tuple
from .constant import FREQUENCIES, KO_NAMES, LANGUAGE_SUPPORTED_COUNT, TOO_SMALL_SEQUENCE, ZH_NAMES
from .md import is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import is_accentuated, is_latin, is_multi_byte_encoding, is_unicode_range_secondary, unicode_range

def encoding_unicode_range(iana_name: str) -> List[str]:
    """
    Return associated unicode ranges in a single byte code page.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.encoding_unicode_range', 'encoding_unicode_range(iana_name)', {'is_multi_byte_encoding': is_multi_byte_encoding, 'IOError': IOError, 'importlib': importlib, 'IncrementalDecoder': IncrementalDecoder, 'Dict': Dict, 'Optional': Optional, 'unicode_range': unicode_range, 'is_unicode_range_secondary': is_unicode_range_secondary, 'iana_name': iana_name, 'List': List, 'str': str}, 1)

def unicode_range_languages(primary_range: str) -> List[str]:
    """
    Return inferred languages used with a unicode range.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.unicode_range_languages', 'unicode_range_languages(primary_range)', {'List': List, 'FREQUENCIES': FREQUENCIES, 'unicode_range': unicode_range, 'primary_range': primary_range, 'List': List, 'str': str}, 1)

@lru_cache()
def encoding_languages(iana_name: str) -> List[str]:
    """
    Single-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.encoding_languages', 'encoding_languages(iana_name)', {'List': List, 'encoding_unicode_range': encoding_unicode_range, 'Optional': Optional, 'unicode_range_languages': unicode_range_languages, 'lru_cache': lru_cache, 'iana_name': iana_name, 'List': List, 'str': str}, 1)

@lru_cache()
def mb_encoding_languages(iana_name: str) -> List[str]:
    """
    Multi-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.mb_encoding_languages', 'mb_encoding_languages(iana_name)', {'ZH_NAMES': ZH_NAMES, 'KO_NAMES': KO_NAMES, 'lru_cache': lru_cache, 'iana_name': iana_name, 'List': List, 'str': str}, 1)

@lru_cache(maxsize=LANGUAGE_SUPPORTED_COUNT)
def get_target_features(language: str) -> Tuple[(bool, bool)]:
    """
    Determine main aspects from a supported language if it contains accents and if is pure Latin.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.get_target_features', 'get_target_features(language)', {'FREQUENCIES': FREQUENCIES, 'is_accentuated': is_accentuated, 'is_latin': is_latin, 'lru_cache': lru_cache, 'LANGUAGE_SUPPORTED_COUNT': LANGUAGE_SUPPORTED_COUNT, 'language': language, 'Tuple': Tuple}, 2)

def alphabet_languages(characters: List[str], ignore_non_latin: bool = False) -> List[str]:
    """
    Return associated languages associated to given characters.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.alphabet_languages', 'alphabet_languages(characters, ignore_non_latin=False)', {'List': List, 'Tuple': Tuple, 'is_accentuated': is_accentuated, 'FREQUENCIES': FREQUENCIES, 'get_target_features': get_target_features, 'characters': characters, 'ignore_non_latin': ignore_non_latin, 'List': List, 'str': str, 'List': List, 'str': str}, 1)

def characters_popularity_compare(language: str, ordered_characters: List[str]) -> float:
    """
    Determine if a ordered characters list (by occurrence from most appearance to rarest) match a particular language.
    The result is a ratio between 0. (absolutely no correspondence) and 1. (near perfect fit).
    Beware that is function is not strict on the match in order to ease the detection. (Meaning close match is 1.)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.characters_popularity_compare', 'characters_popularity_compare(language, ordered_characters)', {'FREQUENCIES': FREQUENCIES, 'List': List, 'language': language, 'ordered_characters': ordered_characters, 'List': List, 'str': str}, 1)

def alpha_unicode_split(decoded_sequence: str) -> List[str]:
    """
    Given a decoded text sequence, return a list of str. Unicode range / alphabet separation.
    Ex. a text containing English/Latin with a bit a Hebrew will return two items in the resulting list;
    One containing the latin letters and the other hebrew.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.alpha_unicode_split', 'alpha_unicode_split(decoded_sequence)', {'Dict': Dict, 'Optional': Optional, 'unicode_range': unicode_range, 'is_suspiciously_successive_range': is_suspiciously_successive_range, 'decoded_sequence': decoded_sequence, 'List': List, 'str': str}, 1)

def merge_coherence_ratios(results: List[CoherenceMatches]) -> CoherenceMatches:
    """
    This function merge results previously given by the function coherence_ratio.
    The return type is the same as coherence_ratio.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.merge_coherence_ratios', 'merge_coherence_ratios(results)', {'Dict': Dict, 'List': List, 'results': results, 'List': List, 'CoherenceMatches': CoherenceMatches}, 1)

def filter_alt_coherence_matches(results: CoherenceMatches) -> CoherenceMatches:
    """
    We shall NOT return "English—" in CoherenceMatches because it is an alternative
    of "English". This function only keeps the best match and remove the em-dash in it.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.filter_alt_coherence_matches', 'filter_alt_coherence_matches(results)', {'Dict': Dict, 'List': List, 'CoherenceMatches': CoherenceMatches, 'results': results}, 1)

@lru_cache(maxsize=2048)
def coherence_ratio(decoded_sequence: str, threshold: float = 0.1, lg_inclusion: Optional[str] = None) -> CoherenceMatches:
    """
    Detect ANY language that can be identified in given sequence. The sequence will be analysed by layers.
    A layer = Character extraction by alphabets/ranges.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.cd.coherence_ratio', 'coherence_ratio(decoded_sequence, threshold=0.1, lg_inclusion=None)', {'List': List, 'Tuple': Tuple, 'alpha_unicode_split': alpha_unicode_split, 'TypeCounter': TypeCounter, 'Counter': Counter, 'TOO_SMALL_SEQUENCE': TOO_SMALL_SEQUENCE, 'alphabet_languages': alphabet_languages, 'characters_popularity_compare': characters_popularity_compare, 'filter_alt_coherence_matches': filter_alt_coherence_matches, 'lru_cache': lru_cache, 'decoded_sequence': decoded_sequence, 'threshold': threshold, 'lg_inclusion': lg_inclusion, 'Optional': Optional, 'str': str}, 1)

