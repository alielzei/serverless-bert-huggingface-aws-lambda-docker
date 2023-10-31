import importlib
import logging
import unicodedata
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import Generator, List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder
from .constant import ENCODING_MARKS, IANA_SUPPORTED_SIMILAR, RE_POSSIBLE_ENCODING_INDICATION, UNICODE_RANGES_COMBINED, UNICODE_SECONDARY_RANGE_KEYWORD, UTF8_MAXIMAL_ALLOCATION

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_accentuated(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_accentuated', 'is_accentuated(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def remove_accent(character: str) -> str:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.remove_accent', 'remove_accent(character)', {'unicodedata': unicodedata, 'List': List, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def unicode_range(character: str) -> Optional[str]:
    """
    Retrieve the Unicode range official name from a single character.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.unicode_range', 'unicode_range(character)', {'UNICODE_RANGES_COMBINED': UNICODE_RANGES_COMBINED, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character, 'Optional': Optional, 'str': str}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_latin(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_latin', 'is_latin(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_punctuation(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_punctuation', 'is_punctuation(character)', {'unicodedata': unicodedata, 'Optional': Optional, 'unicode_range': unicode_range, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_symbol(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_symbol', 'is_symbol(character)', {'unicodedata': unicodedata, 'Optional': Optional, 'unicode_range': unicode_range, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_emoticon(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_emoticon', 'is_emoticon(character)', {'Optional': Optional, 'unicode_range': unicode_range, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_separator(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_separator', 'is_separator(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_case_variable(character: str) -> bool:
    return character.islower() != character.isupper()

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_cjk(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_cjk', 'is_cjk(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_hiragana(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_hiragana', 'is_hiragana(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_katakana(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_katakana', 'is_katakana(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_hangul(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_hangul', 'is_hangul(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_thai(character: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.is_thai', 'is_thai(character)', {'unicodedata': unicodedata, 'lru_cache': lru_cache, 'UTF8_MAXIMAL_ALLOCATION': UTF8_MAXIMAL_ALLOCATION, 'character': character}, 1)

@lru_cache(maxsize=len(UNICODE_RANGES_COMBINED))
def is_unicode_range_secondary(range_name: str) -> bool:
    return any((keyword in range_name for keyword in UNICODE_SECONDARY_RANGE_KEYWORD))

@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_unprintable(character: str) -> bool:
    return (character.isspace() is False and character.isprintable() is False and character != '\x1a' and character != '\ufeff')

def any_specified_encoding(sequence: bytes, search_zone: int = 8192) -> Optional[str]:
    """
    Extract using ASCII-only decoder any specified encoding in the first n-bytes.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.any_specified_encoding', 'any_specified_encoding(sequence, search_zone=8192)', {'List': List, 'findall': findall, 'RE_POSSIBLE_ENCODING_INDICATION': RE_POSSIBLE_ENCODING_INDICATION, 'aliases': aliases, 'sequence': sequence, 'search_zone': search_zone, 'Optional': Optional, 'str': str}, 1)

@lru_cache(maxsize=128)
def is_multi_byte_encoding(name: str) -> bool:
    """
    Verify is a specific encoding is a multi byte one based on it IANA name
    """
    return (name in {'utf_8', 'utf_8_sig', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_32', 'utf_32_le', 'utf_32_be', 'utf_7'} or issubclass(importlib.import_module('encodings.{}'.format(name)).IncrementalDecoder, MultibyteIncrementalDecoder))

def identify_sig_or_bom(sequence: bytes) -> Tuple[(Optional[str], bytes)]:
    """
    Identify and extract SIG/BOM in given sequence.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.identify_sig_or_bom', 'identify_sig_or_bom(sequence)', {'ENCODING_MARKS': ENCODING_MARKS, 'Union': Union, 'List': List, 'sequence': sequence, 'Tuple': Tuple}, 2)

def should_strip_sig_or_bom(iana_encoding: str) -> bool:
    return iana_encoding not in {'utf_16', 'utf_32'}

def iana_name(cp_name: str, strict: bool = True) -> str:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.iana_name', 'iana_name(cp_name, strict=True)', {'aliases': aliases, 'cp_name': cp_name, 'strict': strict}, 1)

def range_scan(decoded_sequence: str) -> List[str]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.range_scan', 'range_scan(decoded_sequence)', {'Set': Set, 'Optional': Optional, 'unicode_range': unicode_range, 'decoded_sequence': decoded_sequence, 'List': List, 'str': str}, 1)

def cp_similarity(iana_name_a: str, iana_name_b: str) -> float:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.utils.cp_similarity', 'cp_similarity(iana_name_a, iana_name_b)', {'is_multi_byte_encoding': is_multi_byte_encoding, 'importlib': importlib, 'IncrementalDecoder': IncrementalDecoder, 'iana_name_a': iana_name_a, 'iana_name_b': iana_name_b}, 1)

def is_cp_similar(iana_name_a: str, iana_name_b: str) -> bool:
    """
    Determine if two code page are at least 80% similar. IANA_SUPPORTED_SIMILAR dict was generated using
    the function cp_similarity.
    """
    return (iana_name_a in IANA_SUPPORTED_SIMILAR and iana_name_b in IANA_SUPPORTED_SIMILAR[iana_name_a])

def set_logging_handler(name: str = 'charset_normalizer', level: int = logging.INFO, format_string: str = '%(asctime)s | %(levelname)s | %(message)s') -> None:
    import custom_funtemplate
    custom_funtemplate.rewrite_template('charset_normalizer.utils.set_logging_handler', "set_logging_handler(name='charset_normalizer', level=logging.INFO, format_string='%(asctime)s | %(levelname)s | %(message)s')", {'logging': logging, 'name': name, 'level': level, 'format_string': format_string}, 0)

def cut_sequence_chunks(sequences: bytes, encoding_iana: str, offsets: range, chunk_size: int, bom_or_sig_available: bool, strip_sig_or_bom: bool, sig_payload: bytes, is_multi_byte_decoder: bool, decoded_payload: Optional[str] = None) -> Generator[(str, None, None)]:
    import custom_funtemplate
    custom_funtemplate.rewrite_template('charset_normalizer.utils.cut_sequence_chunks', 'cut_sequence_chunks(sequences, encoding_iana, offsets, chunk_size, bom_or_sig_available, strip_sig_or_bom, sig_payload, is_multi_byte_decoder, decoded_payload=None)', {'sequences': sequences, 'encoding_iana': encoding_iana, 'offsets': offsets, 'chunk_size': chunk_size, 'bom_or_sig_available': bom_or_sig_available, 'strip_sig_or_bom': strip_sig_or_bom, 'sig_payload': sig_payload, 'is_multi_byte_decoder': is_multi_byte_decoder, 'decoded_payload': decoded_payload, 'Optional': Optional, 'str': str, 'Generator': Generator}, 0)

