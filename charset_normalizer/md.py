from functools import lru_cache
from logging import getLogger
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, TRACE, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import is_accentuated, is_case_variable, is_cjk, is_emoticon, is_hangul, is_hiragana, is_katakana, is_latin, is_punctuation, is_separator, is_symbol, is_thai, is_unprintable, remove_accent, unicode_range


class MessDetectorPlugin:
    """
    Base abstract class used for mess detection plugins.
    All detectors MUST extend and implement given methods.
    """
    
    def eligible(self, character: str) -> bool:
        """
        Determine if given character should be fed in.
        """
        raise NotImplementedError
    
    def feed(self, character: str) -> None:
        """
        The main routine to be executed upon character.
        Insert the logic in witch the text would be considered chaotic.
        """
        raise NotImplementedError
    
    def reset(self) -> None:
        """
        Permit to reset the plugin to the initial state.
        """
        raise NotImplementedError
    
    @property
    def ratio(self) -> float:
        """
        Compute the chaos ratio based on what your feed() has seen.
        Must NOT be lower than 0.; No restriction gt 0.
        """
        raise NotImplementedError



class TooManySymbolOrPunctuationPlugin(MessDetectorPlugin):
    
    def __init__(self) -> None:
        self._punctuation_count: int = 0
        self._symbol_count: int = 0
        self._character_count: int = 0
        self._last_printable_char: Optional[str] = None
        self._frenzy_symbol_in_word: bool = False
    
    def eligible(self, character: str) -> bool:
        return character.isprintable()
    
    def feed(self, character: str) -> None:
        self._character_count += 1
        if (character != self._last_printable_char and character not in COMMON_SAFE_ASCII_CHARACTERS):
            if is_punctuation(character):
                self._punctuation_count += 1
            elif (character.isdigit() is False and is_symbol(character) and is_emoticon(character) is False):
                self._symbol_count += 2
        self._last_printable_char = character
    
    def reset(self) -> None:
        self._punctuation_count = 0
        self._character_count = 0
        self._symbol_count = 0
    
    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        ratio_of_punctuation: float = (self._punctuation_count + self._symbol_count) / self._character_count
        return (ratio_of_punctuation if ratio_of_punctuation >= 0.3 else 0.0)



class TooManyAccentuatedPlugin(MessDetectorPlugin):
    
    def __init__(self) -> None:
        self._character_count: int = 0
        self._accentuated_count: int = 0
    
    def eligible(self, character: str) -> bool:
        return character.isalpha()
    
    def feed(self, character: str) -> None:
        self._character_count += 1
        if is_accentuated(character):
            self._accentuated_count += 1
    
    def reset(self) -> None:
        self._character_count = 0
        self._accentuated_count = 0
    
    @property
    def ratio(self) -> float:
        if (self._character_count == 0 or self._character_count < 8):
            return 0.0
        ratio_of_accentuation: float = self._accentuated_count / self._character_count
        return (ratio_of_accentuation if ratio_of_accentuation >= 0.35 else 0.0)



class UnprintablePlugin(MessDetectorPlugin):
    
    def __init__(self) -> None:
        self._unprintable_count: int = 0
        self._character_count: int = 0
    
    def eligible(self, character: str) -> bool:
        return True
    
    def feed(self, character: str) -> None:
        if is_unprintable(character):
            self._unprintable_count += 1
        self._character_count += 1
    
    def reset(self) -> None:
        self._unprintable_count = 0
    
    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        return self._unprintable_count * 8 / self._character_count



class SuspiciousDuplicateAccentPlugin(MessDetectorPlugin):
    
    def __init__(self) -> None:
        self._successive_count: int = 0
        self._character_count: int = 0
        self._last_latin_character: Optional[str] = None
    
    def eligible(self, character: str) -> bool:
        return (character.isalpha() and is_latin(character))
    
    def feed(self, character: str) -> None:
        self._character_count += 1
        if (self._last_latin_character is not None and is_accentuated(character) and is_accentuated(self._last_latin_character)):
            if (character.isupper() and self._last_latin_character.isupper()):
                self._successive_count += 1
            if remove_accent(character) == remove_accent(self._last_latin_character):
                self._successive_count += 1
        self._last_latin_character = character
    
    def reset(self) -> None:
        self._successive_count = 0
        self._character_count = 0
        self._last_latin_character = None
    
    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        return self._successive_count * 2 / self._character_count



class SuspiciousRange(MessDetectorPlugin):
    
    def __init__(self) -> None:
        self._suspicious_successive_range_count: int = 0
        self._character_count: int = 0
        self._last_printable_seen: Optional[str] = None
    
    def eligible(self, character: str) -> bool:
        return character.isprintable()
    
    def feed(self, character: str) -> None:
        self._character_count += 1
        if (character.isspace() or is_punctuation(character) or character in COMMON_SAFE_ASCII_CHARACTERS):
            self._last_printable_seen = None
            return
        if self._last_printable_seen is None:
            self._last_printable_seen = character
            return
        unicode_range_a: Optional[str] = unicode_range(self._last_printable_seen)
        unicode_range_b: Optional[str] = unicode_range(character)
        if is_suspiciously_successive_range(unicode_range_a, unicode_range_b):
            self._suspicious_successive_range_count += 1
        self._last_printable_seen = character
    
    def reset(self) -> None:
        self._character_count = 0
        self._suspicious_successive_range_count = 0
        self._last_printable_seen = None
    
    @property
    def ratio(self) -> float:
        if self._character_count <= 24:
            return 0.0
        ratio_of_suspicious_range_usage: float = self._suspicious_successive_range_count * 2 / self._character_count
        return ratio_of_suspicious_range_usage



class SuperWeirdWordPlugin(MessDetectorPlugin):
    
    def __init__(self) -> None:
        self._word_count: int = 0
        self._bad_word_count: int = 0
        self._foreign_long_count: int = 0
        self._is_current_word_bad: bool = False
        self._foreign_long_watch: bool = False
        self._character_count: int = 0
        self._bad_character_count: int = 0
        self._buffer: str = ''
        self._buffer_accent_count: int = 0
    
    def eligible(self, character: str) -> bool:
        return True
    
    def feed(self, character: str) -> None:
        if character.isalpha():
            self._buffer += character
            if is_accentuated(character):
                self._buffer_accent_count += 1
            if (self._foreign_long_watch is False and ((is_latin(character) is False or is_accentuated(character))) and is_cjk(character) is False and is_hangul(character) is False and is_katakana(character) is False and is_hiragana(character) is False and is_thai(character) is False):
                self._foreign_long_watch = True
            return
        if not self._buffer:
            return
        if (((character.isspace() or is_punctuation(character) or is_separator(character))) and self._buffer):
            self._word_count += 1
            buffer_length: int = len(self._buffer)
            self._character_count += buffer_length
            if buffer_length >= 4:
                if self._buffer_accent_count / buffer_length > 0.34:
                    self._is_current_word_bad = True
                if (is_accentuated(self._buffer[-1]) and self._buffer[-1].isupper() and all((_.isupper() for _ in self._buffer)) is False):
                    self._foreign_long_count += 1
                    self._is_current_word_bad = True
            if (buffer_length >= 24 and self._foreign_long_watch):
                camel_case_dst = [i for (c, i) in zip(self._buffer, range(0, buffer_length)) if c.isupper()]
                probable_camel_cased: bool = False
                if (camel_case_dst and len(camel_case_dst) / buffer_length <= 0.3):
                    probable_camel_cased = True
                if not probable_camel_cased:
                    self._foreign_long_count += 1
                    self._is_current_word_bad = True
            if self._is_current_word_bad:
                self._bad_word_count += 1
                self._bad_character_count += len(self._buffer)
                self._is_current_word_bad = False
            self._foreign_long_watch = False
            self._buffer = ''
            self._buffer_accent_count = 0
        elif (character not in {'<', '>', '-', '=', '~', '|', '_'} and character.isdigit() is False and is_symbol(character)):
            self._is_current_word_bad = True
            self._buffer += character
    
    def reset(self) -> None:
        self._buffer = ''
        self._is_current_word_bad = False
        self._foreign_long_watch = False
        self._bad_word_count = 0
        self._word_count = 0
        self._character_count = 0
        self._bad_character_count = 0
        self._foreign_long_count = 0
    
    @property
    def ratio(self) -> float:
        if (self._word_count <= 10 and self._foreign_long_count == 0):
            return 0.0
        return self._bad_character_count / self._character_count



class CjkInvalidStopPlugin(MessDetectorPlugin):
    """
    GB(Chinese) based encoding often render the stop incorrectly when the content does not fit and
    can be easily detected. Searching for the overuse of '丅' and '丄'.
    """
    
    def __init__(self) -> None:
        self._wrong_stop_count: int = 0
        self._cjk_character_count: int = 0
    
    def eligible(self, character: str) -> bool:
        return True
    
    def feed(self, character: str) -> None:
        if character in {'丅', '丄'}:
            self._wrong_stop_count += 1
            return
        if is_cjk(character):
            self._cjk_character_count += 1
    
    def reset(self) -> None:
        self._wrong_stop_count = 0
        self._cjk_character_count = 0
    
    @property
    def ratio(self) -> float:
        if self._cjk_character_count < 16:
            return 0.0
        return self._wrong_stop_count / self._cjk_character_count



class ArchaicUpperLowerPlugin(MessDetectorPlugin):
    
    def __init__(self) -> None:
        self._buf: bool = False
        self._character_count_since_last_sep: int = 0
        self._successive_upper_lower_count: int = 0
        self._successive_upper_lower_count_final: int = 0
        self._character_count: int = 0
        self._last_alpha_seen: Optional[str] = None
        self._current_ascii_only: bool = True
    
    def eligible(self, character: str) -> bool:
        return True
    
    def feed(self, character: str) -> None:
        is_concerned = (character.isalpha() and is_case_variable(character))
        chunk_sep = is_concerned is False
        if (chunk_sep and self._character_count_since_last_sep > 0):
            if (self._character_count_since_last_sep <= 64 and character.isdigit() is False and self._current_ascii_only is False):
                self._successive_upper_lower_count_final += self._successive_upper_lower_count
            self._successive_upper_lower_count = 0
            self._character_count_since_last_sep = 0
            self._last_alpha_seen = None
            self._buf = False
            self._character_count += 1
            self._current_ascii_only = True
            return
        if (self._current_ascii_only is True and character.isascii() is False):
            self._current_ascii_only = False
        if self._last_alpha_seen is not None:
            if ((character.isupper() and self._last_alpha_seen.islower()) or (character.islower() and self._last_alpha_seen.isupper())):
                if self._buf is True:
                    self._successive_upper_lower_count += 2
                    self._buf = False
                else:
                    self._buf = True
            else:
                self._buf = False
        self._character_count += 1
        self._character_count_since_last_sep += 1
        self._last_alpha_seen = character
    
    def reset(self) -> None:
        self._character_count = 0
        self._character_count_since_last_sep = 0
        self._successive_upper_lower_count = 0
        self._successive_upper_lower_count_final = 0
        self._last_alpha_seen = None
        self._buf = False
        self._current_ascii_only = True
    
    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        return self._successive_upper_lower_count_final / self._character_count


@lru_cache(maxsize=1024)
def is_suspiciously_successive_range(unicode_range_a: Optional[str], unicode_range_b: Optional[str]) -> bool:
    """
    Determine if two Unicode range seen next to each other can be considered as suspicious.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.md.is_suspiciously_successive_range', 'is_suspiciously_successive_range(unicode_range_a, unicode_range_b)', {'UNICODE_SECONDARY_RANGE_KEYWORD': UNICODE_SECONDARY_RANGE_KEYWORD, 'lru_cache': lru_cache, 'unicode_range_a': unicode_range_a, 'unicode_range_b': unicode_range_b, 'Optional': Optional, 'str': str, 'Optional': Optional, 'str': str}, 1)

@lru_cache(maxsize=2048)
def mess_ratio(decoded_sequence: str, maximum_threshold: float = 0.2, debug: bool = False) -> float:
    """
    Compute a mess ratio given a decoded bytes sequence. The maximum threshold does stop the computation earlier.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.md.mess_ratio', 'mess_ratio(decoded_sequence, maximum_threshold=0.2, debug=False)', {'List': List, 'MessDetectorPlugin': MessDetectorPlugin, 'getLogger': getLogger, 'TRACE': TRACE, 'lru_cache': lru_cache, 'decoded_sequence': decoded_sequence, 'maximum_threshold': maximum_threshold, 'debug': debug}, 1)

