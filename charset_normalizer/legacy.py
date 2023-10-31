from typing import Any, Dict, Optional, Union
from warnings import warn
from .api import from_bytes
from .constant import CHARDET_CORRESPONDENCE

def detect(byte_str: bytes, should_rename_legacy: bool = False, **kwargs) -> Dict[(str, Optional[Union[(str, float)]])]:
    """
    chardet legacy method
    Detect the encoding of the given byte string. It should be mostly backward-compatible.
    Encoding name will match Chardet own writing whenever possible. (Not on encoding name unsupported by it)
    This function is deprecated and should be used to migrate your project easily, consult the documentation for
    further information. Not planned for removal.

    :param byte_str:     The byte sequence to examine.
    :param should_rename_legacy:  Should we rename legacy encodings
                                  to their more modern equivalents?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('charset_normalizer.legacy.detect', 'detect(byte_str, should_rename_legacy=False, **kwargs)', {'warn': warn, 'CHARDET_CORRESPONDENCE': CHARDET_CORRESPONDENCE, 'byte_str': byte_str, 'should_rename_legacy': should_rename_legacy, 'kwargs': kwargs, 'Dict': Dict}, 1)

