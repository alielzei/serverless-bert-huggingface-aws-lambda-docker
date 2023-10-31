import re
from typing import FrozenSet, NewType, Tuple, Union, cast
from .tags import Tag, parse_tag
from .version import InvalidVersion, Version
BuildTag = Union[(Tuple[()], Tuple[(int, str)])]
NormalizedName = NewType('NormalizedName', str)


class InvalidName(ValueError):
    """
    An invalid distribution name; users should refer to the packaging user guide.
    """
    



class InvalidWheelFilename(ValueError):
    """
    An invalid wheel filename was found, users should refer to PEP 427.
    """
    



class InvalidSdistFilename(ValueError):
    """
    An invalid sdist filename was found, users should refer to the packaging user guide.
    """
    

_validate_regex = re.compile('^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$', re.IGNORECASE)
_canonicalize_regex = re.compile('[-_.]+')
_normalized_regex = re.compile('^([a-z0-9]|[a-z0-9]([a-z0-9-](?!--))*[a-z0-9])$')
_build_tag_regex = re.compile('(\\d+)(.*)')

def canonicalize_name(name: str, *, validate: bool = False) -> NormalizedName:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.utils.canonicalize_name', 'canonicalize_name(name, validate: bool = False)', {'_validate_regex': _validate_regex, 'InvalidName': InvalidName, '_canonicalize_regex': _canonicalize_regex, 'NormalizedName': NormalizedName, 'name': name, 'validate': validate}, 1)

def is_normalized_name(name: str) -> bool:
    return _normalized_regex.match(name) is not None

def canonicalize_version(version: Union[(Version, str)], *, strip_trailing_zero: bool = True) -> str:
    """
    This is very similar to Version.__str__, but has one subtle difference
    with the way it handles the release segment.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.utils.canonicalize_version', 'canonicalize_version(version, strip_trailing_zero: bool = True)', {'Version': Version, 'InvalidVersion': InvalidVersion, 're': re, 'version': version, 'strip_trailing_zero': strip_trailing_zero, 'Union': Union}, 1)

def parse_wheel_filename(filename: str) -> Tuple[(NormalizedName, Version, BuildTag, FrozenSet[Tag])]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.utils.parse_wheel_filename', 'parse_wheel_filename(filename)', {'InvalidWheelFilename': InvalidWheelFilename, 're': re, 'canonicalize_name': canonicalize_name, 'Version': Version, 'InvalidVersion': InvalidVersion, '_build_tag_regex': _build_tag_regex, 'BuildTag': BuildTag, 'parse_tag': parse_tag, 'filename': filename, 'Tuple': Tuple}, 4)

def parse_sdist_filename(filename: str) -> Tuple[(NormalizedName, Version)]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.utils.parse_sdist_filename', 'parse_sdist_filename(filename)', {'InvalidSdistFilename': InvalidSdistFilename, 'canonicalize_name': canonicalize_name, 'Version': Version, 'InvalidVersion': InvalidVersion, 'filename': filename, 'Tuple': Tuple}, 2)

