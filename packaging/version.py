"""
.. testsetup::

    from packaging.version import parse, Version
"""

import itertools
import re
from typing import Any, Callable, NamedTuple, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
__all__ = ['VERSION_PATTERN', 'parse', 'Version', 'InvalidVersion']
LocalType = Tuple[(Union[(int, str)], ...)]
CmpPrePostDevType = Union[(InfinityType, NegativeInfinityType, Tuple[(str, int)])]
CmpLocalType = Union[(NegativeInfinityType, Tuple[(Union[(Tuple[(int, str)], Tuple[(NegativeInfinityType, Union[(int, str)])])], ...)])]
CmpKey = Tuple[(int, Tuple[(int, ...)], CmpPrePostDevType, CmpPrePostDevType, CmpPrePostDevType, CmpLocalType)]
VersionComparisonMethod = Callable[([CmpKey, CmpKey], bool)]


class _Version(NamedTuple):
    epoch: int
    release: Tuple[(int, ...)]
    dev: Optional[Tuple[(str, int)]]
    pre: Optional[Tuple[(str, int)]]
    post: Optional[Tuple[(str, int)]]
    local: Optional[LocalType]


def parse(version: str) -> 'Version':
    """Parse the given version string.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """
    



class _BaseVersion:
    _key: Tuple[(Any, ...)]
    
    def __hash__(self) -> int:
        return hash(self._key)
    
    def __lt__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key < other._key
    
    def __le__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key <= other._key
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key == other._key
    
    def __ge__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key >= other._key
    
    def __gt__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key > other._key
    
    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key != other._key

_VERSION_PATTERN = '\n    v?\n    (?:\n        (?:(?P<epoch>[0-9]+)!)?                           # epoch\n        (?P<release>[0-9]+(?:\\.[0-9]+)*)                  # release segment\n        (?P<pre>                                          # pre-release\n            [-_\\.]?\n            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)\n            [-_\\.]?\n            (?P<pre_n>[0-9]+)?\n        )?\n        (?P<post>                                         # post release\n            (?:-(?P<post_n1>[0-9]+))\n            |\n            (?:\n                [-_\\.]?\n                (?P<post_l>post|rev|r)\n                [-_\\.]?\n                (?P<post_n2>[0-9]+)?\n            )\n        )?\n        (?P<dev>                                          # dev release\n            [-_\\.]?\n            (?P<dev_l>dev)\n            [-_\\.]?\n            (?P<dev_n>[0-9]+)?\n        )?\n    )\n    (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       # local version\n'
VERSION_PATTERN = _VERSION_PATTERN
'\nA string containing the regular expression used to match a valid version.\n\nThe pattern is not anchored at either end, and is intended for embedding in larger\nexpressions (for example, matching a version number as part of a file name). The\nregular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``\nflags set.\n\n:meta hide-value:\n'


class Version(_BaseVersion):
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True
    """
    _regex = re.compile('^\\s*' + VERSION_PATTERN + '\\s*$', re.VERBOSE | re.IGNORECASE)
    _key: CmpKey
    
    def __init__(self, version: str) -> None:
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")
        self._version = _Version(epoch=(int(match.group('epoch')) if match.group('epoch') else 0), release=tuple((int(i) for i in match.group('release').split('.'))), pre=_parse_letter_version(match.group('pre_l'), match.group('pre_n')), post=_parse_letter_version(match.group('post_l'), (match.group('post_n1') or match.group('post_n2'))), dev=_parse_letter_version(match.group('dev_l'), match.group('dev_n')), local=_parse_local_version(match.group('local')))
        self._key = _cmpkey(self._version.epoch, self._version.release, self._version.pre, self._version.post, self._version.dev, self._version.local)
    
    def __repr__(self) -> str:
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<Version('{self}')>"
    
    def __str__(self) -> str:
        """A string representation of the version that can be rounded-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        parts = []
        if self.epoch != 0:
            parts.append(f'{self.epoch}!')
        parts.append('.'.join((str(x) for x in self.release)))
        if self.pre is not None:
            parts.append(''.join((str(x) for x in self.pre)))
        if self.post is not None:
            parts.append(f'.post{self.post}')
        if self.dev is not None:
            parts.append(f'.dev{self.dev}')
        if self.local is not None:
            parts.append(f'+{self.local}')
        return ''.join(parts)
    
    @property
    def epoch(self) -> int:
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._version.epoch
    
    @property
    def release(self) -> Tuple[(int, ...)]:
        """The components of the "release" segment of the version.

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        Includes trailing zeroes but not the epoch or any pre-release / development /
        post-release suffixes.
        """
        return self._version.release
    
    @property
    def pre(self) -> Optional[Tuple[(str, int)]]:
        """The pre-release segment of the version.

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._version.pre
    
    @property
    def post(self) -> Optional[int]:
        """The post-release number of the version.

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return (self._version.post[1] if self._version.post else None)
    
    @property
    def dev(self) -> Optional[int]:
        """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return (self._version.dev[1] if self._version.dev else None)
    
    @property
    def local(self) -> Optional[str]:
        """The local version segment of the version.

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._version.local:
            return '.'.join((str(x) for x in self._version.local))
        else:
            return None
    
    @property
    def public(self) -> str:
        """The public portion of the version.

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1.2.3+abc.dev1").public
        '1.2.3'
        """
        return str(self).split('+', 1)[0]
    
    @property
    def base_version(self) -> str:
        """The "base version" of the version.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3+abc.dev1").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
        parts = []
        if self.epoch != 0:
            parts.append(f'{self.epoch}!')
        parts.append('.'.join((str(x) for x in self.release)))
        return ''.join(parts)
    
    @property
    def is_prerelease(self) -> bool:
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        return (self.dev is not None or self.pre is not None)
    
    @property
    def is_postrelease(self) -> bool:
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        return self.post is not None
    
    @property
    def is_devrelease(self) -> bool:
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        return self.dev is not None
    
    @property
    def major(self) -> int:
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1
        """
        return (self.release[0] if len(self.release) >= 1 else 0)
    
    @property
    def minor(self) -> int:
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0
        """
        return (self.release[1] if len(self.release) >= 2 else 0)
    
    @property
    def micro(self) -> int:
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
        return (self.release[2] if len(self.release) >= 3 else 0)


def _parse_letter_version(letter: Optional[str], number: Union[(str, bytes, SupportsInt, None)]) -> Optional[Tuple[(str, int)]]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.version._parse_letter_version', '_parse_letter_version(letter, number)', {'letter': letter, 'number': number, 'Optional': Optional, 'str': str, 'Union': Union, 'Optional': Optional}, 2)
_local_version_separators = re.compile('[\\._-]')

def _parse_local_version(local: Optional[str]) -> Optional[LocalType]:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.version._parse_local_version', '_parse_local_version(local)', {'_local_version_separators': _local_version_separators, 'local': local, 'Optional': Optional, 'str': str, 'Optional': Optional, 'LocalType': LocalType}, 1)

def _cmpkey(epoch: int, release: Tuple[(int, ...)], pre: Optional[Tuple[(str, int)]], post: Optional[Tuple[(str, int)]], dev: Optional[Tuple[(str, int)]], local: Optional[LocalType]) -> CmpKey:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.version._cmpkey', '_cmpkey(epoch, release, pre, post, dev, local)', {'itertools': itertools, 'CmpPrePostDevType': CmpPrePostDevType, 'NegativeInfinity': NegativeInfinity, 'Infinity': Infinity, 'CmpLocalType': CmpLocalType, 'epoch': epoch, 'release': release, 'pre': pre, 'post': post, 'dev': dev, 'local': local, 'Tuple': Tuple, 'Optional': Optional, 'Optional': Optional, 'Optional': Optional, 'Optional': Optional, 'LocalType': LocalType}, 6)

