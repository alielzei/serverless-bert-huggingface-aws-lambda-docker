"""
.. testsetup::

    from packaging.specifiers import Specifier, SpecifierSet, InvalidSpecifier
    from packaging.version import Version
"""

import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
UnparsedVersion = Union[(Version, str)]
UnparsedVersionVar = TypeVar('UnparsedVersionVar', bound=UnparsedVersion)
CallableOperator = Callable[([Version, str], bool)]

def _coerce_version(version: UnparsedVersion) -> Version:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.specifiers._coerce_version', '_coerce_version(version)', {'Version': Version, 'version': version}, 1)


class InvalidSpecifier(ValueError):
    """
    Raised when attempting to create a :class:`Specifier` with a specifier
    string that is invalid.

    >>> Specifier("lolwat")
    Traceback (most recent call last):
        ...
    packaging.specifiers.InvalidSpecifier: Invalid specifier: 'lolwat'
    """
    



class BaseSpecifier(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns the str representation of this Specifier-like object. This
        should be representative of the Specifier itself.
        """
        
    
    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Returns a hash value for this Specifier-like object.
        """
        
    
    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Returns a boolean representing whether or not the two Specifier-like
        objects are equal.

        :param other: The other object to check against.
        """
        
    
    @property
    @abc.abstractmethod
    def prereleases(self) -> Optional[bool]:
        """Whether or not pre-releases as a whole are allowed.

        This can be set to either ``True`` or ``False`` to explicitly enable or disable
        prereleases or it can be set to ``None`` (the default) to use default semantics.
        """
        
    
    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        """Setter for :attr:`prereleases`.

        :param value: The value to set.
        """
        
    
    @abc.abstractmethod
    def contains(self, item: str, prereleases: Optional[bool] = None) -> bool:
        """
        Determines if the given item is contained within this specifier.
        """
        
    
    @abc.abstractmethod
    def filter(self, iterable: Iterable[UnparsedVersionVar], prereleases: Optional[bool] = None) -> Iterator[UnparsedVersionVar]:
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """
        



class Specifier(BaseSpecifier):
    """This class abstracts handling of version specifiers.

    .. tip::

        It is generally not required to instantiate this manually. You should instead
        prefer to work with :class:`SpecifierSet` instead, which can parse
        comma-separated version specifiers (which is what package metadata contains).
    """
    _operator_regex_str = '\n        (?P<operator>(~=|==|!=|<=|>=|<|>|===))\n        '
    _version_regex_str = "\n        (?P<version>\n            (?:\n                # The identity operators allow for an escape hatch that will\n                # do an exact string match of the version you wish to install.\n                # This will not be parsed by PEP 440 and we cannot determine\n                # any semantic meaning from it. This operator is discouraged\n                # but included entirely as an escape hatch.\n                (?<====)  # Only match for the identity operator\n                \\s*\n                [^\\s;)]*  # The arbitrary version can be just about anything,\n                          # we match everything except for whitespace, a\n                          # semi-colon for marker support, and a closing paren\n                          # since versions can be enclosed in them.\n            )\n            |\n            (?:\n                # The (non)equality operators allow for wild card and local\n                # versions to be specified so we have to define these two\n                # operators separately to enable that.\n                (?<===|!=)            # Only match for equals and not equals\n\n                \\s*\n                v?\n                (?:[0-9]+!)?          # epoch\n                [0-9]+(?:\\.[0-9]+)*   # release\n\n                # You cannot use a wild card and a pre-release, post-release, a dev or\n                # local version together so group them with a | and make them optional.\n                (?:\n                    \\.\\*  # Wild card syntax of .*\n                    |\n                    (?:                                  # pre release\n                        [-_\\.]?\n                        (alpha|beta|preview|pre|a|b|c|rc)\n                        [-_\\.]?\n                        [0-9]*\n                    )?\n                    (?:                                  # post release\n                        (?:-[0-9]+)|(?:[-_\\.]?(post|rev|r)[-_\\.]?[0-9]*)\n                    )?\n                    (?:[-_\\.]?dev[-_\\.]?[0-9]*)?         # dev release\n                    (?:\\+[a-z0-9]+(?:[-_\\.][a-z0-9]+)*)? # local\n                )?\n            )\n            |\n            (?:\n                # The compatible operator requires at least two digits in the\n                # release segment.\n                (?<=~=)               # Only match for the compatible operator\n\n                \\s*\n                v?\n                (?:[0-9]+!)?          # epoch\n                [0-9]+(?:\\.[0-9]+)+   # release  (We have a + instead of a *)\n                (?:                   # pre release\n                    [-_\\.]?\n                    (alpha|beta|preview|pre|a|b|c|rc)\n                    [-_\\.]?\n                    [0-9]*\n                )?\n                (?:                                   # post release\n                    (?:-[0-9]+)|(?:[-_\\.]?(post|rev|r)[-_\\.]?[0-9]*)\n                )?\n                (?:[-_\\.]?dev[-_\\.]?[0-9]*)?          # dev release\n            )\n            |\n            (?:\n                # All other operators only allow a sub set of what the\n                # (non)equality operators do. Specifically they do not allow\n                # local versions to be specified nor do they allow the prefix\n                # matching wild cards.\n                (?<!==|!=|~=)         # We have special cases for these\n                                      # operators so we want to make sure they\n                                      # don't match here.\n\n                \\s*\n                v?\n                (?:[0-9]+!)?          # epoch\n                [0-9]+(?:\\.[0-9]+)*   # release\n                (?:                   # pre release\n                    [-_\\.]?\n                    (alpha|beta|preview|pre|a|b|c|rc)\n                    [-_\\.]?\n                    [0-9]*\n                )?\n                (?:                                   # post release\n                    (?:-[0-9]+)|(?:[-_\\.]?(post|rev|r)[-_\\.]?[0-9]*)\n                )?\n                (?:[-_\\.]?dev[-_\\.]?[0-9]*)?          # dev release\n            )\n        )\n        "
    _regex = re.compile('^\\s*' + _operator_regex_str + _version_regex_str + '\\s*$', re.VERBOSE | re.IGNORECASE)
    _operators = {'~=': 'compatible', '==': 'equal', '!=': 'not_equal', '<=': 'less_than_equal', '>=': 'greater_than_equal', '<': 'less_than', '>': 'greater_than', '===': 'arbitrary'}
    
    def __init__(self, spec: str = '', prereleases: Optional[bool] = None) -> None:
        """Initialize a Specifier instance.

        :param spec:
            The string representation of a specifier which will be parsed and
            normalized before use.
        :param prereleases:
            This tells the specifier if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.
        :raises InvalidSpecifier:
            If the given specifier is invalid (i.e. bad syntax).
        """
        match = self._regex.search(spec)
        if not match:
            raise InvalidSpecifier(f"Invalid specifier: '{spec}'")
        self._spec: Tuple[(str, str)] = (match.group('operator').strip(), match.group('version').strip())
        self._prereleases = prereleases
    
    @property
    def prereleases(self) -> bool:
        if self._prereleases is not None:
            return self._prereleases
        (operator, version) = self._spec
        if operator in ['==', '>=', '<=', '~=', '===']:
            if (operator == '==' and version.endswith('.*')):
                version = version[:-2]
            if Version(version).is_prerelease:
                return True
        return False
    
    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        self._prereleases = value
    
    @property
    def operator(self) -> str:
        """The operator of this specifier.

        >>> Specifier("==1.2.3").operator
        '=='
        """
        return self._spec[0]
    
    @property
    def version(self) -> str:
        """The version of this specifier.

        >>> Specifier("==1.2.3").version
        '1.2.3'
        """
        return self._spec[1]
    
    def __repr__(self) -> str:
        """A representation of the Specifier that shows all internal state.

        >>> Specifier('>=1.0.0')
        <Specifier('>=1.0.0')>
        >>> Specifier('>=1.0.0', prereleases=False)
        <Specifier('>=1.0.0', prereleases=False)>
        >>> Specifier('>=1.0.0', prereleases=True)
        <Specifier('>=1.0.0', prereleases=True)>
        """
        pre = (f', prereleases={self.prereleases!r}' if self._prereleases is not None else '')
        return f'<{self.__class__.__name__}({str(self)!r}{pre})>'
    
    def __str__(self) -> str:
        """A string representation of the Specifier that can be round-tripped.

        >>> str(Specifier('>=1.0.0'))
        '>=1.0.0'
        >>> str(Specifier('>=1.0.0', prereleases=False))
        '>=1.0.0'
        """
        return '{}{}'.format(*self._spec)
    
    @property
    def _canonical_spec(self) -> Tuple[(str, str)]:
        canonical_version = canonicalize_version(self._spec[1], strip_trailing_zero=self._spec[0] != '~=')
        return (self._spec[0], canonical_version)
    
    def __hash__(self) -> int:
        return hash(self._canonical_spec)
    
    def __eq__(self, other: object) -> bool:
        """Whether or not the two Specifier-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> Specifier("==1.2.3") == Specifier("== 1.2.3.0")
        True
        >>> (Specifier("==1.2.3", prereleases=False) ==
        ...  Specifier("==1.2.3", prereleases=True))
        True
        >>> Specifier("==1.2.3") == "==1.2.3"
        True
        >>> Specifier("==1.2.3") == Specifier("==1.2.4")
        False
        >>> Specifier("==1.2.3") == Specifier("~=1.2.3")
        False
        """
        if isinstance(other, str):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented
        return self._canonical_spec == other._canonical_spec
    
    def _get_operator(self, op: str) -> CallableOperator:
        operator_callable: CallableOperator = getattr(self, f'_compare_{self._operators[op]}')
        return operator_callable
    
    def _compare_compatible(self, prospective: Version, spec: str) -> bool:
        prefix = '.'.join(list(itertools.takewhile(_is_not_suffix, _version_split(spec)))[:-1])
        prefix += '.*'
        return (self._get_operator('>=')(prospective, spec) and self._get_operator('==')(prospective, prefix))
    
    def _compare_equal(self, prospective: Version, spec: str) -> bool:
        if spec.endswith('.*'):
            normalized_prospective = canonicalize_version(prospective.public, strip_trailing_zero=False)
            normalized_spec = canonicalize_version(spec[:-2], strip_trailing_zero=False)
            split_spec = _version_split(normalized_spec)
            split_prospective = _version_split(normalized_prospective)
            (padded_prospective, _) = _pad_version(split_prospective, split_spec)
            shortened_prospective = padded_prospective[:len(split_spec)]
            return shortened_prospective == split_spec
        else:
            spec_version = Version(spec)
            if not spec_version.local:
                prospective = Version(prospective.public)
            return prospective == spec_version
    
    def _compare_not_equal(self, prospective: Version, spec: str) -> bool:
        return not self._compare_equal(prospective, spec)
    
    def _compare_less_than_equal(self, prospective: Version, spec: str) -> bool:
        return Version(prospective.public) <= Version(spec)
    
    def _compare_greater_than_equal(self, prospective: Version, spec: str) -> bool:
        return Version(prospective.public) >= Version(spec)
    
    def _compare_less_than(self, prospective: Version, spec_str: str) -> bool:
        spec = Version(spec_str)
        if not prospective < spec:
            return False
        if (not spec.is_prerelease and prospective.is_prerelease):
            if Version(prospective.base_version) == Version(spec.base_version):
                return False
        return True
    
    def _compare_greater_than(self, prospective: Version, spec_str: str) -> bool:
        spec = Version(spec_str)
        if not prospective > spec:
            return False
        if (not spec.is_postrelease and prospective.is_postrelease):
            if Version(prospective.base_version) == Version(spec.base_version):
                return False
        if prospective.local is not None:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False
        return True
    
    def _compare_arbitrary(self, prospective: Version, spec: str) -> bool:
        return str(prospective).lower() == str(spec).lower()
    
    def __contains__(self, item: Union[(str, Version)]) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in Specifier(">=1.2.3")
        True
        >>> Version("1.2.3") in Specifier(">=1.2.3")
        True
        >>> "1.0.0" in Specifier(">=1.2.3")
        False
        >>> "1.3.0a1" in Specifier(">=1.2.3")
        False
        >>> "1.3.0a1" in Specifier(">=1.2.3", prereleases=True)
        True
        """
        return self.contains(item)
    
    def contains(self, item: UnparsedVersion, prereleases: Optional[bool] = None) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item:
            The item to check for, which can be a version string or a
            :class:`Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this Specifier. If set to
            ``None`` (the default), it uses :attr:`prereleases` to determine
            whether or not prereleases are allowed.

        >>> Specifier(">=1.2.3").contains("1.2.3")
        True
        >>> Specifier(">=1.2.3").contains(Version("1.2.3"))
        True
        >>> Specifier(">=1.2.3").contains("1.0.0")
        False
        >>> Specifier(">=1.2.3").contains("1.3.0a1")
        False
        >>> Specifier(">=1.2.3", prereleases=True).contains("1.3.0a1")
        True
        >>> Specifier(">=1.2.3").contains("1.3.0a1", prereleases=True)
        True
        """
        if prereleases is None:
            prereleases = self.prereleases
        normalized_item = _coerce_version(item)
        if (normalized_item.is_prerelease and not prereleases):
            return False
        operator_callable: CallableOperator = self._get_operator(self.operator)
        return operator_callable(normalized_item, self.version)
    
    def filter(self, iterable: Iterable[UnparsedVersionVar], prereleases: Optional[bool] = None) -> Iterator[UnparsedVersionVar]:
        """Filter items in the given iterable, that match the specifier.

        :param iterable:
            An iterable that can contain version strings and :class:`Version` instances.
            The items in the iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will be intelligently decide whether to allow
            prereleases or not (based on the :attr:`prereleases` attribute, and
            whether the only versions matching are prereleases).

        This method is smarter than just ``filter(Specifier().contains, [...])``
        because it implements the rule from :pep:`440` that a prerelease item
        SHOULD be accepted if no other versions match the given specifier.

        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.2.3", "1.3", Version("1.4")]))
        ['1.2.3', '1.3', <Version('1.4')>]
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.5a1"]))
        ['1.5a1']
        >>> list(Specifier(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(Specifier(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        """
        yielded = False
        found_prereleases = []
        kw = {'prereleases': (prereleases if prereleases is not None else True)}
        for version in iterable:
            parsed_version = _coerce_version(version)
            if self.contains(parsed_version, **kw):
                if (parsed_version.is_prerelease and not ((prereleases or self.prereleases))):
                    found_prereleases.append(version)
                else:
                    yielded = True
                    yield version
        if (not yielded and found_prereleases):
            for version in found_prereleases:
                yield version

_prefix_regex = re.compile('^([0-9]+)((?:a|b|c|rc)[0-9]+)$')

def _version_split(version: str) -> List[str]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.specifiers._version_split', '_version_split(version)', {'List': List, '_prefix_regex': _prefix_regex, 'version': version, 'List': List, 'str': str}, 1)

def _is_not_suffix(segment: str) -> bool:
    return not any((segment.startswith(prefix) for prefix in ('dev', 'a', 'b', 'rc', 'post')))

def _pad_version(left: List[str], right: List[str]) -> Tuple[(List[str], List[str])]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.specifiers._pad_version', '_pad_version(left, right)', {'itertools': itertools, 'left': left, 'right': right, 'List': List, 'str': str, 'List': List, 'str': str, 'Tuple': Tuple}, 2)


class SpecifierSet(BaseSpecifier):
    """This class abstracts handling of a set of version specifiers.

    It can be passed a single specifier (``>=3.0``), a comma-separated list of
    specifiers (``>=3.0,!=3.1``), or no specifier at all.
    """
    
    def __init__(self, specifiers: str = '', prereleases: Optional[bool] = None) -> None:
        """Initialize a SpecifierSet instance.

        :param specifiers:
            The string representation of a specifier or a comma-separated list of
            specifiers which will be parsed and normalized before use.
        :param prereleases:
            This tells the SpecifierSet if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.

        :raises InvalidSpecifier:
            If the given ``specifiers`` are not parseable than this exception will be
            raised.
        """
        split_specifiers = [s.strip() for s in specifiers.split(',') if s.strip()]
        parsed: Set[Specifier] = set()
        for specifier in split_specifiers:
            parsed.add(Specifier(specifier))
        self._specs = frozenset(parsed)
        self._prereleases = prereleases
    
    @property
    def prereleases(self) -> Optional[bool]:
        if self._prereleases is not None:
            return self._prereleases
        if not self._specs:
            return None
        return any((s.prereleases for s in self._specs))
    
    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        self._prereleases = value
    
    def __repr__(self) -> str:
        """A representation of the specifier set that shows all internal state.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> SpecifierSet('>=1.0.0,!=2.0.0')
        <SpecifierSet('!=2.0.0,>=1.0.0')>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=False)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=False)>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=True)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=True)>
        """
        pre = (f', prereleases={self.prereleases!r}' if self._prereleases is not None else '')
        return f'<SpecifierSet({str(self)!r}{pre})>'
    
    def __str__(self) -> str:
        """A string representation of the specifier set that can be round-tripped.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> str(SpecifierSet(">=1.0.0,!=1.0.1"))
        '!=1.0.1,>=1.0.0'
        >>> str(SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False))
        '!=1.0.1,>=1.0.0'
        """
        return ','.join(sorted((str(s) for s in self._specs)))
    
    def __hash__(self) -> int:
        return hash(self._specs)
    
    def __and__(self, other: Union[('SpecifierSet', str)]) -> 'SpecifierSet':
        """Return a SpecifierSet which is a combination of the two sets.

        :param other: The other object to combine with.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") & '<=2.0.0,!=2.0.1'
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        >>> SpecifierSet(">=1.0.0,!=1.0.1") & SpecifierSet('<=2.0.0,!=2.0.1')
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        """
        if isinstance(other, str):
            other = SpecifierSet(other)
        elif not isinstance(other, SpecifierSet):
            return NotImplemented
        specifier = SpecifierSet()
        specifier._specs = frozenset(self._specs | other._specs)
        if (self._prereleases is None and other._prereleases is not None):
            specifier._prereleases = other._prereleases
        elif (self._prereleases is not None and other._prereleases is None):
            specifier._prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            specifier._prereleases = self._prereleases
        else:
            raise ValueError('Cannot combine SpecifierSets with True and False prerelease overrides.')
        return specifier
    
    def __eq__(self, other: object) -> bool:
        """Whether or not the two SpecifierSet-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> (SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False) ==
        ...  SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == ">=1.0.0,!=1.0.1"
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.2")
        False
        """
        if isinstance(other, (str, Specifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented
        return self._specs == other._specs
    
    def __len__(self) -> int:
        """Returns the number of specifiers in this specifier set."""
        return len(self._specs)
    
    def __iter__(self) -> Iterator[Specifier]:
        """
        Returns an iterator over all the underlying :class:`Specifier` instances
        in this specifier set.

        >>> sorted(SpecifierSet(">=1.0.0,!=1.0.1"), key=str)
        [<Specifier('!=1.0.1')>, <Specifier('>=1.0.0')>]
        """
        return iter(self._specs)
    
    def __contains__(self, item: UnparsedVersion) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> Version("1.2.3") in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> "1.0.1" in SpecifierSet(">=1.0.0,!=1.0.1")
        False
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1")
        False
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True)
        True
        """
        return self.contains(item)
    
    def contains(self, item: UnparsedVersion, prereleases: Optional[bool] = None, installed: Optional[bool] = None) -> bool:
        """Return whether or not the item is contained in this SpecifierSet.

        :param item:
            The item to check for, which can be a version string or a
            :class:`Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this SpecifierSet. If set to
            ``None`` (the default), it uses :attr:`prereleases` to determine
            whether or not prereleases are allowed.

        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.2.3")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains(Version("1.2.3"))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.0.1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True).contains("1.3.0a1")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1", prereleases=True)
        True
        """
        if not isinstance(item, Version):
            item = Version(item)
        if prereleases is None:
            prereleases = self.prereleases
        if (not prereleases and item.is_prerelease):
            return False
        if (installed and item.is_prerelease):
            item = Version(item.base_version)
        return all((s.contains(item, prereleases=prereleases) for s in self._specs))
    
    def filter(self, iterable: Iterable[UnparsedVersionVar], prereleases: Optional[bool] = None) -> Iterator[UnparsedVersionVar]:
        """Filter items in the given iterable, that match the specifiers in this set.

        :param iterable:
            An iterable that can contain version strings and :class:`Version` instances.
            The items in the iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will be intelligently decide whether to allow
            prereleases or not (based on the :attr:`prereleases` attribute, and
            whether the only versions matching are prereleases).

        This method is smarter than just ``filter(SpecifierSet(...).contains, [...])``
        because it implements the rule from :pep:`440` that a prerelease item
        SHOULD be accepted if no other versions match the given specifier.

        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", Version("1.4")]))
        ['1.3', <Version('1.4')>]
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.5a1"]))
        []
        >>> list(SpecifierSet(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']

        An "empty" SpecifierSet will filter items based on the presence of prerelease
        versions in the set.

        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet("").filter(["1.5a1"]))
        ['1.5a1']
        >>> list(SpecifierSet("", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        """
        if prereleases is None:
            prereleases = self.prereleases
        if self._specs:
            for spec in self._specs:
                iterable = spec.filter(iterable, prereleases=bool(prereleases))
            return iter(iterable)
        else:
            filtered: List[UnparsedVersionVar] = []
            found_prereleases: List[UnparsedVersionVar] = []
            for item in iterable:
                parsed_version = _coerce_version(item)
                if (parsed_version.is_prerelease and not prereleases):
                    if not filtered:
                        found_prereleases.append(item)
                else:
                    filtered.append(item)
            if (not filtered and found_prereleases and prereleases is None):
                return iter(found_prereleases)
            return iter(filtered)


