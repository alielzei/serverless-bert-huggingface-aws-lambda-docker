import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, Union, cast
from . import requirements, specifiers, utils, version as version_module
T = typing.TypeVar('T')
if sys.version_info[:2] >= (3, 8):
    from typing import Literal, TypedDict
elif typing.TYPE_CHECKING:
    from typing_extensions import Literal, TypedDict
else:
    try:
        from typing_extensions import Literal, TypedDict
    except ImportError:
        
        
        class Literal:
            
            def __init_subclass__(*_args, **_kwargs):
                pass
        
        
        
        class TypedDict:
            
            def __init_subclass__(*_args, **_kwargs):
                pass
        
try:
    ExceptionGroup = __builtins__.ExceptionGroup
except AttributeError:
    
    
    class ExceptionGroup(Exception):
        """A minimal implementation of :external:exc:`ExceptionGroup` from Python 3.11.

        If :external:exc:`ExceptionGroup` is already defined by Python itself,
        that version is used instead.
        """
        message: str
        exceptions: List[Exception]
        
        def __init__(self, message: str, exceptions: List[Exception]) -> None:
            self.message = message
            self.exceptions = exceptions
        
        def __repr__(self) -> str:
            return f'{self.__class__.__name__}({self.message!r}, {self.exceptions!r})'
    


class InvalidMetadata(ValueError):
    """A metadata field contains invalid data."""
    field: str
    'The name of the field that contains invalid data.'
    
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)



class RawMetadata(TypedDict, total=False):
    """A dictionary of raw core metadata.

    Each field in core metadata maps to a key of this dictionary (when data is
    provided). The key is lower-case and underscores are used instead of dashes
    compared to the equivalent core metadata field. Any core metadata field that
    can be specified multiple times or can hold multiple values in a single
    field have a key with a plural name. See :class:`Metadata` whose attributes
    match the keys of this dictionary.

    Core metadata fields that can be specified multiple times are stored as a
    list or dict depending on which is appropriate for the field. Any fields
    which hold multiple values in a single field are stored as a list.

    """
    metadata_version: str
    name: str
    version: str
    platforms: List[str]
    summary: str
    description: str
    keywords: List[str]
    home_page: str
    author: str
    author_email: str
    license: str
    supported_platforms: List[str]
    download_url: str
    classifiers: List[str]
    requires: List[str]
    provides: List[str]
    obsoletes: List[str]
    maintainer: str
    maintainer_email: str
    requires_dist: List[str]
    provides_dist: List[str]
    obsoletes_dist: List[str]
    requires_python: str
    requires_external: List[str]
    project_urls: Dict[(str, str)]
    description_content_type: str
    provides_extra: List[str]
    dynamic: List[str]

_STRING_FIELDS = {'author', 'author_email', 'description', 'description_content_type', 'download_url', 'home_page', 'license', 'maintainer', 'maintainer_email', 'metadata_version', 'name', 'requires_python', 'summary', 'version'}
_LIST_FIELDS = {'classifiers', 'dynamic', 'obsoletes', 'obsoletes_dist', 'platforms', 'provides', 'provides_dist', 'provides_extra', 'requires', 'requires_dist', 'requires_external', 'supported_platforms'}
_DICT_FIELDS = {'project_urls'}

def _parse_keywords(data: str) -> List[str]:
    """Split a string of comma-separate keyboards into a list of keywords."""
    return [k.strip() for k in data.split(',')]

def _parse_project_urls(data: List[str]) -> Dict[(str, str)]:
    """Parse a list of label/URL string pairings separated by a comma."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.metadata._parse_project_urls', '_parse_project_urls(data)', {'data': data, 'List': List, 'str': str, 'Dict': Dict}, 1)

def _get_payload(msg: email.message.Message, source: Union[(bytes, str)]) -> str:
    """Get the body of the message."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.metadata._get_payload', '_get_payload(msg, source)', {'msg': msg, 'source': source, 'email': email, 'Union': Union}, 1)
_EMAIL_TO_RAW_MAPPING = {'author': 'author', 'author-email': 'author_email', 'classifier': 'classifiers', 'description': 'description', 'description-content-type': 'description_content_type', 'download-url': 'download_url', 'dynamic': 'dynamic', 'home-page': 'home_page', 'keywords': 'keywords', 'license': 'license', 'maintainer': 'maintainer', 'maintainer-email': 'maintainer_email', 'metadata-version': 'metadata_version', 'name': 'name', 'obsoletes': 'obsoletes', 'obsoletes-dist': 'obsoletes_dist', 'platform': 'platforms', 'project-url': 'project_urls', 'provides': 'provides', 'provides-dist': 'provides_dist', 'provides-extra': 'provides_extra', 'requires': 'requires', 'requires-dist': 'requires_dist', 'requires-external': 'requires_external', 'requires-python': 'requires_python', 'summary': 'summary', 'supported-platform': 'supported_platforms', 'version': 'version'}
_RAW_TO_EMAIL_MAPPING = {raw: email for (email, raw) in _EMAIL_TO_RAW_MAPPING.items()}

def parse_email(data: Union[(bytes, str)]) -> Tuple[(RawMetadata, Dict[(str, List[str])])]:
    """Parse a distribution's metadata stored as email headers (e.g. from ``METADATA``).

    This function returns a two-item tuple of dicts. The first dict is of
    recognized fields from the core metadata specification. Fields that can be
    parsed and translated into Python's built-in types are converted
    appropriately. All other fields are left as-is. Fields that are allowed to
    appear multiple times are stored as lists.

    The second dict contains all other fields from the metadata. This includes
    any unrecognized fields. It also includes any fields which are expected to
    be parsed into a built-in type but were not formatted appropriately. Finally,
    any fields that are expected to appear only once but are repeated are
    included in this dict.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.metadata.parse_email', 'parse_email(data)', {'Dict': Dict, 'Union': Union, 'List': List, 'email': email, 'Tuple': Tuple, 'Optional': Optional, '_EMAIL_TO_RAW_MAPPING': _EMAIL_TO_RAW_MAPPING, '_STRING_FIELDS': _STRING_FIELDS, '_LIST_FIELDS': _LIST_FIELDS, '_parse_keywords': _parse_keywords, '_parse_project_urls': _parse_project_urls, '_get_payload': _get_payload, 'RawMetadata': RawMetadata, 'data': data, 'Union': Union, 'Tuple': Tuple}, 2)
_NOT_FOUND = object()
_VALID_METADATA_VERSIONS = ['1.0', '1.1', '1.2', '2.1', '2.2', '2.3']
_MetadataVersion = Literal[('1.0', '1.1', '1.2', '2.1', '2.2', '2.3')]
_REQUIRED_ATTRS = frozenset(['metadata_version', 'name', 'version'])


class _Validator(Generic[T]):
    """Validate a metadata field.

    All _process_*() methods correspond to a core metadata field. The method is
    called with the field's raw value. If the raw value is valid it is returned
    in its "enriched" form (e.g. ``version.Version`` for the ``Version`` field).
    If the raw value is invalid, :exc:`InvalidMetadata` is raised (with a cause
    as appropriate).
    """
    name: str
    raw_name: str
    added: _MetadataVersion
    
    def __init__(self, *, added: _MetadataVersion = '1.0') -> None:
        self.added = added
    
    def __set_name__(self, _owner: 'Metadata', name: str) -> None:
        self.name = name
        self.raw_name = _RAW_TO_EMAIL_MAPPING[name]
    
    def __get__(self, instance: 'Metadata', _owner: Type['Metadata']) -> T:
        cache = instance.__dict__
        try:
            value = instance._raw[self.name]
        except KeyError:
            if self.name in _STRING_FIELDS:
                value = ''
            elif self.name in _LIST_FIELDS:
                value = []
            elif self.name in _DICT_FIELDS:
                value = {}
            else:
                assert False
        try:
            converter: Callable[([Any], T)] = getattr(self, f'_process_{self.name}')
        except AttributeError:
            pass
        else:
            value = converter(value)
        cache[self.name] = value
        try:
            del instance._raw[self.name]
        except KeyError:
            pass
        return cast(T, value)
    
    def _invalid_metadata(self, msg: str, cause: Optional[Exception] = None) -> InvalidMetadata:
        exc = InvalidMetadata(self.raw_name, msg.format_map({'field': repr(self.raw_name)}))
        exc.__cause__ = cause
        return exc
    
    def _process_metadata_version(self, value: str) -> _MetadataVersion:
        if value not in _VALID_METADATA_VERSIONS:
            raise self._invalid_metadata(f'{value!r} is not a valid metadata version')
        return cast(_MetadataVersion, value)
    
    def _process_name(self, value: str) -> str:
        if not value:
            raise self._invalid_metadata('{field} is a required field')
        try:
            utils.canonicalize_name(value, validate=True)
        except utils.InvalidName as exc:
            raise self._invalid_metadata(f'{value!r} is invalid for {{field}}', cause=exc)
        else:
            return value
    
    def _process_version(self, value: str) -> version_module.Version:
        if not value:
            raise self._invalid_metadata('{field} is a required field')
        try:
            return version_module.parse(value)
        except version_module.InvalidVersion as exc:
            raise self._invalid_metadata(f'{value!r} is invalid for {{field}}', cause=exc)
    
    def _process_summary(self, value: str) -> str:
        """Check the field contains no newlines."""
        if '\n' in value:
            raise self._invalid_metadata('{field} must be a single line')
        return value
    
    def _process_description_content_type(self, value: str) -> str:
        content_types = {'text/plain', 'text/x-rst', 'text/markdown'}
        message = email.message.EmailMessage()
        message['content-type'] = value
        (content_type, parameters) = (message.get_content_type().lower(), message['content-type'].params)
        if (content_type not in content_types or content_type not in value.lower()):
            raise self._invalid_metadata(f'{{field}} must be one of {list(content_types)}, not {value!r}')
        charset = parameters.get('charset', 'UTF-8')
        if charset != 'UTF-8':
            raise self._invalid_metadata(f'{{field}} can only specify the UTF-8 charset, not {list(charset)}')
        markdown_variants = {'GFM', 'CommonMark'}
        variant = parameters.get('variant', 'GFM')
        if (content_type == 'text/markdown' and variant not in markdown_variants):
            raise self._invalid_metadata(f'valid Markdown variants for {{field}} are {list(markdown_variants)}, not {variant!r}')
        return value
    
    def _process_dynamic(self, value: List[str]) -> List[str]:
        for dynamic_field in map(str.lower, value):
            if dynamic_field in {'name', 'version', 'metadata-version'}:
                raise self._invalid_metadata(f'{value!r} is not allowed as a dynamic field')
            elif dynamic_field not in _EMAIL_TO_RAW_MAPPING:
                raise self._invalid_metadata(f'{value!r} is not a valid dynamic field')
        return list(map(str.lower, value))
    
    def _process_provides_extra(self, value: List[str]) -> List[utils.NormalizedName]:
        normalized_names = []
        try:
            for name in value:
                normalized_names.append(utils.canonicalize_name(name, validate=True))
        except utils.InvalidName as exc:
            raise self._invalid_metadata(f'{name!r} is invalid for {{field}}', cause=exc)
        else:
            return normalized_names
    
    def _process_requires_python(self, value: str) -> specifiers.SpecifierSet:
        try:
            return specifiers.SpecifierSet(value)
        except specifiers.InvalidSpecifier as exc:
            raise self._invalid_metadata(f'{value!r} is invalid for {{field}}', cause=exc)
    
    def _process_requires_dist(self, value: List[str]) -> List[requirements.Requirement]:
        reqs = []
        try:
            for req in value:
                reqs.append(requirements.Requirement(req))
        except requirements.InvalidRequirement as exc:
            raise self._invalid_metadata(f'{req!r} is invalid for {{field}}', cause=exc)
        else:
            return reqs



class Metadata:
    """Representation of distribution metadata.

    Compared to :class:`RawMetadata`, this class provides objects representing
    metadata fields instead of only using built-in types. Any invalid metadata
    will cause :exc:`InvalidMetadata` to be raised (with a
    :py:attr:`~BaseException.__cause__` attribute as appropriate).
    """
    _raw: RawMetadata
    
    @classmethod
    def from_raw(cls, data: RawMetadata, *, validate: bool = True) -> 'Metadata':
        """Create an instance from :class:`RawMetadata`.

        If *validate* is true, all metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
        ins = cls()
        ins._raw = data.copy()
        if validate:
            exceptions: List[InvalidMetadata] = []
            try:
                metadata_version = ins.metadata_version
                metadata_age = _VALID_METADATA_VERSIONS.index(metadata_version)
            except InvalidMetadata as metadata_version_exc:
                exceptions.append(metadata_version_exc)
                metadata_version = None
            fields_to_check = frozenset(ins._raw) | _REQUIRED_ATTRS
            fields_to_check -= {'metadata_version'}
            for key in fields_to_check:
                try:
                    if metadata_version:
                        try:
                            field_metadata_version = cls.__dict__[key].added
                        except KeyError:
                            exc = InvalidMetadata(key, f'unrecognized field: {key!r}')
                            exceptions.append(exc)
                            continue
                        field_age = _VALID_METADATA_VERSIONS.index(field_metadata_version)
                        if field_age > metadata_age:
                            field = _RAW_TO_EMAIL_MAPPING[key]
                            exc = InvalidMetadata(field, '{field} introduced in metadata version {field_metadata_version}, not {metadata_version}')
                            exceptions.append(exc)
                            continue
                    getattr(ins, key)
                except InvalidMetadata as exc:
                    exceptions.append(exc)
            if exceptions:
                raise ExceptionGroup('invalid metadata', exceptions)
        return ins
    
    @classmethod
    def from_email(cls, data: Union[(bytes, str)], *, validate: bool = True) -> 'Metadata':
        """Parse metadata from email headers.

        If *validate* is true, the metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
        exceptions: list[InvalidMetadata] = []
        (raw, unparsed) = parse_email(data)
        if validate:
            for unparsed_key in unparsed:
                if unparsed_key in _EMAIL_TO_RAW_MAPPING:
                    message = f'{unparsed_key!r} has invalid data'
                else:
                    message = f'unrecognized field: {unparsed_key!r}'
                exceptions.append(InvalidMetadata(unparsed_key, message))
            if exceptions:
                raise ExceptionGroup('unparsed', exceptions)
        try:
            return cls.from_raw(raw, validate=validate)
        except ExceptionGroup as exc_group:
            exceptions.extend(exc_group.exceptions)
            raise ExceptionGroup('invalid or unparsed metadata', exceptions) from None
    metadata_version: _Validator[_MetadataVersion] = _Validator()
    ':external:ref:`core-metadata-metadata-version`\n    (required; validated to be a valid metadata version)'
    name: _Validator[str] = _Validator()
    ':external:ref:`core-metadata-name`\n    (required; validated using :func:`~packaging.utils.canonicalize_name` and its\n    *validate* parameter)'
    version: _Validator[version_module.Version] = _Validator()
    ':external:ref:`core-metadata-version` (required)'
    dynamic: _Validator[List[str]] = _Validator(added='2.2')
    ':external:ref:`core-metadata-dynamic`\n    (validated against core metadata field names and lowercased)'
    platforms: _Validator[List[str]] = _Validator()
    ':external:ref:`core-metadata-platform`'
    supported_platforms: _Validator[List[str]] = _Validator(added='1.1')
    ':external:ref:`core-metadata-supported-platform`'
    summary: _Validator[str] = _Validator()
    ':external:ref:`core-metadata-summary` (validated to contain no newlines)'
    description: _Validator[str] = _Validator()
    ':external:ref:`core-metadata-description`'
    description_content_type: _Validator[str] = _Validator(added='2.1')
    ':external:ref:`core-metadata-description-content-type` (validated)'
    keywords: _Validator[List[str]] = _Validator()
    ':external:ref:`core-metadata-keywords`'
    home_page: _Validator[str] = _Validator()
    ':external:ref:`core-metadata-home-page`'
    download_url: _Validator[str] = _Validator(added='1.1')
    ':external:ref:`core-metadata-download-url`'
    author: _Validator[str] = _Validator()
    ':external:ref:`core-metadata-author`'
    author_email: _Validator[str] = _Validator()
    ':external:ref:`core-metadata-author-email`'
    maintainer: _Validator[str] = _Validator(added='1.2')
    ':external:ref:`core-metadata-maintainer`'
    maintainer_email: _Validator[str] = _Validator(added='1.2')
    ':external:ref:`core-metadata-maintainer-email`'
    license: _Validator[str] = _Validator()
    ':external:ref:`core-metadata-license`'
    classifiers: _Validator[List[str]] = _Validator(added='1.1')
    ':external:ref:`core-metadata-classifier`'
    requires_dist: _Validator[List[requirements.Requirement]] = _Validator(added='1.2')
    ':external:ref:`core-metadata-requires-dist`'
    requires_python: _Validator[specifiers.SpecifierSet] = _Validator(added='1.2')
    ':external:ref:`core-metadata-requires-python`'
    requires_external: _Validator[List[str]] = _Validator(added='1.2')
    ':external:ref:`core-metadata-requires-external`'
    project_urls: _Validator[Dict[(str, str)]] = _Validator(added='1.2')
    ':external:ref:`core-metadata-project-url`'
    provides_extra: _Validator[List[utils.NormalizedName]] = _Validator(added='2.1')
    ':external:ref:`core-metadata-provides-extra`'
    provides_dist: _Validator[List[str]] = _Validator(added='1.2')
    ':external:ref:`core-metadata-provides-dist`'
    obsoletes_dist: _Validator[List[str]] = _Validator(added='1.2')
    ':external:ref:`core-metadata-obsoletes-dist`'
    requires: _Validator[List[str]] = _Validator(added='1.1')
    '``Requires`` (deprecated)'
    provides: _Validator[List[str]] = _Validator(added='1.1')
    '``Provides`` (deprecated)'
    obsoletes: _Validator[List[str]] = _Validator(added='1.1')
    '``Obsoletes`` (deprecated)'


