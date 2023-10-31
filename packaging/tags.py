import logging
import platform
import struct
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, cast
from . import _manylinux, _musllinux
logger = logging.getLogger(__name__)
PythonVersion = Sequence[int]
MacVersion = Tuple[(int, int)]
INTERPRETER_SHORT_NAMES: Dict[(str, str)] = {'python': 'py', 'cpython': 'cp', 'pypy': 'pp', 'ironpython': 'ip', 'jython': 'jy'}
_32_BIT_INTERPRETER = struct.calcsize('P') == 4


class Tag:
    """
    A representation of the tag triple for a wheel.

    Instances are considered immutable and thus are hashable. Equality checking
    is also supported.
    """
    __slots__ = ['_interpreter', '_abi', '_platform', '_hash']
    
    def __init__(self, interpreter: str, abi: str, platform: str) -> None:
        self._interpreter = interpreter.lower()
        self._abi = abi.lower()
        self._platform = platform.lower()
        self._hash = hash((self._interpreter, self._abi, self._platform))
    
    @property
    def interpreter(self) -> str:
        return self._interpreter
    
    @property
    def abi(self) -> str:
        return self._abi
    
    @property
    def platform(self) -> str:
        return self._platform
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented
        return (self._hash == other._hash and self._platform == other._platform and self._abi == other._abi and self._interpreter == other._interpreter)
    
    def __hash__(self) -> int:
        return self._hash
    
    def __str__(self) -> str:
        return f'{self._interpreter}-{self._abi}-{self._platform}'
    
    def __repr__(self) -> str:
        return f'<{self} @ {id(self)}>'


def parse_tag(tag: str) -> FrozenSet[Tag]:
    """
    Parses the provided tag (e.g. `py3-none-any`) into a frozenset of Tag instances.

    Returning a set is required due to the possibility that the tag is a
    compressed tag set.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags.parse_tag', 'parse_tag(tag)', {'Tag': Tag, 'tag': tag, 'FrozenSet': FrozenSet, 'Tag': Tag}, 1)

def _get_config_var(name: str, warn: bool = False) -> Union[(int, str, None)]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags._get_config_var', '_get_config_var(name, warn=False)', {'Union': Union, 'sysconfig': sysconfig, 'logger': logger, 'name': name, 'warn': warn, 'Union': Union}, 1)

def _normalize_string(string: str) -> str:
    return string.replace('.', '_').replace('-', '_').replace(' ', '_')

def _abi3_applies(python_version: PythonVersion) -> bool:
    """
    Determine if the Python version supports abi3.

    PEP 384 was first implemented in Python 3.2.
    """
    return (len(python_version) > 1 and tuple(python_version) >= (3, 2))

def _cpython_abis(py_version: PythonVersion, warn: bool = False) -> List[str]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags._cpython_abis', '_cpython_abis(py_version, warn=False)', {'_version_nodot': _version_nodot, '_get_config_var': _get_config_var, 'sys': sys, 'EXTENSION_SUFFIXES': EXTENSION_SUFFIXES, 'py_version': py_version, 'warn': warn, 'List': List, 'str': str}, 1)

def cpython_tags(python_version: Optional[PythonVersion] = None, abis: Optional[Iterable[str]] = None, platforms: Optional[Iterable[str]] = None, *, warn: bool = False) -> Iterator[Tag]:
    """
    Yields the tags for a CPython interpreter.

    The tags consist of:
    - cp<python_version>-<abi>-<platform>
    - cp<python_version>-abi3-<platform>
    - cp<python_version>-none-<platform>
    - cp<less than python_version>-abi3-<platform>  # Older Python versions down to 3.2.

    If python_version only specifies a major version then user-provided ABIs and
    the 'none' ABItag will be used.

    If 'abi3' or 'none' are specified in 'abis' then they will be yielded at
    their normal position and not at the beginning.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('packaging.tags.cpython_tags', 'cpython_tags(python_version=None, abis=None, platforms=None, warn: bool = False)', {'sys': sys, '_version_nodot': _version_nodot, '_cpython_abis': _cpython_abis, 'platform_tags': platform_tags, 'Tag': Tag, '_abi3_applies': _abi3_applies, 'python_version': python_version, 'abis': abis, 'platforms': platforms, 'warn': warn, 'Optional': Optional, 'PythonVersion': PythonVersion, 'Optional': Optional, 'Optional': Optional, 'Iterator': Iterator, 'Tag': Tag}, 0)

def _generic_abi() -> List[str]:
    """
    Return the ABI tag based on EXT_SUFFIX.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags._generic_abi', '_generic_abi()', {'_get_config_var': _get_config_var, '_cpython_abis': _cpython_abis, 'sys': sys, '_normalize_string': _normalize_string, 'List': List, 'str': str}, 1)

def generic_tags(interpreter: Optional[str] = None, abis: Optional[Iterable[str]] = None, platforms: Optional[Iterable[str]] = None, *, warn: bool = False) -> Iterator[Tag]:
    """
    Yields the tags for a generic interpreter.

    The tags consist of:
    - <interpreter>-<abi>-<platform>

    The "none" ABI will be added if it was not explicitly provided.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('packaging.tags.generic_tags', 'generic_tags(interpreter=None, abis=None, platforms=None, warn: bool = False)', {'interpreter_name': interpreter_name, 'interpreter_version': interpreter_version, '_generic_abi': _generic_abi, 'platform_tags': platform_tags, 'Tag': Tag, 'interpreter': interpreter, 'abis': abis, 'platforms': platforms, 'warn': warn, 'Optional': Optional, 'str': str, 'Optional': Optional, 'Optional': Optional, 'Iterator': Iterator, 'Tag': Tag}, 0)

def _py_interpreter_range(py_version: PythonVersion) -> Iterator[str]:
    """
    Yields Python versions in descending order.

    After the latest version, the major-only version will be yielded, and then
    all previous versions of that major version.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('packaging.tags._py_interpreter_range', '_py_interpreter_range(py_version)', {'_version_nodot': _version_nodot, 'py_version': py_version, 'Iterator': Iterator, 'str': str}, 0)

def compatible_tags(python_version: Optional[PythonVersion] = None, interpreter: Optional[str] = None, platforms: Optional[Iterable[str]] = None) -> Iterator[Tag]:
    """
    Yields the sequence of tags that are compatible with a specific version of Python.

    The tags consist of:
    - py*-none-<platform>
    - <interpreter>-none-any  # ... if `interpreter` is provided.
    - py*-none-any
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('packaging.tags.compatible_tags', 'compatible_tags(python_version=None, interpreter=None, platforms=None)', {'sys': sys, 'platform_tags': platform_tags, '_py_interpreter_range': _py_interpreter_range, 'Tag': Tag, 'python_version': python_version, 'interpreter': interpreter, 'platforms': platforms, 'Optional': Optional, 'PythonVersion': PythonVersion, 'Optional': Optional, 'str': str, 'Optional': Optional, 'Iterator': Iterator, 'Tag': Tag}, 0)

def _mac_arch(arch: str, is_32bit: bool = _32_BIT_INTERPRETER) -> str:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags._mac_arch', '_mac_arch(arch, is_32bit=_32_BIT_INTERPRETER)', {'arch': arch, 'is_32bit': is_32bit, '_32_BIT_INTERPRETER': _32_BIT_INTERPRETER}, 1)

def _mac_binary_formats(version: MacVersion, cpu_arch: str) -> List[str]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags._mac_binary_formats', '_mac_binary_formats(version, cpu_arch)', {'version': version, 'cpu_arch': cpu_arch, 'List': List, 'str': str}, 1)

def mac_platforms(version: Optional[MacVersion] = None, arch: Optional[str] = None) -> Iterator[str]:
    """
    Yields the platform tags for a macOS system.

    The `version` parameter is a two-item tuple specifying the macOS version to
    generate platform tags for. The `arch` parameter is the CPU architecture to
    generate platform tags for. Both parameters default to the appropriate value
    for the current system.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('packaging.tags.mac_platforms', 'mac_platforms(version=None, arch=None)', {'platform': platform, 'subprocess': subprocess, 'sys': sys, '_mac_arch': _mac_arch, '_mac_binary_formats': _mac_binary_formats, 'version': version, 'arch': arch, 'Optional': Optional, 'MacVersion': MacVersion, 'Optional': Optional, 'str': str, 'Iterator': Iterator, 'str': str}, 0)

def _linux_platforms(is_32bit: bool = _32_BIT_INTERPRETER) -> Iterator[str]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags._linux_platforms', '_linux_platforms(is_32bit=_32_BIT_INTERPRETER)', {'_normalize_string': _normalize_string, 'sysconfig': sysconfig, '_manylinux': _manylinux, '_musllinux': _musllinux, 'is_32bit': is_32bit, '_32_BIT_INTERPRETER': _32_BIT_INTERPRETER, 'Iterator': Iterator, 'str': str}, 1)

def _generic_platforms() -> Iterator[str]:
    yield _normalize_string(sysconfig.get_platform())

def platform_tags() -> Iterator[str]:
    """
    Provides the platform tags for this installation.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags.platform_tags', 'platform_tags()', {'platform': platform, 'mac_platforms': mac_platforms, '_linux_platforms': _linux_platforms, '_generic_platforms': _generic_platforms, 'Iterator': Iterator, 'str': str}, 1)

def interpreter_name() -> str:
    """
    Returns the name of the running interpreter.

    Some implementations have a reserved, two-letter abbreviation which will
    be returned when appropriate.
    """
    name = sys.implementation.name
    return (INTERPRETER_SHORT_NAMES.get(name) or name)

def interpreter_version(*, warn: bool = False) -> str:
    """
    Returns the version of the running interpreter.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.tags.interpreter_version', 'interpreter_version(*, warn: bool = False)', {'_get_config_var': _get_config_var, '_version_nodot': _version_nodot, 'sys': sys, 'warn': warn}, 1)

def _version_nodot(version: PythonVersion) -> str:
    return ''.join(map(str, version))

def sys_tags(*, warn: bool = False) -> Iterator[Tag]:
    """
    Returns the sequence of tag triples for the running interpreter.

    The order of the sequence corresponds to priority order for the
    interpreter, from most to least important.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('packaging.tags.sys_tags', 'sys_tags(*, warn: bool = False)', {'interpreter_name': interpreter_name, 'cpython_tags': cpython_tags, 'generic_tags': generic_tags, 'interpreter_version': interpreter_version, 'compatible_tags': compatible_tags, 'warn': warn, 'Iterator': Iterator, 'Tag': Tag}, 0)

