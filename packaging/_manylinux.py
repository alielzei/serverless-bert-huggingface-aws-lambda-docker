import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Sequence, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
EF_ARM_ABIMASK = 4278190080
EF_ARM_ABI_VER5 = 83886080
EF_ARM_ABI_FLOAT_HARD = 1024

@contextlib.contextmanager
def _parse_elf(path: str) -> Generator[(Optional[ELFFile], None, None)]:
    import custom_funtemplate
    custom_funtemplate.rewrite_template('packaging._manylinux._parse_elf', '_parse_elf(path)', {'ELFFile': ELFFile, 'contextlib': contextlib, 'path': path, 'Generator': Generator}, 0)

def _is_linux_armhf(executable: str) -> bool:
    with _parse_elf(executable) as f:
        return (f is not None and f.capacity == EIClass.C32 and f.encoding == EIData.Lsb and f.machine == EMachine.Arm and f.flags & EF_ARM_ABIMASK == EF_ARM_ABI_VER5 and f.flags & EF_ARM_ABI_FLOAT_HARD == EF_ARM_ABI_FLOAT_HARD)

def _is_linux_i686(executable: str) -> bool:
    with _parse_elf(executable) as f:
        return (f is not None and f.capacity == EIClass.C32 and f.encoding == EIData.Lsb and f.machine == EMachine.I386)

def _have_compatible_abi(executable: str, archs: Sequence[str]) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._manylinux._have_compatible_abi', '_have_compatible_abi(executable, archs)', {'_is_linux_armhf': _is_linux_armhf, '_is_linux_i686': _is_linux_i686, 'executable': executable, 'archs': archs, 'Sequence': Sequence, 'str': str}, 1)
_LAST_GLIBC_MINOR: Dict[(int, int)] = collections.defaultdict(lambda: 50)


class _GLibCVersion(NamedTuple):
    major: int
    minor: int


def _glibc_version_string_confstr() -> Optional[str]:
    """
    Primary implementation of glibc_version_string using os.confstr.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._manylinux._glibc_version_string_confstr', '_glibc_version_string_confstr()', {'os': os, 'Optional': Optional, 'str': str}, 1)

def _glibc_version_string_ctypes() -> Optional[str]:
    """
    Fallback implementation of glibc_version_string using ctypes.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._manylinux._glibc_version_string_ctypes', '_glibc_version_string_ctypes()', {'Optional': Optional, 'str': str}, 1)

def _glibc_version_string() -> Optional[str]:
    """Returns glibc version string, or None if not using glibc."""
    return (_glibc_version_string_confstr() or _glibc_version_string_ctypes())

def _parse_glibc_version(version_str: str) -> Tuple[(int, int)]:
    """Parse glibc version.

    We use a regexp instead of str.split because we want to discard any
    random junk that might come after the minor version -- this might happen
    in patched/forked versions of glibc (e.g. Linaro's version of glibc
    uses version strings like "2.20-2014.11"). See gh-3588.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._manylinux._parse_glibc_version', '_parse_glibc_version(version_str)', {'re': re, 'warnings': warnings, 'version_str': version_str, 'Tuple': Tuple}, 2)

@functools.lru_cache()
def _get_glibc_version() -> Tuple[(int, int)]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._manylinux._get_glibc_version', '_get_glibc_version()', {'_glibc_version_string': _glibc_version_string, '_parse_glibc_version': _parse_glibc_version, 'functools': functools, 'Tuple': Tuple}, 2)

def _is_compatible(arch: str, version: _GLibCVersion) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._manylinux._is_compatible', '_is_compatible(arch, version)', {'_get_glibc_version': _get_glibc_version, '_GLibCVersion': _GLibCVersion, 'arch': arch, 'version': version}, 1)
_LEGACY_MANYLINUX_MAP = {(2, 17): 'manylinux2014', (2, 12): 'manylinux2010', (2, 5): 'manylinux1'}

def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate manylinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be manylinux-compatible.

    :returns: An iterator of compatible manylinux tags.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._manylinux.platform_tags', 'platform_tags(archs)', {'_have_compatible_abi': _have_compatible_abi, 'sys': sys, '_GLibCVersion': _GLibCVersion, '_get_glibc_version': _get_glibc_version, '_LAST_GLIBC_MINOR': _LAST_GLIBC_MINOR, '_is_compatible': _is_compatible, '_LEGACY_MANYLINUX_MAP': _LEGACY_MANYLINUX_MAP, 'archs': archs, 'Sequence': Sequence, 'str': str, 'Iterator': Iterator, 'str': str}, 1)

