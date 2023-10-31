"""PEP 656 support.

This module implements logic to detect if the currently running Python is
linked against musl, and what musl version is used.
"""

import functools
import re
import subprocess
import sys
from typing import Iterator, NamedTuple, Optional, Sequence
from ._elffile import ELFFile


class _MuslVersion(NamedTuple):
    major: int
    minor: int


def _parse_musl_version(output: str) -> Optional[_MuslVersion]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._musllinux._parse_musl_version', '_parse_musl_version(output)', {'re': re, '_MuslVersion': _MuslVersion, 'output': output, 'Optional': Optional, '_MuslVersion': _MuslVersion}, 1)

@functools.lru_cache()
def _get_musl_version(executable: str) -> Optional[_MuslVersion]:
    """Detect currently-running musl runtime version.

    This is done by checking the specified executable's dynamic linking
    information, and invoking the loader to parse its output for a version
    string. If the loader is musl, the output would be something like::

        musl libc (x86_64)
        Version 1.2.2
        Dynamic Program Loader
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._musllinux._get_musl_version', '_get_musl_version(executable)', {'ELFFile': ELFFile, 'subprocess': subprocess, '_parse_musl_version': _parse_musl_version, 'functools': functools, 'executable': executable, 'Optional': Optional, '_MuslVersion': _MuslVersion}, 1)

def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate musllinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be musllinux-compatible.

    :returns: An iterator of compatible musllinux tags.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._musllinux.platform_tags', 'platform_tags(archs)', {'_get_musl_version': _get_musl_version, 'sys': sys, 'archs': archs, 'Sequence': Sequence, 'str': str, 'Iterator': Iterator, 'str': str}, 1)
if __name__ == '__main__':
    import sysconfig
    plat = sysconfig.get_platform()
    assert plat.startswith('linux-'), 'not linux'
    print('plat:', plat)
    print('musl:', _get_musl_version(sys.executable))
    print('tags:', end=' ')
    for t in platform_tags(re.sub('[.-]', '_', plat.split('-', 1)[-1])):
        print(t, end='\n      ')

