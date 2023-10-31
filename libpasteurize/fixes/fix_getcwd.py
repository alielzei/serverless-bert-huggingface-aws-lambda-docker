"""
Fixer for os.getcwd() -> os.getcwdu().
Also warns about "from os import getcwd", suggesting the above form.
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name


class FixGetcwd(fixer_base.BaseFix):
    PATTERN = "\n              power< 'os' trailer< dot='.' name='getcwd' > any* >\n              |\n              import_from< 'from' 'os' 'import' bad='getcwd' >\n              "
    
    def transform(self, node, results):
        if 'name' in results:
            name = results['name']
            name.replace(Name('getcwdu', prefix=name.prefix))
        elif 'bad' in results:
            self.cannot_convert(node, 'import os, use os.getcwd() instead.')
            return
        else:
            raise ValueError('For some reason, the pattern matcher failed.')


