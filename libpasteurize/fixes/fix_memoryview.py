"""
Fixer for memoryview(s) -> buffer(s).
Explicit because some memoryview methods are invalid on buffer objects.
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name


class FixMemoryview(fixer_base.BaseFix):
    explicit = True
    PATTERN = "\n              power< name='memoryview' trailer< '(' [any] ')' >\n              rest=any* >\n              "
    
    def transform(self, node, results):
        name = results['name']
        name.replace(Name('buffer', prefix=name.prefix))


