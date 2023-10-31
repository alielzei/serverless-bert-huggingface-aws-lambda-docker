"""
Fixer for standard library imports renamed in Python 3
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, is_probably_builtin, Newline, does_tree_import
from lib2to3.pygram import python_symbols as syms
from lib2to3.pgen2 import token
from lib2to3.pytree import Node, Leaf
from libfuturize.fixer_util import touch_import_top
MAPPING = {'reprlib': 'repr', 'winreg': '_winreg', 'configparser': 'ConfigParser', 'copyreg': 'copy_reg', 'queue': 'Queue', 'socketserver': 'SocketServer', '_markupbase': 'markupbase', 'test.support': 'test.test_support', 'dbm.bsd': 'dbhash', 'dbm.ndbm': 'dbm', 'dbm.dumb': 'dumbdbm', 'dbm.gnu': 'gdbm', 'html.parser': 'HTMLParser', 'html.entities': 'htmlentitydefs', 'http.client': 'httplib', 'http.cookies': 'Cookie', 'http.cookiejar': 'cookielib', 'tkinter.dialog': 'Dialog', 'tkinter._fix': 'FixTk', 'tkinter.scrolledtext': 'ScrolledText', 'tkinter.tix': 'Tix', 'tkinter.constants': 'Tkconstants', 'tkinter.dnd': 'Tkdnd', 'tkinter.__init__': 'Tkinter', 'tkinter.colorchooser': 'tkColorChooser', 'tkinter.commondialog': 'tkCommonDialog', 'tkinter.font': 'tkFont', 'tkinter.ttk': 'ttk', 'tkinter.messagebox': 'tkMessageBox', 'tkinter.turtle': 'turtle', 'urllib.robotparser': 'robotparser', 'xmlrpc.client': 'xmlrpclib', 'builtins': '__builtin__'}
simple_name_match = "name='%s'"
subname_match = "attr='%s'"
dotted_name_match = "dotted_name=dotted_name< %s '.' %s >"
power_onename_match = '%s'
power_twoname_match = "power< %s trailer< '.' %s > any* >"
power_subname_match = 'power< %s any* >'
from_import_match = "from_import=import_from< 'from' %s 'import' imported=any >"
from_import_submod_match = "from_import_submod=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* > ) >"
name_import_match = "name_import=import_name< 'import' %s > | name_import=import_name< 'import' dotted_as_name< %s 'as' renamed=any > >"
multiple_name_import_match = "name_import=import_name< 'import' dotted_as_names< names=any* > >"

def all_patterns(name):
    """
    Accepts a string and returns a pattern of possible patterns involving that name
    Called by simple_mapping_to_pattern for each name in the mapping it receives.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_imports.all_patterns', 'all_patterns(name)', {'simple_name_match': simple_name_match, 'subname_match': subname_match, 'dotted_name_match': dotted_name_match, 'from_import_match': from_import_match, 'from_import_submod_match': from_import_submod_match, 'name_import_match': name_import_match, 'power_twoname_match': power_twoname_match, 'power_subname_match': power_subname_match, 'power_onename_match': power_onename_match, 'name': name}, 1)


class FixImports(fixer_base.BaseFix):
    PATTERN = ' | \n'.join([all_patterns(name) for name in MAPPING])
    PATTERN = ' | \n'.join((PATTERN, multiple_name_import_match))
    
    def transform(self, node, results):
        touch_import_top('future', 'standard_library', node)


