"""
Fixer for complicated imports
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, String, FromImport, Newline, Comma
from libfuturize.fixer_util import touch_import_top
TK_BASE_NAMES = ('ACTIVE', 'ALL', 'ANCHOR', 'ARC', 'BASELINE', 'BEVEL', 'BOTH', 'BOTTOM', 'BROWSE', 'BUTT', 'CASCADE', 'CENTER', 'CHAR', 'CHECKBUTTON', 'CHORD', 'COMMAND', 'CURRENT', 'DISABLED', 'DOTBOX', 'E', 'END', 'EW', 'EXCEPTION', 'EXTENDED', 'FALSE', 'FIRST', 'FLAT', 'GROOVE', 'HIDDEN', 'HORIZONTAL', 'INSERT', 'INSIDE', 'LAST', 'LEFT', 'MITER', 'MOVETO', 'MULTIPLE', 'N', 'NE', 'NO', 'NONE', 'NORMAL', 'NS', 'NSEW', 'NUMERIC', 'NW', 'OFF', 'ON', 'OUTSIDE', 'PAGES', 'PIESLICE', 'PROJECTING', 'RADIOBUTTON', 'RAISED', 'READABLE', 'RIDGE', 'RIGHT', 'ROUND', 'S', 'SCROLL', 'SE', 'SEL', 'SEL_FIRST', 'SEL_LAST', 'SEPARATOR', 'SINGLE', 'SOLID', 'SUNKEN', 'SW', 'StringTypes', 'TOP', 'TRUE', 'TclVersion', 'TkVersion', 'UNDERLINE', 'UNITS', 'VERTICAL', 'W', 'WORD', 'WRITABLE', 'X', 'Y', 'YES', 'wantobjects')
PY2MODULES = {'urllib2': ('AbstractBasicAuthHandler', 'AbstractDigestAuthHandler', 'AbstractHTTPHandler', 'BaseHandler', 'CacheFTPHandler', 'FTPHandler', 'FileHandler', 'HTTPBasicAuthHandler', 'HTTPCookieProcessor', 'HTTPDefaultErrorHandler', 'HTTPDigestAuthHandler', 'HTTPError', 'HTTPErrorProcessor', 'HTTPHandler', 'HTTPPasswordMgr', 'HTTPPasswordMgrWithDefaultRealm', 'HTTPRedirectHandler', 'HTTPSHandler', 'OpenerDirector', 'ProxyBasicAuthHandler', 'ProxyDigestAuthHandler', 'ProxyHandler', 'Request', 'StringIO', 'URLError', 'UnknownHandler', 'addinfourl', 'build_opener', 'install_opener', 'parse_http_list', 'parse_keqv_list', 'randombytes', 'request_host', 'urlopen'), 'urllib': ('ContentTooShortError', 'FancyURLopener', 'URLopener', 'basejoin', 'ftperrors', 'getproxies', 'getproxies_environment', 'localhost', 'pathname2url', 'quote', 'quote_plus', 'splitattr', 'splithost', 'splitnport', 'splitpasswd', 'splitport', 'splitquery', 'splittag', 'splittype', 'splituser', 'splitvalue', 'thishost', 'unquote', 'unquote_plus', 'unwrap', 'url2pathname', 'urlcleanup', 'urlencode', 'urlopen', 'urlretrieve'), 'urlparse': ('parse_qs', 'parse_qsl', 'urldefrag', 'urljoin', 'urlparse', 'urlsplit', 'urlunparse', 'urlunsplit'), 'dbm': ('ndbm', 'gnu', 'dumb'), 'anydbm': ('error', 'open'), 'whichdb': ('whichdb', ), 'BaseHTTPServer': ('BaseHTTPRequestHandler', 'HTTPServer'), 'CGIHTTPServer': ('CGIHTTPRequestHandler', ), 'SimpleHTTPServer': ('SimpleHTTPRequestHandler', ), 'FileDialog': TK_BASE_NAMES + ('FileDialog', 'LoadFileDialog', 'SaveFileDialog', 'dialogstates', 'test'), 'tkFileDialog': ('Directory', 'Open', 'SaveAs', '_Dialog', 'askdirectory', 'askopenfile', 'askopenfilename', 'askopenfilenames', 'askopenfiles', 'asksaveasfile', 'asksaveasfilename'), 'SimpleDialog': TK_BASE_NAMES + ('SimpleDialog', ), 'tkSimpleDialog': TK_BASE_NAMES + ('askfloat', 'askinteger', 'askstring', 'Dialog'), 'SimpleXMLRPCServer': ('CGIXMLRPCRequestHandler', 'SimpleXMLRPCDispatcher', 'SimpleXMLRPCRequestHandler', 'SimpleXMLRPCServer', 'list_public_methods', 'remove_duplicates', 'resolve_dotted_attribute'), 'DocXMLRPCServer': ('DocCGIXMLRPCRequestHandler', 'DocXMLRPCRequestHandler', 'DocXMLRPCServer', 'ServerHTMLDoc', 'XMLRPCDocGenerator')}
MAPPING = {'urllib.request': ('urllib2', 'urllib'), 'urllib.error': ('urllib2', 'urllib'), 'urllib.parse': ('urllib2', 'urllib', 'urlparse'), 'dbm.__init__': ('anydbm', 'whichdb'), 'http.server': ('CGIHTTPServer', 'SimpleHTTPServer', 'BaseHTTPServer'), 'tkinter.filedialog': ('tkFileDialog', 'FileDialog'), 'tkinter.simpledialog': ('tkSimpleDialog', 'SimpleDialog'), 'xmlrpc.server': ('DocXMLRPCServer', 'SimpleXMLRPCServer')}
simple_name = "name='%s'"
simple_attr = "attr='%s'"
simple_using = "using='%s'"
dotted_name = "dotted_name=dotted_name< %s '.' %s >"
power_twoname = "pow=power< %s trailer< '.' %s > trailer< '.' using=any > any* >"
power_onename = "pow=power< %s trailer< '.' using=any > any* >"
from_import = "from_import=import_from< 'from' %s 'import' (import_as_name< using=any 'as' renamed=any> | in_list=import_as_names< using=any* > | using='*' | using=NAME) >"
name_import = "name_import=import_name< 'import' (%s | in_list=dotted_as_names< imp_list=any* >) >"
name_import_rename = "name_import_rename=dotted_as_name< %s 'as' renamed=any >"
from_import_rename = "from_import_rename=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | in_list=import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* >) >"

def all_modules_subpattern():
    """
    Builds a pattern for all toplevel names
    (urllib, http, etc)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_imports2.all_modules_subpattern', 'all_modules_subpattern()', {'MAPPING': MAPPING, 'dotted_name': dotted_name, 'simple_name': simple_name, 'simple_attr': simple_attr}, 1)

def build_import_pattern(mapping1, mapping2):
    """
    mapping1: A dict mapping py3k modules to all possible py2k replacements
    mapping2: A dict mapping py2k modules to the things they do
    This builds a HUGE pattern to match all ways that things can be imported
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_imports2.build_import_pattern', 'build_import_pattern(mapping1, mapping2)', {'from_import': from_import, 'all_modules_subpattern': all_modules_subpattern, 'simple_name': simple_name, 'simple_attr': simple_attr, 'dotted_name': dotted_name, 'name_import': name_import, 'power_twoname': power_twoname, 'power_onename': power_onename, 'name_import_rename': name_import_rename, 'from_import_rename': from_import_rename, 'mapping1': mapping1, 'mapping2': mapping2}, 0)


class FixImports2(fixer_base.BaseFix):
    run_order = 4
    PATTERN = ' | \n'.join(build_import_pattern(MAPPING, PY2MODULES))
    
    def transform(self, node, results):
        touch_import_top('future', 'standard_library', node)


