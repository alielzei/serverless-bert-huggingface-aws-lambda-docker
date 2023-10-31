"""Shared support for scanning document type declarations in HTML and XHTML.

Backported for python-future from Python 3.3. Reason: ParserBase is an
old-style class in the Python 2.7 source of markupbase.py, which I suspect
might be the cause of sporadic unit-test failures on travis-ci.org with
test_htmlparser.py.  The test failures look like this:

    ======================================================================

ERROR: test_attr_entity_replacement (future.tests.test_htmlparser.AttributesStrictTestCase)

----------------------------------------------------------------------

Traceback (most recent call last):
  File "/home/travis/build/edschofield/python-future/future/tests/test_htmlparser.py", line 661, in test_attr_entity_replacement
    [("starttag", "a", [("b", "&><"'")])])
  File "/home/travis/build/edschofield/python-future/future/tests/test_htmlparser.py", line 93, in _run_check
    collector = self.get_collector()
  File "/home/travis/build/edschofield/python-future/future/tests/test_htmlparser.py", line 617, in get_collector
    return EventCollector(strict=True)
  File "/home/travis/build/edschofield/python-future/future/tests/test_htmlparser.py", line 27, in __init__
    html.parser.HTMLParser.__init__(self, *args, **kw)
  File "/home/travis/build/edschofield/python-future/future/backports/html/parser.py", line 135, in __init__
    self.reset()
  File "/home/travis/build/edschofield/python-future/future/backports/html/parser.py", line 143, in reset
    _markupbase.ParserBase.reset(self)

TypeError: unbound method reset() must be called with ParserBase instance as first argument (got EventCollector instance instead)

This module is used as a foundation for the html.parser module.  It has no
documented public API and should not be used directly.

"""

import re
_declname_match = re.compile('[a-zA-Z][-_.a-zA-Z0-9]*\\s*').match
_declstringlit_match = re.compile('(\\\'[^\\\']*\\\'|"[^"]*")\\s*').match
_commentclose = re.compile('--\\s*>')
_markedsectionclose = re.compile(']\\s*]\\s*>')
_msmarkedsectionclose = re.compile(']\\s*>')
del re


class ParserBase(object):
    """Parser base class which provides some common support methods used
    by the SGML/HTML and XHTML parsers."""
    
    def __init__(self):
        if self.__class__ is ParserBase:
            raise RuntimeError('_markupbase.ParserBase must be subclassed')
    
    def error(self, message):
        raise NotImplementedError('subclasses of ParserBase must override error()')
    
    def reset(self):
        self.lineno = 1
        self.offset = 0
    
    def getpos(self):
        """Return current line number and offset."""
        return (self.lineno, self.offset)
    
    def updatepos(self, i, j):
        if i >= j:
            return j
        rawdata = self.rawdata
        nlines = rawdata.count('\n', i, j)
        if nlines:
            self.lineno = self.lineno + nlines
            pos = rawdata.rindex('\n', i, j)
            self.offset = j - (pos + 1)
        else:
            self.offset = self.offset + j - i
        return j
    _decl_otherchars = ''
    
    def parse_declaration(self, i):
        rawdata = self.rawdata
        j = i + 2
        assert rawdata[i:j] == '<!', 'unexpected call to parse_declaration'
        if rawdata[j:j + 1] == '>':
            return j + 1
        if rawdata[j:j + 1] in ('-', ''):
            return -1
        n = len(rawdata)
        if rawdata[j:j + 2] == '--':
            return self.parse_comment(i)
        elif rawdata[j] == '[':
            return self.parse_marked_section(i)
        else:
            (decltype, j) = self._scan_name(j, i)
        if j < 0:
            return j
        if decltype == 'doctype':
            self._decl_otherchars = ''
        while j < n:
            c = rawdata[j]
            if c == '>':
                data = rawdata[i + 2:j]
                if decltype == 'doctype':
                    self.handle_decl(data)
                else:
                    self.unknown_decl(data)
                return j + 1
            if c in '"\'':
                m = _declstringlit_match(rawdata, j)
                if not m:
                    return -1
                j = m.end()
            elif c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                (name, j) = self._scan_name(j, i)
            elif c in self._decl_otherchars:
                j = j + 1
            elif c == '[':
                if decltype == 'doctype':
                    j = self._parse_doctype_subset(j + 1, i)
                elif decltype in set(['attlist', 'linktype', 'link', 'element']):
                    self.error("unsupported '[' char in %s declaration" % decltype)
                else:
                    self.error("unexpected '[' char in declaration")
            else:
                self.error('unexpected %r char in declaration' % rawdata[j])
            if j < 0:
                return j
        return -1
    
    def parse_marked_section(self, i, report=1):
        rawdata = self.rawdata
        assert rawdata[i:i + 3] == '<![', 'unexpected call to parse_marked_section()'
        (sectName, j) = self._scan_name(i + 3, i)
        if j < 0:
            return j
        if sectName in set(['temp', 'cdata', 'ignore', 'include', 'rcdata']):
            match = _markedsectionclose.search(rawdata, i + 3)
        elif sectName in set(['if', 'else', 'endif']):
            match = _msmarkedsectionclose.search(rawdata, i + 3)
        else:
            self.error('unknown status keyword %r in marked section' % rawdata[i + 3:j])
        if not match:
            return -1
        if report:
            j = match.start(0)
            self.unknown_decl(rawdata[i + 3:j])
        return match.end(0)
    
    def parse_comment(self, i, report=1):
        rawdata = self.rawdata
        if rawdata[i:i + 4] != '<!--':
            self.error('unexpected call to parse_comment()')
        match = _commentclose.search(rawdata, i + 4)
        if not match:
            return -1
        if report:
            j = match.start(0)
            self.handle_comment(rawdata[i + 4:j])
        return match.end(0)
    
    def _parse_doctype_subset(self, i, declstartpos):
        rawdata = self.rawdata
        n = len(rawdata)
        j = i
        while j < n:
            c = rawdata[j]
            if c == '<':
                s = rawdata[j:j + 2]
                if s == '<':
                    return -1
                if s != '<!':
                    self.updatepos(declstartpos, j + 1)
                    self.error('unexpected char in internal subset (in %r)' % s)
                if j + 2 == n:
                    return -1
                if j + 4 > n:
                    return -1
                if rawdata[j:j + 4] == '<!--':
                    j = self.parse_comment(j, report=0)
                    if j < 0:
                        return j
                    continue
                (name, j) = self._scan_name(j + 2, declstartpos)
                if j == -1:
                    return -1
                if name not in set(['attlist', 'element', 'entity', 'notation']):
                    self.updatepos(declstartpos, j + 2)
                    self.error('unknown declaration %r in internal subset' % name)
                meth = getattr(self, '_parse_doctype_' + name)
                j = meth(j, declstartpos)
                if j < 0:
                    return j
            elif c == '%':
                if j + 1 == n:
                    return -1
                (s, j) = self._scan_name(j + 1, declstartpos)
                if j < 0:
                    return j
                if rawdata[j] == ';':
                    j = j + 1
            elif c == ']':
                j = j + 1
                while (j < n and rawdata[j].isspace()):
                    j = j + 1
                if j < n:
                    if rawdata[j] == '>':
                        return j
                    self.updatepos(declstartpos, j)
                    self.error('unexpected char after internal subset')
                else:
                    return -1
            elif c.isspace():
                j = j + 1
            else:
                self.updatepos(declstartpos, j)
                self.error('unexpected char %r in internal subset' % c)
        return -1
    
    def _parse_doctype_element(self, i, declstartpos):
        (name, j) = self._scan_name(i, declstartpos)
        if j == -1:
            return -1
        rawdata = self.rawdata
        if '>' in rawdata[j:]:
            return rawdata.find('>', j) + 1
        return -1
    
    def _parse_doctype_attlist(self, i, declstartpos):
        rawdata = self.rawdata
        (name, j) = self._scan_name(i, declstartpos)
        c = rawdata[j:j + 1]
        if c == '':
            return -1
        if c == '>':
            return j + 1
        while 1:
            (name, j) = self._scan_name(j, declstartpos)
            if j < 0:
                return j
            c = rawdata[j:j + 1]
            if c == '':
                return -1
            if c == '(':
                if ')' in rawdata[j:]:
                    j = rawdata.find(')', j) + 1
                else:
                    return -1
                while rawdata[j:j + 1].isspace():
                    j = j + 1
                if not rawdata[j:]:
                    return -1
            else:
                (name, j) = self._scan_name(j, declstartpos)
            c = rawdata[j:j + 1]
            if not c:
                return -1
            if c in '\'"':
                m = _declstringlit_match(rawdata, j)
                if m:
                    j = m.end()
                else:
                    return -1
                c = rawdata[j:j + 1]
                if not c:
                    return -1
            if c == '#':
                if rawdata[j:] == '#':
                    return -1
                (name, j) = self._scan_name(j + 1, declstartpos)
                if j < 0:
                    return j
                c = rawdata[j:j + 1]
                if not c:
                    return -1
            if c == '>':
                return j + 1
    
    def _parse_doctype_notation(self, i, declstartpos):
        (name, j) = self._scan_name(i, declstartpos)
        if j < 0:
            return j
        rawdata = self.rawdata
        while 1:
            c = rawdata[j:j + 1]
            if not c:
                return -1
            if c == '>':
                return j + 1
            if c in '\'"':
                m = _declstringlit_match(rawdata, j)
                if not m:
                    return -1
                j = m.end()
            else:
                (name, j) = self._scan_name(j, declstartpos)
                if j < 0:
                    return j
    
    def _parse_doctype_entity(self, i, declstartpos):
        rawdata = self.rawdata
        if rawdata[i:i + 1] == '%':
            j = i + 1
            while 1:
                c = rawdata[j:j + 1]
                if not c:
                    return -1
                if c.isspace():
                    j = j + 1
                else:
                    break
        else:
            j = i
        (name, j) = self._scan_name(j, declstartpos)
        if j < 0:
            return j
        while 1:
            c = self.rawdata[j:j + 1]
            if not c:
                return -1
            if c in '\'"':
                m = _declstringlit_match(rawdata, j)
                if m:
                    j = m.end()
                else:
                    return -1
            elif c == '>':
                return j + 1
            else:
                (name, j) = self._scan_name(j, declstartpos)
                if j < 0:
                    return j
    
    def _scan_name(self, i, declstartpos):
        rawdata = self.rawdata
        n = len(rawdata)
        if i == n:
            return (None, -1)
        m = _declname_match(rawdata, i)
        if m:
            s = m.group()
            name = s.strip()
            if i + len(s) == n:
                return (None, -1)
            return (name.lower(), m.end())
        else:
            self.updatepos(declstartpos, i)
            self.error('expected name token at %r' % rawdata[declstartpos:declstartpos + 20])
    
    def unknown_decl(self, data):
        pass


