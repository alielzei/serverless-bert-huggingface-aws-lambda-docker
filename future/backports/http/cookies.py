"""
http.cookies module ported to python-future from Py3.3

Here's a sample session to show how to use this module.
At the moment, this is the only documentation.

The Basics
----------

Importing is easy...

   >>> from http import cookies

Most of the time you start by creating a cookie.

   >>> C = cookies.SimpleCookie()

Once you've created your Cookie, you can add values just as if it were
a dictionary.

   >>> C = cookies.SimpleCookie()
   >>> C["fig"] = "newton"
   >>> C["sugar"] = "wafer"
   >>> C.output()
   'Set-Cookie: fig=newton
Set-Cookie: sugar=wafer'

Notice that the printable representation of a Cookie is the
appropriate format for a Set-Cookie: header.  This is the
default behavior.  You can change the header and printed
attributes by using the .output() function

   >>> C = cookies.SimpleCookie()
   >>> C["rocky"] = "road"
   >>> C["rocky"]["path"] = "/cookie"
   >>> print(C.output(header="Cookie:"))
   Cookie: rocky=road; Path=/cookie
   >>> print(C.output(attrs=[], header="Cookie:"))
   Cookie: rocky=road

The load() method of a Cookie extracts cookies from a string.  In a
CGI script, you would use this method to extract the cookies from the
HTTP_COOKIE environment variable.

   >>> C = cookies.SimpleCookie()
   >>> C.load("chips=ahoy; vienna=finger")
   >>> C.output()
   'Set-Cookie: chips=ahoy
Set-Cookie: vienna=finger'

The load() method is darn-tootin smart about identifying cookies
within a string.  Escaped quotation marks, nested semicolons, and other
such trickeries do not confuse it.

   >>> C = cookies.SimpleCookie()
   >>> C.load('keebler="E=everybody; L=\"Loves\"; fudge=\012;";')
   >>> print(C)
   Set-Cookie: keebler="E=everybody; L="Loves"; fudge=
;"

Each element of the Cookie also supports all of the RFC 2109
Cookie attributes.  Here's an example which sets the Path
attribute.

   >>> C = cookies.SimpleCookie()
   >>> C["oreo"] = "doublestuff"
   >>> C["oreo"]["path"] = "/"
   >>> print(C)
   Set-Cookie: oreo=doublestuff; Path=/

Each dictionary element has a 'value' attribute, which gives you
back the value associated with the key.

   >>> C = cookies.SimpleCookie()
   >>> C["twix"] = "none for you"
   >>> C["twix"].value
   'none for you'

The SimpleCookie expects that all values should be standard strings.
Just to be sure, SimpleCookie invokes the str() builtin to convert
the value to a string, when the values are set dictionary-style.

   >>> C = cookies.SimpleCookie()
   >>> C["number"] = 7
   >>> C["string"] = "seven"
   >>> C["number"].value
   '7'
   >>> C["string"].value
   'seven'
   >>> C.output()
   'Set-Cookie: number=7
Set-Cookie: string=seven'

Finis.
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import chr, dict, int, str
from future.utils import PY2, as_native_str
import re
if PY2:
    re.ASCII = 0
import string
__all__ = ['CookieError', 'BaseCookie', 'SimpleCookie']
_nulljoin = ''.join
_semispacejoin = '; '.join
_spacejoin = ' '.join


class CookieError(Exception):
    pass

_LegalChars = string.ascii_letters + string.digits + "!#$%&'*+-.^_`|~:"
_Translator = {'\x00': '\\000', '\x01': '\\001', '\x02': '\\002', '\x03': '\\003', '\x04': '\\004', '\x05': '\\005', '\x06': '\\006', '\x07': '\\007', '\x08': '\\010', '\t': '\\011', '\n': '\\012', '\x0b': '\\013', '\x0c': '\\014', '\r': '\\015', '\x0e': '\\016', '\x0f': '\\017', '\x10': '\\020', '\x11': '\\021', '\x12': '\\022', '\x13': '\\023', '\x14': '\\024', '\x15': '\\025', '\x16': '\\026', '\x17': '\\027', '\x18': '\\030', '\x19': '\\031', '\x1a': '\\032', '\x1b': '\\033', '\x1c': '\\034', '\x1d': '\\035', '\x1e': '\\036', '\x1f': '\\037', ',': '\\054', ';': '\\073', '"': '\\"', '\\': '\\\\', '\x7f': '\\177', '\x80': '\\200', '\x81': '\\201', '\x82': '\\202', '\x83': '\\203', '\x84': '\\204', '\x85': '\\205', '\x86': '\\206', '\x87': '\\207', '\x88': '\\210', '\x89': '\\211', '\x8a': '\\212', '\x8b': '\\213', '\x8c': '\\214', '\x8d': '\\215', '\x8e': '\\216', '\x8f': '\\217', '\x90': '\\220', '\x91': '\\221', '\x92': '\\222', '\x93': '\\223', '\x94': '\\224', '\x95': '\\225', '\x96': '\\226', '\x97': '\\227', '\x98': '\\230', '\x99': '\\231', '\x9a': '\\232', '\x9b': '\\233', '\x9c': '\\234', '\x9d': '\\235', '\x9e': '\\236', '\x9f': '\\237', '\xa0': '\\240', '¡': '\\241', '¢': '\\242', '£': '\\243', '¤': '\\244', '¥': '\\245', '¦': '\\246', '§': '\\247', '¨': '\\250', '©': '\\251', 'ª': '\\252', '«': '\\253', '¬': '\\254', '\xad': '\\255', '®': '\\256', '¯': '\\257', '°': '\\260', '±': '\\261', '²': '\\262', '³': '\\263', '´': '\\264', 'µ': '\\265', '¶': '\\266', '·': '\\267', '¸': '\\270', '¹': '\\271', 'º': '\\272', '»': '\\273', '¼': '\\274', '½': '\\275', '¾': '\\276', '¿': '\\277', 'À': '\\300', 'Á': '\\301', 'Â': '\\302', 'Ã': '\\303', 'Ä': '\\304', 'Å': '\\305', 'Æ': '\\306', 'Ç': '\\307', 'È': '\\310', 'É': '\\311', 'Ê': '\\312', 'Ë': '\\313', 'Ì': '\\314', 'Í': '\\315', 'Î': '\\316', 'Ï': '\\317', 'Ð': '\\320', 'Ñ': '\\321', 'Ò': '\\322', 'Ó': '\\323', 'Ô': '\\324', 'Õ': '\\325', 'Ö': '\\326', '×': '\\327', 'Ø': '\\330', 'Ù': '\\331', 'Ú': '\\332', 'Û': '\\333', 'Ü': '\\334', 'Ý': '\\335', 'Þ': '\\336', 'ß': '\\337', 'à': '\\340', 'á': '\\341', 'â': '\\342', 'ã': '\\343', 'ä': '\\344', 'å': '\\345', 'æ': '\\346', 'ç': '\\347', 'è': '\\350', 'é': '\\351', 'ê': '\\352', 'ë': '\\353', 'ì': '\\354', 'í': '\\355', 'î': '\\356', 'ï': '\\357', 'ð': '\\360', 'ñ': '\\361', 'ò': '\\362', 'ó': '\\363', 'ô': '\\364', 'õ': '\\365', 'ö': '\\366', '÷': '\\367', 'ø': '\\370', 'ù': '\\371', 'ú': '\\372', 'û': '\\373', 'ü': '\\374', 'ý': '\\375', 'þ': '\\376', 'ÿ': '\\377'}

def _quote(str, LegalChars=_LegalChars):
    """Quote a string for use in a cookie header.

    If the string does not need to be double-quoted, then just return the
    string.  Otherwise, surround the string in doublequotes and quote
    (with a \) special characters.
    """
    if all((c in LegalChars for c in str)):
        return str
    else:
        return '"' + _nulljoin((_Translator.get(s, s) for s in str)) + '"'
_OctalPatt = re.compile('\\\\[0-3][0-7][0-7]')
_QuotePatt = re.compile('[\\\\].')

def _unquote(mystr):
    if len(mystr) < 2:
        return mystr
    if (mystr[0] != '"' or mystr[-1] != '"'):
        return mystr
    mystr = mystr[1:-1]
    i = 0
    n = len(mystr)
    res = []
    while 0 <= i < n:
        o_match = _OctalPatt.search(mystr, i)
        q_match = _QuotePatt.search(mystr, i)
        if (not o_match and not q_match):
            res.append(mystr[i:])
            break
        j = k = -1
        if o_match:
            j = o_match.start(0)
        if q_match:
            k = q_match.start(0)
        if (q_match and ((not o_match or k < j))):
            res.append(mystr[i:k])
            res.append(mystr[k + 1])
            i = k + 2
        else:
            res.append(mystr[i:j])
            res.append(chr(int(mystr[j + 1:j + 4], 8)))
            i = j + 4
    return _nulljoin(res)
_weekdayname = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
_monthname = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def _getdate(future=0, weekdayname=_weekdayname, monthname=_monthname):
    from time import gmtime, time
    now = time()
    (year, month, day, hh, mm, ss, wd, y, z) = gmtime(now + future)
    return '%s, %02d %3s %4d %02d:%02d:%02d GMT' % (weekdayname[wd], day, monthname[month], year, hh, mm, ss)


class Morsel(dict):
    """A class to hold ONE (key, value) pair.

    In a cookie, each such pair may have several attributes, so this class is
    used to keep the attributes associated with the appropriate key,value pair.
    This class also includes a coded_value attribute, which is used to hold
    the network representation of the value.  This is most useful when Python
    objects are pickled for network transit.
    """
    _reserved = {'expires': 'expires', 'path': 'Path', 'comment': 'Comment', 'domain': 'Domain', 'max-age': 'Max-Age', 'secure': 'secure', 'httponly': 'httponly', 'version': 'Version'}
    _flags = set(['secure', 'httponly'])
    
    def __init__(self):
        self.key = self.value = self.coded_value = None
        for key in self._reserved:
            dict.__setitem__(self, key, '')
    
    def __setitem__(self, K, V):
        K = K.lower()
        if not K in self._reserved:
            raise CookieError('Invalid Attribute %s' % K)
        dict.__setitem__(self, K, V)
    
    def isReservedKey(self, K):
        return K.lower() in self._reserved
    
    def set(self, key, val, coded_val, LegalChars=_LegalChars):
        if key.lower() in self._reserved:
            raise CookieError('Attempt to set a reserved key: %s' % key)
        if any((c not in LegalChars for c in key)):
            raise CookieError('Illegal key value: %s' % key)
        self.key = key
        self.value = val
        self.coded_value = coded_val
    
    def output(self, attrs=None, header='Set-Cookie:'):
        return '%s %s' % (header, self.OutputString(attrs))
    __str__ = output
    
    @as_native_str()
    def __repr__(self):
        if (PY2 and isinstance(self.value, unicode)):
            val = str(self.value)
        else:
            val = self.value
        return '<%s: %s=%s>' % (self.__class__.__name__, str(self.key), repr(val))
    
    def js_output(self, attrs=None):
        return '\n        <script type="text/javascript">\n        <!-- begin hiding\n        document.cookie = "%s";\n        // end hiding -->\n        </script>\n        ' % self.OutputString(attrs).replace('"', '\\"')
    
    def OutputString(self, attrs=None):
        result = []
        append = result.append
        append('%s=%s' % (self.key, self.coded_value))
        if attrs is None:
            attrs = self._reserved
        items = sorted(self.items())
        for (key, value) in items:
            if value == '':
                continue
            if key not in attrs:
                continue
            if (key == 'expires' and isinstance(value, int)):
                append('%s=%s' % (self._reserved[key], _getdate(value)))
            elif (key == 'max-age' and isinstance(value, int)):
                append('%s=%d' % (self._reserved[key], value))
            elif key == 'secure':
                append(str(self._reserved[key]))
            elif key == 'httponly':
                append(str(self._reserved[key]))
            else:
                append('%s=%s' % (self._reserved[key], value))
        return _semispacejoin(result)

_LegalCharsPatt = "[\\w\\d!#%&'~_`><@,:/\\$\\*\\+\\-\\.\\^\\|\\)\\(\\?\\}\\{\\=]"
_CookiePattern = re.compile("\n    (?x)                           # This is a verbose pattern\n    (?P<key>                       # Start of group 'key'\n    " + _LegalCharsPatt + '+?   # Any word of at least one letter\n    )                              # End of group \'key\'\n    (                              # Optional group: there may not be a value.\n    \\s*=\\s*                          # Equal Sign\n    (?P<val>                         # Start of group \'val\'\n    "(?:[^\\\\"]|\\\\.)*"                  # Any doublequoted string\n    |                                  # or\n    \\w{3},\\s[\\w\\d\\s-]{9,11}\\s[\\d:]{8}\\sGMT  # Special case for "expires" attr\n    |                                  # or\n    ' + _LegalCharsPatt + "*      # Any word or empty string\n    )                                # End of group 'val'\n    )?                             # End of optional value group\n    \\s*                            # Any number of spaces.\n    (\\s+|;|$)                      # Ending either at space, semicolon, or EOS.\n    ", re.ASCII)


class BaseCookie(dict):
    """A container class for a set of Morsels."""
    
    def value_decode(self, val):
        """real_value, coded_value = value_decode(STRING)
        Called prior to setting a cookie's value from the network
        representation.  The VALUE is the value read from HTTP
        header.
        Override this function to modify the behavior of cookies.
        """
        return (val, val)
    
    def value_encode(self, val):
        """real_value, coded_value = value_encode(VALUE)
        Called prior to setting a cookie's value from the dictionary
        representation.  The VALUE is the value being assigned.
        Override this function to modify the behavior of cookies.
        """
        strval = str(val)
        return (strval, strval)
    
    def __init__(self, input=None):
        if input:
            self.load(input)
    
    def __set(self, key, real_value, coded_value):
        """Private method for setting a cookie's value"""
        M = self.get(key, Morsel())
        M.set(key, real_value, coded_value)
        dict.__setitem__(self, key, M)
    
    def __setitem__(self, key, value):
        """Dictionary style assignment."""
        (rval, cval) = self.value_encode(value)
        self.__set(key, rval, cval)
    
    def output(self, attrs=None, header='Set-Cookie:', sep='\r\n'):
        """Return a string suitable for HTTP."""
        result = []
        items = sorted(self.items())
        for (key, value) in items:
            result.append(value.output(attrs, header))
        return sep.join(result)
    __str__ = output
    
    @as_native_str()
    def __repr__(self):
        l = []
        items = sorted(self.items())
        for (key, value) in items:
            if (PY2 and isinstance(value.value, unicode)):
                val = str(value.value)
            else:
                val = value.value
            l.append('%s=%s' % (str(key), repr(val)))
        return '<%s: %s>' % (self.__class__.__name__, _spacejoin(l))
    
    def js_output(self, attrs=None):
        """Return a string suitable for JavaScript."""
        result = []
        items = sorted(self.items())
        for (key, value) in items:
            result.append(value.js_output(attrs))
        return _nulljoin(result)
    
    def load(self, rawdata):
        """Load cookies from a string (presumably HTTP_COOKIE) or
        from a dictionary.  Loading cookies from a dictionary 'd'
        is equivalent to calling:
            map(Cookie.__setitem__, d.keys(), d.values())
        """
        if isinstance(rawdata, str):
            self.__parse_string(rawdata)
        else:
            for (key, value) in rawdata.items():
                self[key] = value
        return
    
    def __parse_string(self, mystr, patt=_CookiePattern):
        i = 0
        n = len(mystr)
        M = None
        while 0 <= i < n:
            match = patt.search(mystr, i)
            if not match:
                break
            (key, value) = (match.group('key'), match.group('val'))
            i = match.end(0)
            if key[0] == '$':
                if M:
                    M[key[1:]] = value
            elif key.lower() in Morsel._reserved:
                if M:
                    if value is None:
                        if key.lower() in Morsel._flags:
                            M[key] = True
                    else:
                        M[key] = _unquote(value)
            elif value is not None:
                (rval, cval) = self.value_decode(value)
                self.__set(key, rval, cval)
                M = self[key]



class SimpleCookie(BaseCookie):
    """
    SimpleCookie supports strings as cookie values.  When setting
    the value using the dictionary assignment notation, SimpleCookie
    calls the builtin str() to convert the value to a string.  Values
    received from HTTP are kept as strings.
    """
    
    def value_decode(self, val):
        return (_unquote(val), val)
    
    def value_encode(self, val):
        strval = str(val)
        return (strval, _quote(strval))


