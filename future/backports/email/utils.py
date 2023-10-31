"""Miscellaneous utilities."""

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import utils
from future.builtins import bytes, int, str
__all__ = ['collapse_rfc2231_value', 'decode_params', 'decode_rfc2231', 'encode_rfc2231', 'formataddr', 'formatdate', 'format_datetime', 'getaddresses', 'make_msgid', 'mktime_tz', 'parseaddr', 'parsedate', 'parsedate_tz', 'parsedate_to_datetime', 'unquote']
import os
import re
if utils.PY2:
    re.ASCII = 0
import time
import base64
import random
import socket
from future.backports import datetime
from future.backports.urllib.parse import quote as url_quote, unquote as url_unquote
import warnings
from io import StringIO
from future.backports.email._parseaddr import quote
from future.backports.email._parseaddr import AddressList as _AddressList
from future.backports.email._parseaddr import mktime_tz
from future.backports.email._parseaddr import parsedate, parsedate_tz, _parsedate_tz
from quopri import decodestring as _qdecode
from future.backports.email.encoders import _bencode, _qencode
from future.backports.email.charset import Charset
COMMASPACE = ', '
EMPTYSTRING = ''
UEMPTYSTRING = ''
CRLF = '\r\n'
TICK = "'"
specialsre = re.compile('[][\\\\()<>@,:;".]')
escapesre = re.compile('[\\\\"]')
_has_surrogates = re.compile('([^\ud800-\udbff]|\\A)[\udc00-\udfff]([^\udc00-\udfff]|\\Z)').search

def _sanitize(string):
    original_bytes = string.encode('ascii', 'surrogateescape')
    return original_bytes.decode('ascii', 'replace')

def formataddr(pair, charset='utf-8'):
    """The inverse of parseaddr(), this takes a 2-tuple of the form
    (realname, email_address) and returns the string value suitable
    for an RFC 2822 From, To or Cc header.

    If the first element of pair is false, then the second element is
    returned unmodified.

    Optional charset if given is the character set that is used to encode
    realname in case realname is not ASCII safe.  Can be an instance of str or
    a Charset-like object which has a header_encode method.  Default is
    'utf-8'.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.email.utils.formataddr', "formataddr(pair, charset='utf-8')", {'Charset': Charset, 'specialsre': specialsre, 'escapesre': escapesre, 'pair': pair, 'charset': charset}, 1)

def getaddresses(fieldvalues):
    """Return a list of (REALNAME, EMAIL) for each fieldvalue."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.email.utils.getaddresses', 'getaddresses(fieldvalues)', {'COMMASPACE': COMMASPACE, '_AddressList': _AddressList, 'fieldvalues': fieldvalues}, 1)
ecre = re.compile('\n  =\\?                   # literal =?\n  (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset\n  \\?                    # literal ?\n  (?P<encoding>[qb])    # either a "q" or a "b", case insensitive\n  \\?                    # literal ?\n  (?P<atom>.*?)         # non-greedy up to the next ?= is the atom\n  \\?=                   # literal ?=\n  ', re.VERBOSE | re.IGNORECASE)

def _format_timetuple_and_zone(timetuple, zone):
    return '%s, %02d %s %04d %02d:%02d:%02d %s' % (['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][timetuple[6]], timetuple[2], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][timetuple[1] - 1], timetuple[0], timetuple[3], timetuple[4], timetuple[5], zone)

def formatdate(timeval=None, localtime=False, usegmt=False):
    """Returns a date string as specified by RFC 2822, e.g.:

    Fri, 09 Nov 2001 01:08:47 -0000

    Optional timeval if given is a floating point time value as accepted by
    gmtime() and localtime(), otherwise the current time is used.

    Optional localtime is a flag that when True, interprets timeval, and
    returns a date relative to the local timezone instead of UTC, properly
    taking daylight savings time into account.

    Optional argument usegmt means that the timezone is written out as
    an ascii string, not numeric one (so "GMT" instead of "+0000"). This
    is needed for HTTP, and is only used when localtime==False.
    """
    if timeval is None:
        timeval = time.time()
    if localtime:
        now = time.localtime(timeval)
        if (time.daylight and now[-1]):
            offset = time.altzone
        else:
            offset = time.timezone
        (hours, minutes) = divmod(abs(offset), 3600)
        if offset > 0:
            sign = '-'
        else:
            sign = '+'
        zone = '%s%02d%02d' % (sign, hours, minutes // 60)
    else:
        now = time.gmtime(timeval)
        if usegmt:
            zone = 'GMT'
        else:
            zone = '-0000'
    return _format_timetuple_and_zone(now, zone)

def format_datetime(dt, usegmt=False):
    """Turn a datetime into a date string as specified in RFC 2822.

    If usegmt is True, dt must be an aware datetime with an offset of zero.  In
    this case 'GMT' will be rendered instead of the normal +0000 required by
    RFC2822.  This is to support HTTP headers involving date stamps.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.email.utils.format_datetime', 'format_datetime(dt, usegmt=False)', {'datetime': datetime, '_format_timetuple_and_zone': _format_timetuple_and_zone, 'dt': dt, 'usegmt': usegmt}, 1)

def make_msgid(idstring=None, domain=None):
    """Returns a string suitable for RFC 2822 compliant Message-ID, e.g:

    <20020201195627.33539.96671@nightshade.la.mastaler.com>

    Optional idstring if given is a string used to strengthen the
    uniqueness of the message id.  Optional domain if given provides the
    portion of the message id after the '@'.  It defaults to the locally
    defined hostname.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.email.utils.make_msgid', 'make_msgid(idstring=None, domain=None)', {'time': time, 'os': os, 'random': random, 'socket': socket, 'idstring': idstring, 'domain': domain}, 1)

def parsedate_to_datetime(data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.email.utils.parsedate_to_datetime', 'parsedate_to_datetime(data)', {'_parsedate_tz': _parsedate_tz, 'datetime': datetime, 'data': data}, 1)

def parseaddr(addr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.email.utils.parseaddr', 'parseaddr(addr)', {'_AddressList': _AddressList, 'addr': addr}, 2)

def unquote(str):
    """Remove quotes from a string."""
    if len(str) > 1:
        if (str.startswith('"') and str.endswith('"')):
            return str[1:-1].replace('\\\\', '\\').replace('\\"', '"')
        if (str.startswith('<') and str.endswith('>')):
            return str[1:-1]
    return str

def decode_rfc2231(s):
    """Decode string according to RFC 2231"""
    parts = s.split(TICK, 2)
    if len(parts) <= 2:
        return (None, None, s)
    return parts

def encode_rfc2231(s, charset=None, language=None):
    """Encode string according to RFC 2231.

    If neither charset nor language is given, then s is returned as-is.  If
    charset is given but not language, the string is encoded using the empty
    string for language.
    """
    s = url_quote(s, safe='', encoding=(charset or 'ascii'))
    if (charset is None and language is None):
        return s
    if language is None:
        language = ''
    return "%s'%s'%s" % (charset, language, s)
rfc2231_continuation = re.compile('^(?P<name>\\w+)\\*((?P<num>[0-9]+)\\*?)?$', re.ASCII)

def decode_params(params):
    """Decode parameters list according to RFC 2231.

    params is a sequence of 2-tuples containing (param name, string value).
    """
    params = params[:]
    new_params = []
    rfc2231_params = {}
    (name, value) = params.pop(0)
    new_params.append((name, value))
    while params:
        (name, value) = params.pop(0)
        if name.endswith('*'):
            encoded = True
        else:
            encoded = False
        value = unquote(value)
        mo = rfc2231_continuation.match(name)
        if mo:
            (name, num) = mo.group('name', 'num')
            if num is not None:
                num = int(num)
            rfc2231_params.setdefault(name, []).append((num, value, encoded))
        else:
            new_params.append((name, '"%s"' % quote(value)))
    if rfc2231_params:
        for (name, continuations) in rfc2231_params.items():
            value = []
            extended = False
            continuations.sort()
            for (num, s, encoded) in continuations:
                if encoded:
                    s = url_unquote(s, encoding='latin-1')
                    extended = True
                value.append(s)
            value = quote(EMPTYSTRING.join(value))
            if extended:
                (charset, language, value) = decode_rfc2231(value)
                new_params.append((name, (charset, language, '"%s"' % value)))
            else:
                new_params.append((name, '"%s"' % value))
    return new_params

def collapse_rfc2231_value(value, errors='replace', fallback_charset='us-ascii'):
    if (not isinstance(value, tuple) or len(value) != 3):
        return unquote(value)
    (charset, language, text) = value
    rawbytes = bytes(text, 'raw-unicode-escape')
    try:
        return str(rawbytes, charset, errors)
    except LookupError:
        return unquote(text)

def localtime(dt=None, isdst=-1):
    """Return local time as an aware datetime object.

    If called without arguments, return current time.  Otherwise *dt*
    argument should be a datetime instance, and it is converted to the
    local time zone according to the system time zone database.  If *dt* is
    naive (that is, dt.tzinfo is None), it is assumed to be in local time.
    In this case, a positive or zero value for *isdst* causes localtime to
    presume initially that summer time (for example, Daylight Saving Time)
    is or is not (respectively) in effect for the specified time.  A
    negative value for *isdst* causes the localtime() function to attempt
    to divine whether summer time is in effect for the specified time.

    """
    if dt is None:
        return datetime.datetime.now(datetime.timezone.utc).astimezone()
    if dt.tzinfo is not None:
        return dt.astimezone()
    tm = dt.timetuple()[:-1] + (isdst, )
    seconds = time.mktime(tm)
    localtm = time.localtime(seconds)
    try:
        delta = datetime.timedelta(seconds=localtm.tm_gmtoff)
        tz = datetime.timezone(delta, localtm.tm_zone)
    except AttributeError:
        delta = dt - datetime.datetime(*time.gmtime(seconds)[:6])
        dst = (time.daylight and localtm.tm_isdst > 0)
        gmtoff = -((time.altzone if dst else time.timezone))
        if delta == datetime.timedelta(seconds=gmtoff):
            tz = datetime.timezone(delta, time.tzname[dst])
        else:
            tz = datetime.timezone(delta)
    return dt.replace(tzinfo=tz)

