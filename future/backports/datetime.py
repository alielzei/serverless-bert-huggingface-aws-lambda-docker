"""Concrete date/time and related types.

See http://www.iana.org/time-zones/repository/tz-link.html for
time zone and DST data sources.
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future.builtins import str
from future.builtins import bytes
from future.builtins import map
from future.builtins import round
from future.builtins import int
from future.builtins import object
from future.utils import native_str, PY2
import time as _time
import math as _math

def _cmp(x, y):
    return (0 if x == y else (1 if x > y else -1))
MINYEAR = 1
MAXYEAR = 9999
_MAXORDINAL = 3652059
_DAYS_IN_MONTH = [None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_BEFORE_MONTH = [None]
dbm = 0
for dim in _DAYS_IN_MONTH[1:]:
    _DAYS_BEFORE_MONTH.append(dbm)
    dbm += dim
del dbm, dim

def _is_leap(year):
    """year -> 1 if leap year, else 0."""
    return (year % 4 == 0 and ((year % 100 != 0 or year % 400 == 0)))

def _days_before_year(year):
    """year -> number of days before January 1st of year."""
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400

def _days_in_month(year, month):
    """year, month -> number of days in that month in that year."""
    assert 1 <= month <= 12, month
    if (month == 2 and _is_leap(year)):
        return 29
    return _DAYS_IN_MONTH[month]

def _days_before_month(year, month):
    """year, month -> number of days in year preceding first day of month."""
    assert 1 <= month <= 12, 'month must be in 1..12'
    return _DAYS_BEFORE_MONTH[month] + ((month > 2 and _is_leap(year)))

def _ymd2ord(year, month, day):
    """year, month, day -> ordinal, considering 01-Jan-0001 as day 1."""
    assert 1 <= month <= 12, 'month must be in 1..12'
    dim = _days_in_month(year, month)
    assert 1 <= day <= dim, 'day must be in 1..%d' % dim
    return _days_before_year(year) + _days_before_month(year, month) + day
_DI400Y = _days_before_year(401)
_DI100Y = _days_before_year(101)
_DI4Y = _days_before_year(5)
assert _DI4Y == 4 * 365 + 1
assert _DI400Y == 4 * _DI100Y + 1
assert _DI100Y == 25 * _DI4Y - 1

def _ord2ymd(n):
    """ordinal -> (year, month, day), considering 01-Jan-0001 as day 1."""
    n -= 1
    (n400, n) = divmod(n, _DI400Y)
    year = n400 * 400 + 1
    (n100, n) = divmod(n, _DI100Y)
    (n4, n) = divmod(n, _DI4Y)
    (n1, n) = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if (n1 == 4 or n100 == 4):
        assert n == 0
        return (year - 1, 12, 31)
    leapyear = (n1 == 3 and ((n4 != 24 or n100 == 3)))
    assert leapyear == _is_leap(year)
    month = n + 50 >> 5
    preceding = _DAYS_BEFORE_MONTH[month] + ((month > 2 and leapyear))
    if preceding > n:
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + ((month == 2 and leapyear))
    n -= preceding
    assert 0 <= n < _days_in_month(year, month)
    return (year, month, n + 1)
_MONTHNAMES = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_DAYNAMES = [None, 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def _build_struct_time(y, m, d, hh, mm, ss, dstflag):
    wday = (_ymd2ord(y, m, d) + 6) % 7
    dnum = _days_before_month(y, m) + d
    return _time.struct_time((y, m, d, hh, mm, ss, wday, dnum, dstflag))

def _format_time(hh, mm, ss, us):
    result = '%02d:%02d:%02d' % (hh, mm, ss)
    if us:
        result += '.%06d' % us
    return result

def _wrap_strftime(object, format, timetuple):
    freplace = None
    zreplace = None
    Zreplace = None
    newformat = []
    push = newformat.append
    (i, n) = (0, len(format))
    while i < n:
        ch = format[i]
        i += 1
        if ch == '%':
            if i < n:
                ch = format[i]
                i += 1
                if ch == 'f':
                    if freplace is None:
                        freplace = '%06d' % getattr(object, 'microsecond', 0)
                    newformat.append(freplace)
                elif ch == 'z':
                    if zreplace is None:
                        zreplace = ''
                        if hasattr(object, 'utcoffset'):
                            offset = object.utcoffset()
                            if offset is not None:
                                sign = '+'
                                if offset.days < 0:
                                    offset = -offset
                                    sign = '-'
                                (h, m) = divmod(offset, timedelta(hours=1))
                                assert not m % timedelta(minutes=1), 'whole minute'
                                m //= timedelta(minutes=1)
                                zreplace = '%c%02d%02d' % (sign, h, m)
                    assert '%' not in zreplace
                    newformat.append(zreplace)
                elif ch == 'Z':
                    if Zreplace is None:
                        Zreplace = ''
                        if hasattr(object, 'tzname'):
                            s = object.tzname()
                            if s is not None:
                                Zreplace = s.replace('%', '%%')
                    newformat.append(Zreplace)
                else:
                    push('%')
                    push(ch)
            else:
                push('%')
        else:
            push(ch)
    newformat = ''.join(newformat)
    return _time.strftime(newformat, timetuple)

def _call_tzinfo_method(tzinfo, methname, tzinfoarg):
    if tzinfo is None:
        return None
    return getattr(tzinfo, methname)(tzinfoarg)

def _check_tzname(name):
    if (name is not None and not isinstance(name, str)):
        raise TypeError("tzinfo.tzname() must return None or string, not '%s'" % type(name))

def _check_utc_offset(name, offset):
    assert name in ('utcoffset', 'dst')
    if offset is None:
        return
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.%s() must return None or timedelta, not '%s'" % (name, type(offset)))
    if (offset % timedelta(minutes=1) or offset.microseconds):
        raise ValueError('tzinfo.%s() must return a whole number of minutes, got %s' % (name, offset))
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError('%s()=%s, must be must be strictly between -timedelta(hours=24) and timedelta(hours=24)' % (name, offset))

def _check_date_fields(year, month, day):
    if not isinstance(year, int):
        raise TypeError('int expected')
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError('year must be in %d..%d' % (MINYEAR, MAXYEAR), year)
    if not 1 <= month <= 12:
        raise ValueError('month must be in 1..12', month)
    dim = _days_in_month(year, month)
    if not 1 <= day <= dim:
        raise ValueError('day must be in 1..%d' % dim, day)

def _check_time_fields(hour, minute, second, microsecond):
    if not isinstance(hour, int):
        raise TypeError('int expected')
    if not 0 <= hour <= 23:
        raise ValueError('hour must be in 0..23', hour)
    if not 0 <= minute <= 59:
        raise ValueError('minute must be in 0..59', minute)
    if not 0 <= second <= 59:
        raise ValueError('second must be in 0..59', second)
    if not 0 <= microsecond <= 999999:
        raise ValueError('microsecond must be in 0..999999', microsecond)

def _check_tzinfo_arg(tz):
    if (tz is not None and not isinstance(tz, tzinfo)):
        raise TypeError('tzinfo argument must be None or of a tzinfo subclass')

def _cmperror(x, y):
    raise TypeError("can't compare '%s' to '%s'" % (type(x).__name__, type(y).__name__))


class timedelta(object):
    """Represent the difference between two datetime objects.

    Supported operators:

    - add, subtract timedelta
    - unary plus, minus, abs
    - compare to timedelta
    - multiply, divide by int

    In addition, datetime supports subtraction of two datetime objects
    returning a timedelta, and addition or subtraction of a datetime
    and a timedelta giving a datetime.

    Representation: (days, seconds, microseconds).  Why?  Because I
    felt like it.
    """
    __slots__ = ('_days', '_seconds', '_microseconds')
    
    def __new__(cls, days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
        d = s = us = 0
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000
        if isinstance(days, float):
            (dayfrac, days) = _math.modf(days)
            (daysecondsfrac, daysecondswhole) = _math.modf(dayfrac * (24.0 * 3600.0))
            assert daysecondswhole == int(daysecondswhole)
            s = int(daysecondswhole)
            assert days == int(days)
            d = int(days)
        else:
            daysecondsfrac = 0.0
            d = days
        assert isinstance(daysecondsfrac, float)
        assert abs(daysecondsfrac) <= 1.0
        assert isinstance(d, int)
        assert abs(s) <= 24 * 3600
        if isinstance(seconds, float):
            (secondsfrac, seconds) = _math.modf(seconds)
            assert seconds == int(seconds)
            seconds = int(seconds)
            secondsfrac += daysecondsfrac
            assert abs(secondsfrac) <= 2.0
        else:
            secondsfrac = daysecondsfrac
        assert isinstance(secondsfrac, float)
        assert abs(secondsfrac) <= 2.0
        assert isinstance(seconds, int)
        (days, seconds) = divmod(seconds, 24 * 3600)
        d += days
        s += int(seconds)
        assert isinstance(s, int)
        assert abs(s) <= 2 * 24 * 3600
        usdouble = secondsfrac * 1000000.0
        assert abs(usdouble) < 2100000.0
        if isinstance(microseconds, float):
            microseconds += usdouble
            microseconds = round(microseconds, 0)
            (seconds, microseconds) = divmod(microseconds, 1000000.0)
            assert microseconds == int(microseconds)
            assert seconds == int(seconds)
            (days, seconds) = divmod(seconds, 24.0 * 3600.0)
            assert days == int(days)
            assert seconds == int(seconds)
            d += int(days)
            s += int(seconds)
            assert isinstance(s, int)
            assert abs(s) <= 3 * 24 * 3600
        else:
            (seconds, microseconds) = divmod(microseconds, 1000000)
            (days, seconds) = divmod(seconds, 24 * 3600)
            d += days
            s += int(seconds)
            assert isinstance(s, int)
            assert abs(s) <= 3 * 24 * 3600
            microseconds = float(microseconds)
            microseconds += usdouble
            microseconds = round(microseconds, 0)
        assert abs(s) <= 3 * 24 * 3600
        assert abs(microseconds) < 3100000.0
        assert isinstance(microseconds, float)
        assert int(microseconds) == microseconds
        us = int(microseconds)
        (seconds, us) = divmod(us, 1000000)
        s += seconds
        assert isinstance(s, int)
        (days, s) = divmod(s, 24 * 3600)
        d += days
        assert isinstance(d, int)
        assert (isinstance(s, int) and 0 <= s < 24 * 3600)
        assert (isinstance(us, int) and 0 <= us < 1000000)
        self = object.__new__(cls)
        self._days = d
        self._seconds = s
        self._microseconds = us
        if abs(d) > 999999999:
            raise OverflowError('timedelta # of days is too large: %d' % d)
        return self
    
    def __repr__(self):
        if self._microseconds:
            return '%s(%d, %d, %d)' % ('datetime.' + self.__class__.__name__, self._days, self._seconds, self._microseconds)
        if self._seconds:
            return '%s(%d, %d)' % ('datetime.' + self.__class__.__name__, self._days, self._seconds)
        return '%s(%d)' % ('datetime.' + self.__class__.__name__, self._days)
    
    def __str__(self):
        (mm, ss) = divmod(self._seconds, 60)
        (hh, mm) = divmod(mm, 60)
        s = '%d:%02d:%02d' % (hh, mm, ss)
        if self._days:
            
            def plural(n):
                return (n, ((abs(n) != 1 and 's') or ''))
            s = '%d day%s, ' % plural(self._days) + s
        if self._microseconds:
            s = s + '.%06d' % self._microseconds
        return s
    
    def total_seconds(self):
        """Total seconds in the duration."""
        return ((self.days * 86400 + self.seconds) * 10**6 + self.microseconds) / 10**6
    
    @property
    def days(self):
        """days"""
        return self._days
    
    @property
    def seconds(self):
        """seconds"""
        return self._seconds
    
    @property
    def microseconds(self):
        """microseconds"""
        return self._microseconds
    
    def __add__(self, other):
        if isinstance(other, timedelta):
            return timedelta(self._days + other._days, self._seconds + other._seconds, self._microseconds + other._microseconds)
        return NotImplemented
    __radd__ = __add__
    
    def __sub__(self, other):
        if isinstance(other, timedelta):
            return timedelta(self._days - other._days, self._seconds - other._seconds, self._microseconds - other._microseconds)
        return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, timedelta):
            return -self + other
        return NotImplemented
    
    def __neg__(self):
        return timedelta(-self._days, -self._seconds, -self._microseconds)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        if self._days < 0:
            return -self
        else:
            return self
    
    def __mul__(self, other):
        if isinstance(other, int):
            return timedelta(self._days * other, self._seconds * other, self._microseconds * other)
        if isinstance(other, float):
            (a, b) = other.as_integer_ratio()
            return self * a / b
        return NotImplemented
    __rmul__ = __mul__
    
    def _to_microseconds(self):
        return (self._days * (24 * 3600) + self._seconds) * 1000000 + self._microseconds
    
    def __floordiv__(self, other):
        if not isinstance(other, (int, timedelta)):
            return NotImplemented
        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec // other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, usec // other)
    
    def __truediv__(self, other):
        if not isinstance(other, (int, float, timedelta)):
            return NotImplemented
        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec / other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, usec / other)
        if isinstance(other, float):
            (a, b) = other.as_integer_ratio()
            return timedelta(0, 0, b * usec / a)
    
    def __mod__(self, other):
        if isinstance(other, timedelta):
            r = self._to_microseconds() % other._to_microseconds()
            return timedelta(0, 0, r)
        return NotImplemented
    
    def __divmod__(self, other):
        if isinstance(other, timedelta):
            (q, r) = divmod(self._to_microseconds(), other._to_microseconds())
            return (q, timedelta(0, 0, r))
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) == 0
        else:
            return False
    
    def __ne__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) != 0
        else:
            return True
    
    def __le__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)
    
    def __lt__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)
    
    def __ge__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)
    
    def __gt__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)
    
    def _cmp(self, other):
        assert isinstance(other, timedelta)
        return _cmp(self._getstate(), other._getstate())
    
    def __hash__(self):
        return hash(self._getstate())
    
    def __bool__(self):
        return (self._days != 0 or self._seconds != 0 or self._microseconds != 0)
    
    def _getstate(self):
        return (self._days, self._seconds, self._microseconds)
    
    def __reduce__(self):
        return (self.__class__, self._getstate())

timedelta.min = timedelta(-999999999)
timedelta.max = timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999)
timedelta.resolution = timedelta(microseconds=1)


class date(object):
    """Concrete date type.

    Constructors:

    __new__()
    fromtimestamp()
    today()
    fromordinal()

    Operators:

    __repr__, __str__
    __cmp__, __hash__
    __add__, __radd__, __sub__ (add/radd only with timedelta arg)

    Methods:

    timetuple()
    toordinal()
    weekday()
    isoweekday(), isocalendar(), isoformat()
    ctime()
    strftime()

    Properties (readonly):
    year, month, day
    """
    __slots__ = ('_year', '_month', '_day')
    
    def __new__(cls, year, month=None, day=None):
        """Constructor.

        Arguments:

        year, month, day (required, base 1)
        """
        if (isinstance(year, bytes) and len(year) == 4 and 1 <= year[2] <= 12 and month is None):
            self = object.__new__(cls)
            self.__setstate(year)
            return self
        _check_date_fields(year, month, day)
        self = object.__new__(cls)
        self._year = year
        self._month = month
        self._day = day
        return self
    
    @classmethod
    def fromtimestamp(cls, t):
        """Construct a date from a POSIX timestamp (like time.time())."""
        (y, m, d, hh, mm, ss, weekday, jday, dst) = _time.localtime(t)
        return cls(y, m, d)
    
    @classmethod
    def today(cls):
        """Construct a date from time.time()."""
        t = _time.time()
        return cls.fromtimestamp(t)
    
    @classmethod
    def fromordinal(cls, n):
        """Contruct a date from a proleptic Gregorian ordinal.

        January 1 of year 1 is day 1.  Only the year, month and day are
        non-zero in the result.
        """
        (y, m, d) = _ord2ymd(n)
        return cls(y, m, d)
    
    def __repr__(self):
        """Convert to formal string, for repr().

        >>> dt = datetime(2010, 1, 1)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0)'

        >>> dt = datetime(2010, 1, 1, tzinfo=timezone.utc)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)'
        """
        return '%s(%d, %d, %d)' % ('datetime.' + self.__class__.__name__, self._year, self._month, self._day)
    
    def ctime(self):
        """Return ctime() style string."""
        weekday = (self.toordinal() % 7 or 7)
        return '%s %s %2d 00:00:00 %04d' % (_DAYNAMES[weekday], _MONTHNAMES[self._month], self._day, self._year)
    
    def strftime(self, fmt):
        """Format using strftime()."""
        return _wrap_strftime(self, fmt, self.timetuple())
    
    def __format__(self, fmt):
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)
    
    def isoformat(self):
        """Return the date formatted according to ISO.

        This is 'YYYY-MM-DD'.

        References:
        - http://www.w3.org/TR/NOTE-datetime
        - http://www.cl.cam.ac.uk/~mgk25/iso-time.html
        """
        return '%04d-%02d-%02d' % (self._year, self._month, self._day)
    __str__ = isoformat
    
    @property
    def year(self):
        """year (1-9999)"""
        return self._year
    
    @property
    def month(self):
        """month (1-12)"""
        return self._month
    
    @property
    def day(self):
        """day (1-31)"""
        return self._day
    
    def timetuple(self):
        """Return local time tuple compatible with time.localtime()."""
        return _build_struct_time(self._year, self._month, self._day, 0, 0, 0, -1)
    
    def toordinal(self):
        """Return proleptic Gregorian ordinal for the year, month and day.

        January 1 of year 1 is day 1.  Only the year, month and day values
        contribute to the result.
        """
        return _ymd2ord(self._year, self._month, self._day)
    
    def replace(self, year=None, month=None, day=None):
        """Return a new date with new values for the specified fields."""
        if year is None:
            year = self._year
        if month is None:
            month = self._month
        if day is None:
            day = self._day
        _check_date_fields(year, month, day)
        return date(year, month, day)
    
    def __eq__(self, other):
        if isinstance(other, date):
            return self._cmp(other) == 0
        return NotImplemented
    
    def __ne__(self, other):
        if isinstance(other, date):
            return self._cmp(other) != 0
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, date):
            return self._cmp(other) <= 0
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, date):
            return self._cmp(other) < 0
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, date):
            return self._cmp(other) >= 0
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, date):
            return self._cmp(other) > 0
        return NotImplemented
    
    def _cmp(self, other):
        assert isinstance(other, date)
        (y, m, d) = (self._year, self._month, self._day)
        (y2, m2, d2) = (other._year, other._month, other._day)
        return _cmp((y, m, d), (y2, m2, d2))
    
    def __hash__(self):
        """Hash."""
        return hash(self._getstate())
    
    def __add__(self, other):
        """Add a date to a timedelta."""
        if isinstance(other, timedelta):
            o = self.toordinal() + other.days
            if 0 < o <= _MAXORDINAL:
                return date.fromordinal(o)
            raise OverflowError('result out of range')
        return NotImplemented
    __radd__ = __add__
    
    def __sub__(self, other):
        """Subtract two dates, or a date and a timedelta."""
        if isinstance(other, timedelta):
            return self + timedelta(-other.days)
        if isinstance(other, date):
            days1 = self.toordinal()
            days2 = other.toordinal()
            return timedelta(days1 - days2)
        return NotImplemented
    
    def weekday(self):
        """Return day of the week, where Monday == 0 ... Sunday == 6."""
        return (self.toordinal() + 6) % 7
    
    def isoweekday(self):
        """Return day of the week, where Monday == 1 ... Sunday == 7."""
        return (self.toordinal() % 7 or 7)
    
    def isocalendar(self):
        """Return a 3-tuple containing ISO year, week number, and weekday.

        The first ISO week of the year is the (Mon-Sun) week
        containing the year's first Thursday; everything else derives
        from that.

        The first week is 1; Monday is 1 ... Sunday is 7.

        ISO calendar algorithm taken from
        http://www.phys.uu.nl/~vgent/calendar/isocalendar.htm
        """
        year = self._year
        week1monday = _isoweek1monday(year)
        today = _ymd2ord(self._year, self._month, self._day)
        (week, day) = divmod(today - week1monday, 7)
        if week < 0:
            year -= 1
            week1monday = _isoweek1monday(year)
            (week, day) = divmod(today - week1monday, 7)
        elif week >= 52:
            if today >= _isoweek1monday(year + 1):
                year += 1
                week = 0
        return (year, week + 1, day + 1)
    
    def _getstate(self):
        (yhi, ylo) = divmod(self._year, 256)
        return (bytes([yhi, ylo, self._month, self._day]), )
    
    def __setstate(self, string):
        if (len(string) != 4 or not 1 <= string[2] <= 12):
            raise TypeError('not enough arguments')
        (yhi, ylo, self._month, self._day) = string
        self._year = yhi * 256 + ylo
    
    def __reduce__(self):
        return (self.__class__, self._getstate())

_date_class = date
date.min = date(1, 1, 1)
date.max = date(9999, 12, 31)
date.resolution = timedelta(days=1)


class tzinfo(object):
    """Abstract base class for time zone info classes.

    Subclasses must override the name(), utcoffset() and dst() methods.
    """
    __slots__ = ()
    
    def tzname(self, dt):
        """datetime -> string name of time zone."""
        raise NotImplementedError('tzinfo subclass must override tzname()')
    
    def utcoffset(self, dt):
        """datetime -> minutes east of UTC (negative for west of UTC)"""
        raise NotImplementedError('tzinfo subclass must override utcoffset()')
    
    def dst(self, dt):
        """datetime -> DST offset in minutes east of UTC.

        Return 0 if DST not in effect.  utcoffset() must include the DST
        offset.
        """
        raise NotImplementedError('tzinfo subclass must override dst()')
    
    def fromutc(self, dt):
        """datetime in UTC -> datetime in local time."""
        if not isinstance(dt, datetime):
            raise TypeError('fromutc() requires a datetime argument')
        if dt.tzinfo is not self:
            raise ValueError('dt.tzinfo is not self')
        dtoff = dt.utcoffset()
        if dtoff is None:
            raise ValueError('fromutc() requires a non-None utcoffset() result')
        dtdst = dt.dst()
        if dtdst is None:
            raise ValueError('fromutc() requires a non-None dst() result')
        delta = dtoff - dtdst
        if delta:
            dt += delta
            dtdst = dt.dst()
            if dtdst is None:
                raise ValueError('fromutc(): dt.dst gave inconsistent results; cannot convert')
        return dt + dtdst
    
    def __reduce__(self):
        getinitargs = getattr(self, '__getinitargs__', None)
        if getinitargs:
            args = getinitargs()
        else:
            args = ()
        getstate = getattr(self, '__getstate__', None)
        if getstate:
            state = getstate()
        else:
            state = (getattr(self, '__dict__', None) or None)
        if state is None:
            return (self.__class__, args)
        else:
            return (self.__class__, args, state)

_tzinfo_class = tzinfo


class time(object):
    """Time with time zone.

    Constructors:

    __new__()

    Operators:

    __repr__, __str__
    __cmp__, __hash__

    Methods:

    strftime()
    isoformat()
    utcoffset()
    tzname()
    dst()

    Properties (readonly):
    hour, minute, second, microsecond, tzinfo
    """
    
    def __new__(cls, hour=0, minute=0, second=0, microsecond=0, tzinfo=None):
        """Constructor.

        Arguments:

        hour, minute (required)
        second, microsecond (default to zero)
        tzinfo (default to None)
        """
        self = object.__new__(cls)
        if (isinstance(hour, bytes) and len(hour) == 6):
            self.__setstate(hour, (minute or None))
            return self
        _check_tzinfo_arg(tzinfo)
        _check_time_fields(hour, minute, second, microsecond)
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond
        self._tzinfo = tzinfo
        return self
    
    @property
    def hour(self):
        """hour (0-23)"""
        return self._hour
    
    @property
    def minute(self):
        """minute (0-59)"""
        return self._minute
    
    @property
    def second(self):
        """second (0-59)"""
        return self._second
    
    @property
    def microsecond(self):
        """microsecond (0-999999)"""
        return self._microsecond
    
    @property
    def tzinfo(self):
        """timezone info object"""
        return self._tzinfo
    
    def __eq__(self, other):
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=True) == 0
        else:
            return False
    
    def __ne__(self, other):
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=True) != 0
        else:
            return True
    
    def __le__(self, other):
        if isinstance(other, time):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)
    
    def __lt__(self, other):
        if isinstance(other, time):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)
    
    def __ge__(self, other):
        if isinstance(other, time):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)
    
    def __gt__(self, other):
        if isinstance(other, time):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)
    
    def _cmp(self, other, allow_mixed=False):
        assert isinstance(other, time)
        mytz = self._tzinfo
        ottz = other._tzinfo
        myoff = otoff = None
        if mytz is ottz:
            base_compare = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare = myoff == otoff
        if base_compare:
            return _cmp((self._hour, self._minute, self._second, self._microsecond), (other._hour, other._minute, other._second, other._microsecond))
        if (myoff is None or otoff is None):
            if allow_mixed:
                return 2
            else:
                raise TypeError('cannot compare naive and aware times')
        myhhmm = self._hour * 60 + self._minute - myoff // timedelta(minutes=1)
        othhmm = other._hour * 60 + other._minute - otoff // timedelta(minutes=1)
        return _cmp((myhhmm, self._second, self._microsecond), (othhmm, other._second, other._microsecond))
    
    def __hash__(self):
        """Hash."""
        tzoff = self.utcoffset()
        if not tzoff:
            return hash(self._getstate()[0])
        (h, m) = divmod(timedelta(hours=self.hour, minutes=self.minute) - tzoff, timedelta(hours=1))
        assert not m % timedelta(minutes=1), 'whole minute'
        m //= timedelta(minutes=1)
        if 0 <= h < 24:
            return hash(time(h, m, self.second, self.microsecond))
        return hash((h, m, self.second, self.microsecond))
    
    def _tzstr(self, sep=':'):
        """Return formatted timezone offset (+xx:xx) or None."""
        off = self.utcoffset()
        if off is not None:
            if off.days < 0:
                sign = '-'
                off = -off
            else:
                sign = '+'
            (hh, mm) = divmod(off, timedelta(hours=1))
            assert not mm % timedelta(minutes=1), 'whole minute'
            mm //= timedelta(minutes=1)
            assert 0 <= hh < 24
            off = '%s%02d%s%02d' % (sign, hh, sep, mm)
        return off
    
    def __repr__(self):
        """Convert to formal string, for repr()."""
        if self._microsecond != 0:
            s = ', %d, %d' % (self._second, self._microsecond)
        elif self._second != 0:
            s = ', %d' % self._second
        else:
            s = ''
        s = '%s(%d, %d%s)' % ('datetime.' + self.__class__.__name__, self._hour, self._minute, s)
        if self._tzinfo is not None:
            assert s[-1:] == ')'
            s = s[:-1] + ', tzinfo=%r' % self._tzinfo + ')'
        return s
    
    def isoformat(self):
        """Return the time formatted according to ISO.

        This is 'HH:MM:SS.mmmmmm+zz:zz', or 'HH:MM:SS+zz:zz' if
        self.microsecond == 0.
        """
        s = _format_time(self._hour, self._minute, self._second, self._microsecond)
        tz = self._tzstr()
        if tz:
            s += tz
        return s
    __str__ = isoformat
    
    def strftime(self, fmt):
        """Format using strftime().  The date part of the timestamp passed
        to underlying strftime should not be used.
        """
        timetuple = (1900, 1, 1, self._hour, self._minute, self._second, 0, 1, -1)
        return _wrap_strftime(self, fmt, timetuple)
    
    def __format__(self, fmt):
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)
    
    def utcoffset(self):
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC)."""
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.utcoffset(None)
        _check_utc_offset('utcoffset', offset)
        return offset
    
    def tzname(self):
        """Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        if self._tzinfo is None:
            return None
        name = self._tzinfo.tzname(None)
        _check_tzname(name)
        return name
    
    def dst(self):
        """Return 0 if DST is not in effect, or the DST offset (in minutes
        eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.dst(None)
        _check_utc_offset('dst', offset)
        return offset
    
    def replace(self, hour=None, minute=None, second=None, microsecond=None, tzinfo=True):
        """Return a new time with new values for the specified fields."""
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        _check_time_fields(hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        return time(hour, minute, second, microsecond, tzinfo)
    
    def __bool__(self):
        if (self.second or self.microsecond):
            return True
        offset = (self.utcoffset() or timedelta(0))
        return timedelta(hours=self.hour, minutes=self.minute) != offset
    
    def _getstate(self):
        (us2, us3) = divmod(self._microsecond, 256)
        (us1, us2) = divmod(us2, 256)
        basestate = bytes([self._hour, self._minute, self._second, us1, us2, us3])
        if self._tzinfo is None:
            return (basestate, )
        else:
            return (basestate, self._tzinfo)
    
    def __setstate(self, string, tzinfo):
        if (len(string) != 6 or string[0] >= 24):
            raise TypeError('an integer is required')
        (self._hour, self._minute, self._second, us1, us2, us3) = string
        self._microsecond = (us1 << 8 | us2) << 8 | us3
        if (tzinfo is None or isinstance(tzinfo, _tzinfo_class)):
            self._tzinfo = tzinfo
        else:
            raise TypeError('bad tzinfo state arg %r' % tzinfo)
    
    def __reduce__(self):
        return (time, self._getstate())

_time_class = time
time.min = time(0, 0, 0)
time.max = time(23, 59, 59, 999999)
time.resolution = timedelta(microseconds=1)


class datetime(date):
    """datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

    The year, month and day arguments are required. tzinfo may be None, or an
    instance of a tzinfo subclass. The remaining arguments may be ints.
    """
    __slots__ = date.__slots__ + ('_hour', '_minute', '_second', '_microsecond', '_tzinfo')
    
    def __new__(cls, year, month=None, day=None, hour=0, minute=0, second=0, microsecond=0, tzinfo=None):
        if (isinstance(year, bytes) and len(year) == 10):
            self = date.__new__(cls, year[:4])
            self.__setstate(year, month)
            return self
        _check_tzinfo_arg(tzinfo)
        _check_time_fields(hour, minute, second, microsecond)
        self = date.__new__(cls, year, month, day)
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond
        self._tzinfo = tzinfo
        return self
    
    @property
    def hour(self):
        """hour (0-23)"""
        return self._hour
    
    @property
    def minute(self):
        """minute (0-59)"""
        return self._minute
    
    @property
    def second(self):
        """second (0-59)"""
        return self._second
    
    @property
    def microsecond(self):
        """microsecond (0-999999)"""
        return self._microsecond
    
    @property
    def tzinfo(self):
        """timezone info object"""
        return self._tzinfo
    
    @classmethod
    def fromtimestamp(cls, t, tz=None):
        """Construct a datetime from a POSIX timestamp (like time.time()).

        A timezone info object may be passed in as well.
        """
        _check_tzinfo_arg(tz)
        converter = (_time.localtime if tz is None else _time.gmtime)
        (t, frac) = divmod(t, 1.0)
        us = int(frac * 1000000.0)
        if us == 1000000:
            t += 1
            us = 0
        (y, m, d, hh, mm, ss, weekday, jday, dst) = converter(t)
        ss = min(ss, 59)
        result = cls(y, m, d, hh, mm, ss, us, tz)
        if tz is not None:
            result = tz.fromutc(result)
        return result
    
    @classmethod
    def utcfromtimestamp(cls, t):
        """Construct a UTC datetime from a POSIX timestamp (like time.time())."""
        (t, frac) = divmod(t, 1.0)
        us = int(frac * 1000000.0)
        if us == 1000000:
            t += 1
            us = 0
        (y, m, d, hh, mm, ss, weekday, jday, dst) = _time.gmtime(t)
        ss = min(ss, 59)
        return cls(y, m, d, hh, mm, ss, us)
    
    @classmethod
    def now(cls, tz=None):
        """Construct a datetime from time.time() and optional time zone info."""
        t = _time.time()
        return cls.fromtimestamp(t, tz)
    
    @classmethod
    def utcnow(cls):
        """Construct a UTC datetime from time.time()."""
        t = _time.time()
        return cls.utcfromtimestamp(t)
    
    @classmethod
    def combine(cls, date, time):
        """Construct a datetime from a given date and a given time."""
        if not isinstance(date, _date_class):
            raise TypeError('date argument must be a date instance')
        if not isinstance(time, _time_class):
            raise TypeError('time argument must be a time instance')
        return cls(date.year, date.month, date.day, time.hour, time.minute, time.second, time.microsecond, time.tzinfo)
    
    def timetuple(self):
        """Return local time tuple compatible with time.localtime()."""
        dst = self.dst()
        if dst is None:
            dst = -1
        elif dst:
            dst = 1
        else:
            dst = 0
        return _build_struct_time(self.year, self.month, self.day, self.hour, self.minute, self.second, dst)
    
    def timestamp(self):
        """Return POSIX timestamp as float"""
        if self._tzinfo is None:
            return _time.mktime((self.year, self.month, self.day, self.hour, self.minute, self.second, -1, -1, -1)) + self.microsecond / 1000000.0
        else:
            return (self - _EPOCH).total_seconds()
    
    def utctimetuple(self):
        """Return UTC time tuple compatible with time.gmtime()."""
        offset = self.utcoffset()
        if offset:
            self -= offset
        (y, m, d) = (self.year, self.month, self.day)
        (hh, mm, ss) = (self.hour, self.minute, self.second)
        return _build_struct_time(y, m, d, hh, mm, ss, 0)
    
    def date(self):
        """Return the date part."""
        return date(self._year, self._month, self._day)
    
    def time(self):
        """Return the time part, with tzinfo None."""
        return time(self.hour, self.minute, self.second, self.microsecond)
    
    def timetz(self):
        """Return the time part, with same tzinfo."""
        return time(self.hour, self.minute, self.second, self.microsecond, self._tzinfo)
    
    def replace(self, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, tzinfo=True):
        """Return a new datetime with new values for the specified fields."""
        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if day is None:
            day = self.day
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        _check_date_fields(year, month, day)
        _check_time_fields(hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        return datetime(year, month, day, hour, minute, second, microsecond, tzinfo)
    
    def astimezone(self, tz=None):
        if tz is None:
            if self.tzinfo is None:
                raise ValueError('astimezone() requires an aware datetime')
            ts = (self - _EPOCH) // timedelta(seconds=1)
            localtm = _time.localtime(ts)
            local = datetime(*localtm[:6])
            try:
                gmtoff = localtm.tm_gmtoff
                zone = localtm.tm_zone
            except AttributeError:
                delta = local - datetime(*_time.gmtime(ts)[:6])
                dst = (_time.daylight and localtm.tm_isdst > 0)
                gmtoff = -((_time.altzone if dst else _time.timezone))
                if delta == timedelta(seconds=gmtoff):
                    tz = timezone(delta, _time.tzname[dst])
                else:
                    tz = timezone(delta)
            else:
                tz = timezone(timedelta(seconds=gmtoff), zone)
        elif not isinstance(tz, tzinfo):
            raise TypeError('tz argument must be an instance of tzinfo')
        mytz = self.tzinfo
        if mytz is None:
            raise ValueError('astimezone() requires an aware datetime')
        if tz is mytz:
            return self
        myoffset = self.utcoffset()
        if myoffset is None:
            raise ValueError('astimezone() requires an aware datetime')
        utc = (self - myoffset).replace(tzinfo=tz)
        return tz.fromutc(utc)
    
    def ctime(self):
        """Return ctime() style string."""
        weekday = (self.toordinal() % 7 or 7)
        return '%s %s %2d %02d:%02d:%02d %04d' % (_DAYNAMES[weekday], _MONTHNAMES[self._month], self._day, self._hour, self._minute, self._second, self._year)
    
    def isoformat(self, sep='T'):
        """Return the time formatted according to ISO.

        This is 'YYYY-MM-DD HH:MM:SS.mmmmmm', or 'YYYY-MM-DD HH:MM:SS' if
        self.microsecond == 0.

        If self.tzinfo is not None, the UTC offset is also attached, giving
        'YYYY-MM-DD HH:MM:SS.mmmmmm+HH:MM' or 'YYYY-MM-DD HH:MM:SS+HH:MM'.

        Optional argument sep specifies the separator between date and
        time, default 'T'.
        """
        s = '%04d-%02d-%02d%c' % (self._year, self._month, self._day, sep) + _format_time(self._hour, self._minute, self._second, self._microsecond)
        off = self.utcoffset()
        if off is not None:
            if off.days < 0:
                sign = '-'
                off = -off
            else:
                sign = '+'
            (hh, mm) = divmod(off, timedelta(hours=1))
            assert not mm % timedelta(minutes=1), 'whole minute'
            mm //= timedelta(minutes=1)
            s += '%s%02d:%02d' % (sign, hh, mm)
        return s
    
    def __repr__(self):
        """Convert to formal string, for repr()."""
        L = [self._year, self._month, self._day, self._hour, self._minute, self._second, self._microsecond]
        if L[-1] == 0:
            del L[-1]
        if L[-1] == 0:
            del L[-1]
        s = ', '.join(map(str, L))
        s = '%s(%s)' % ('datetime.' + self.__class__.__name__, s)
        if self._tzinfo is not None:
            assert s[-1:] == ')'
            s = s[:-1] + ', tzinfo=%r' % self._tzinfo + ')'
        return s
    
    def __str__(self):
        """Convert to string, for str()."""
        return self.isoformat(sep=' ')
    
    @classmethod
    def strptime(cls, date_string, format):
        """string, format -> new datetime parsed from a string (like time.strptime())."""
        import _strptime
        return _strptime._strptime_datetime(cls, date_string, format)
    
    def utcoffset(self):
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC)."""
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.utcoffset(self)
        _check_utc_offset('utcoffset', offset)
        return offset
    
    def tzname(self):
        """Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        name = _call_tzinfo_method(self._tzinfo, 'tzname', self)
        _check_tzname(name)
        return name
    
    def dst(self):
        """Return 0 if DST is not in effect, or the DST offset (in minutes
        eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.dst(self)
        _check_utc_offset('dst', offset)
        return offset
    
    def __eq__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=True) == 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            return False
    
    def __ne__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=True) != 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            return True
    
    def __le__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) <= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)
    
    def __lt__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) < 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)
    
    def __ge__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) >= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)
    
    def __gt__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) > 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)
    
    def _cmp(self, other, allow_mixed=False):
        assert isinstance(other, datetime)
        mytz = self._tzinfo
        ottz = other._tzinfo
        myoff = otoff = None
        if mytz is ottz:
            base_compare = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare = myoff == otoff
        if base_compare:
            return _cmp((self._year, self._month, self._day, self._hour, self._minute, self._second, self._microsecond), (other._year, other._month, other._day, other._hour, other._minute, other._second, other._microsecond))
        if (myoff is None or otoff is None):
            if allow_mixed:
                return 2
            else:
                raise TypeError('cannot compare naive and aware datetimes')
        diff = self - other
        if diff.days < 0:
            return -1
        return ((diff and 1) or 0)
    
    def __add__(self, other):
        """Add a datetime and a timedelta."""
        if not isinstance(other, timedelta):
            return NotImplemented
        delta = timedelta(self.toordinal(), hours=self._hour, minutes=self._minute, seconds=self._second, microseconds=self._microsecond)
        delta += other
        (hour, rem) = divmod(delta.seconds, 3600)
        (minute, second) = divmod(rem, 60)
        if 0 < delta.days <= _MAXORDINAL:
            return datetime.combine(date.fromordinal(delta.days), time(hour, minute, second, delta.microseconds, tzinfo=self._tzinfo))
        raise OverflowError('result out of range')
    __radd__ = __add__
    
    def __sub__(self, other):
        """Subtract two datetimes, or a datetime and a timedelta."""
        if not isinstance(other, datetime):
            if isinstance(other, timedelta):
                return self + -other
            return NotImplemented
        days1 = self.toordinal()
        days2 = other.toordinal()
        secs1 = self._second + self._minute * 60 + self._hour * 3600
        secs2 = other._second + other._minute * 60 + other._hour * 3600
        base = timedelta(days1 - days2, secs1 - secs2, self._microsecond - other._microsecond)
        if self._tzinfo is other._tzinfo:
            return base
        myoff = self.utcoffset()
        otoff = other.utcoffset()
        if myoff == otoff:
            return base
        if (myoff is None or otoff is None):
            raise TypeError('cannot mix naive and timezone-aware time')
        return base + otoff - myoff
    
    def __hash__(self):
        tzoff = self.utcoffset()
        if tzoff is None:
            return hash(self._getstate()[0])
        days = _ymd2ord(self.year, self.month, self.day)
        seconds = self.hour * 3600 + self.minute * 60 + self.second
        return hash(timedelta(days, seconds, self.microsecond) - tzoff)
    
    def _getstate(self):
        (yhi, ylo) = divmod(self._year, 256)
        (us2, us3) = divmod(self._microsecond, 256)
        (us1, us2) = divmod(us2, 256)
        basestate = bytes([yhi, ylo, self._month, self._day, self._hour, self._minute, self._second, us1, us2, us3])
        if self._tzinfo is None:
            return (basestate, )
        else:
            return (basestate, self._tzinfo)
    
    def __setstate(self, string, tzinfo):
        (yhi, ylo, self._month, self._day, self._hour, self._minute, self._second, us1, us2, us3) = string
        self._year = yhi * 256 + ylo
        self._microsecond = (us1 << 8 | us2) << 8 | us3
        if (tzinfo is None or isinstance(tzinfo, _tzinfo_class)):
            self._tzinfo = tzinfo
        else:
            raise TypeError('bad tzinfo state arg %r' % tzinfo)
    
    def __reduce__(self):
        return (self.__class__, self._getstate())

datetime.min = datetime(1, 1, 1)
datetime.max = datetime(9999, 12, 31, 23, 59, 59, 999999)
datetime.resolution = timedelta(microseconds=1)

def _isoweek1monday(year):
    THURSDAY = 3
    firstday = _ymd2ord(year, 1, 1)
    firstweekday = (firstday + 6) % 7
    week1monday = firstday - firstweekday
    if firstweekday > THURSDAY:
        week1monday += 7
    return week1monday


class timezone(tzinfo):
    __slots__ = ('_offset', '_name')
    _Omitted = object()
    
    def __new__(cls, offset, name=_Omitted):
        if not isinstance(offset, timedelta):
            raise TypeError('offset must be a timedelta')
        if name is cls._Omitted:
            if not offset:
                return cls.utc
            name = None
        elif not isinstance(name, str):
            if (PY2 and isinstance(name, native_str)):
                name = name.decode()
            else:
                raise TypeError('name must be a string')
        if not cls._minoffset <= offset <= cls._maxoffset:
            raise ValueError('offset must be a timedelta strictly between -timedelta(hours=24) and timedelta(hours=24).')
        if (offset.microseconds != 0 or offset.seconds % 60 != 0):
            raise ValueError('offset must be a timedelta representing a whole number of minutes')
        return cls._create(offset, name)
    
    @classmethod
    def _create(cls, offset, name=None):
        self = tzinfo.__new__(cls)
        self._offset = offset
        self._name = name
        return self
    
    def __getinitargs__(self):
        """pickle support"""
        if self._name is None:
            return (self._offset, )
        return (self._offset, self._name)
    
    def __eq__(self, other):
        if type(other) != timezone:
            return False
        return self._offset == other._offset
    
    def __hash__(self):
        return hash(self._offset)
    
    def __repr__(self):
        """Convert to formal string, for repr().

        >>> tz = timezone.utc
        >>> repr(tz)
        'datetime.timezone.utc'
        >>> tz = timezone(timedelta(hours=-5), 'EST')
        >>> repr(tz)
        "datetime.timezone(datetime.timedelta(-1, 68400), 'EST')"
        """
        if self is self.utc:
            return 'datetime.timezone.utc'
        if self._name is None:
            return '%s(%r)' % ('datetime.' + self.__class__.__name__, self._offset)
        return '%s(%r, %r)' % ('datetime.' + self.__class__.__name__, self._offset, self._name)
    
    def __str__(self):
        return self.tzname(None)
    
    def utcoffset(self, dt):
        if (isinstance(dt, datetime) or dt is None):
            return self._offset
        raise TypeError('utcoffset() argument must be a datetime instance or None')
    
    def tzname(self, dt):
        if (isinstance(dt, datetime) or dt is None):
            if self._name is None:
                return self._name_from_offset(self._offset)
            return self._name
        raise TypeError('tzname() argument must be a datetime instance or None')
    
    def dst(self, dt):
        if (isinstance(dt, datetime) or dt is None):
            return None
        raise TypeError('dst() argument must be a datetime instance or None')
    
    def fromutc(self, dt):
        if isinstance(dt, datetime):
            if dt.tzinfo is not self:
                raise ValueError('fromutc: dt.tzinfo is not self')
            return dt + self._offset
        raise TypeError('fromutc() argument must be a datetime instance or None')
    _maxoffset = timedelta(hours=23, minutes=59)
    _minoffset = -_maxoffset
    
    @staticmethod
    def _name_from_offset(delta):
        if delta < timedelta(0):
            sign = '-'
            delta = -delta
        else:
            sign = '+'
        (hours, rest) = divmod(delta, timedelta(hours=1))
        minutes = rest // timedelta(minutes=1)
        return 'UTC{}{:02d}:{:02d}'.format(sign, hours, minutes)

timezone.utc = timezone._create(timedelta(0))
timezone.min = timezone._create(timezone._minoffset)
timezone.max = timezone._create(timezone._maxoffset)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
'\nSome time zone algebra.  For a datetime x, let\n    x.n = x stripped of its timezone -- its naive time.\n    x.o = x.utcoffset(), and assuming that doesn\'t raise an exception or\n          return None\n    x.d = x.dst(), and assuming that doesn\'t raise an exception or\n          return None\n    x.s = x\'s standard offset, x.o - x.d\n\nNow some derived rules, where k is a duration (timedelta).\n\n1. x.o = x.s + x.d\n   This follows from the definition of x.s.\n\n2. If x and y have the same tzinfo member, x.s = y.s.\n   This is actually a requirement, an assumption we need to make about\n   sane tzinfo classes.\n\n3. The naive UTC time corresponding to x is x.n - x.o.\n   This is again a requirement for a sane tzinfo class.\n\n4. (x+k).s = x.s\n   This follows from #2, and that datimetimetz+timedelta preserves tzinfo.\n\n5. (x+k).n = x.n + k\n   Again follows from how arithmetic is defined.\n\nNow we can explain tz.fromutc(x).  Let\'s assume it\'s an interesting case\n(meaning that the various tzinfo methods exist, and don\'t blow up or return\nNone when called).\n\nThe function wants to return a datetime y with timezone tz, equivalent to x.\nx is already in UTC.\n\nBy #3, we want\n\n    y.n - y.o = x.n                             [1]\n\nThe algorithm starts by attaching tz to x.n, and calling that y.  So\nx.n = y.n at the start.  Then it wants to add a duration k to y, so that [1]\nbecomes true; in effect, we want to solve [2] for k:\n\n   (y+k).n - (y+k).o = x.n                      [2]\n\nBy #1, this is the same as\n\n   (y+k).n - ((y+k).s + (y+k).d) = x.n          [3]\n\nBy #5, (y+k).n = y.n + k, which equals x.n + k because x.n=y.n at the start.\nSubstituting that into [3],\n\n   x.n + k - (y+k).s - (y+k).d = x.n; the x.n terms cancel, leaving\n   k - (y+k).s - (y+k).d = 0; rearranging,\n   k = (y+k).s - (y+k).d; by #4, (y+k).s == y.s, so\n   k = y.s - (y+k).d\n\nOn the RHS, (y+k).d can\'t be computed directly, but y.s can be, and we\napproximate k by ignoring the (y+k).d term at first.  Note that k can\'t be\nvery large, since all offset-returning methods return a duration of magnitude\nless than 24 hours.  For that reason, if y is firmly in std time, (y+k).d must\nbe 0, so ignoring it has no consequence then.\n\nIn any case, the new value is\n\n    z = y + y.s                                 [4]\n\nIt\'s helpful to step back at look at [4] from a higher level:  it\'s simply\nmapping from UTC to tz\'s standard time.\n\nAt this point, if\n\n    z.n - z.o = x.n                             [5]\n\nwe have an equivalent time, and are almost done.  The insecurity here is\nat the start of daylight time.  Picture US Eastern for concreteness.  The wall\ntime jumps from 1:59 to 3:00, and wall hours of the form 2:MM don\'t make good\nsense then.  The docs ask that an Eastern tzinfo class consider such a time to\nbe EDT (because it\'s "after 2"), which is a redundant spelling of 1:MM EST\non the day DST starts.  We want to return the 1:MM EST spelling because that\'s\nthe only spelling that makes sense on the local wall clock.\n\nIn fact, if [5] holds at this point, we do have the standard-time spelling,\nbut that takes a bit of proof.  We first prove a stronger result.  What\'s the\ndifference between the LHS and RHS of [5]?  Let\n\n    diff = x.n - (z.n - z.o)                    [6]\n\nNow\n    z.n =                       by [4]\n    (y + y.s).n =               by #5\n    y.n + y.s =                 since y.n = x.n\n    x.n + y.s =                 since z and y are have the same tzinfo member,\n                                    y.s = z.s by #2\n    x.n + z.s\n\nPlugging that back into [6] gives\n\n    diff =\n    x.n - ((x.n + z.s) - z.o) =     expanding\n    x.n - x.n - z.s + z.o =         cancelling\n    - z.s + z.o =                   by #2\n    z.d\n\nSo diff = z.d.\n\nIf [5] is true now, diff = 0, so z.d = 0 too, and we have the standard-time\nspelling we wanted in the endcase described above.  We\'re done.  Contrarily,\nif z.d = 0, then we have a UTC equivalent, and are also done.\n\nIf [5] is not true now, diff = z.d != 0, and z.d is the offset we need to\nadd to z (in effect, z is in tz\'s standard time, and we need to shift the\nlocal clock into tz\'s daylight time).\n\nLet\n\n    z\' = z + z.d = z + diff                     [7]\n\nand we can again ask whether\n\n    z\'.n - z\'.o = x.n                           [8]\n\nIf so, we\'re done.  If not, the tzinfo class is insane, according to the\nassumptions we\'ve made.  This also requires a bit of proof.  As before, let\'s\ncompute the difference between the LHS and RHS of [8] (and skipping some of\nthe justifications for the kinds of substitutions we\'ve done several times\nalready):\n\n    diff\' = x.n - (z\'.n - z\'.o) =           replacing z\'.n via [7]\n            x.n  - (z.n + diff - z\'.o) =    replacing diff via [6]\n            x.n - (z.n + x.n - (z.n - z.o) - z\'.o) =\n            x.n - z.n - x.n + z.n - z.o + z\'.o =    cancel x.n\n            - z.n + z.n - z.o + z\'.o =              cancel z.n\n            - z.o + z\'.o =                      #1 twice\n            -z.s - z.d + z\'.s + z\'.d =          z and z\' have same tzinfo\n            z\'.d - z.d\n\nSo z\' is UTC-equivalent to x iff z\'.d = z.d at this point.  If they are equal,\nwe\'ve found the UTC-equivalent so are done.  In fact, we stop with [7] and\nreturn z\', not bothering to compute z\'.d.\n\nHow could z.d and z\'d differ?  z\' = z + z.d [7], so merely moving z\' by\na dst() offset, and starting *from* a time already in DST (we know z.d != 0),\nwould have to change the result dst() returns:  we start in DST, and moving\na little further into it takes us out of DST.\n\nThere isn\'t a sane case where this can happen.  The closest it gets is at\nthe end of DST, where there\'s an hour in UTC with no spelling in a hybrid\ntzinfo class.  In US Eastern, that\'s 5:MM UTC = 0:MM EST = 1:MM EDT.  During\nthat hour, on an Eastern clock 1:MM is taken as being in standard time (6:MM\nUTC) because the docs insist on that, but 0:MM is taken as being in daylight\ntime (4:MM UTC).  There is no local time mapping to 5:MM UTC.  The local\nclock jumps from 1:59 back to 1:00 again, and repeats the 1:MM hour in\nstandard time.  Since that\'s what the local clock *does*, we want to map both\nUTC hours 5:MM and 6:MM to 1:MM Eastern.  The result is ambiguous\nin local time, but so it goes -- it\'s the way the local clock works.\n\nWhen x = 5:MM UTC is the input to this algorithm, x.o=0, y.o=-5 and y.d=0,\nso z=0:MM.  z.d=60 (minutes) then, so [5] doesn\'t hold and we keep going.\nz\' = z + z.d = 1:MM then, and z\'.d=0, and z\'.d - z.d = -60 != 0 so [8]\n(correctly) concludes that z\' is not UTC-equivalent to x.\n\nBecause we know z.d said z was in daylight time (else [5] would have held and\nwe would have stopped then), and we know z.d != z\'.d (else [8] would have held\nand we have stopped then), and there are only 2 possible values dst() can\nreturn in Eastern, it follows that z\'.d must be 0 (which it is in the example,\nbut the reasoning doesn\'t depend on the example -- it depends on there being\ntwo possible dst() outcomes, one zero and the other non-zero).  Therefore\nz\' must be in standard time, and is the spelling we want in this case.\n\nNote again that z\' is not UTC-equivalent as far as the hybrid tzinfo class is\nconcerned (because it takes z\' as being in standard time rather than the\ndaylight time we intend here), but returning it gives the real-life "local\nclock repeats an hour" behavior when mapping the "unspellable" UTC hour into\ntz.\n\nWhen the input is 6:MM, z=1:MM and z.d=0, and we stop at once, again with\nthe 1:MM standard time spelling we want.\n\nSo how can this break?  One of the assumptions must be violated.  Two\npossibilities:\n\n1) [2] effectively says that y.s is invariant across all y belong to a given\n   time zone.  This isn\'t true if, for political reasons or continental drift,\n   a region decides to change its base offset from UTC.\n\n2) There may be versions of "double daylight" time where the tail end of\n   the analysis gives up a step too early.  I haven\'t thought about that\n   enough to say.\n\nIn any case, it\'s clear that the default fromutc() is strong enough to handle\n"almost all" time zones:  so long as the standard offset is invariant, it\ndoesn\'t matter if daylight time transition points change from year to year, or\nif daylight time is skipped in some years; it doesn\'t matter how large or\nsmall dst() may get within its bounds; and it doesn\'t even matter if some\nperverse time zone returns a negative dst()).  So a breaking case must be\npretty bizarre, and a tzinfo subclass can override fromutc() if it is.\n'
try:
    from _datetime import *
except ImportError:
    pass
else:
    del (_DAYNAMES, _DAYS_BEFORE_MONTH, _DAYS_IN_MONTH, _DI100Y, _DI400Y, _DI4Y, _MAXORDINAL, _MONTHNAMES, _build_struct_time, _call_tzinfo_method, _check_date_fields, _check_time_fields, _check_tzinfo_arg, _check_tzname, _check_utc_offset, _cmp, _cmperror, _date_class, _days_before_month, _days_before_year, _days_in_month, _format_time, _is_leap, _isoweek1monday, _math, _ord2ymd, _time, _time_class, _tzinfo_class, _wrap_strftime, _ymd2ord)
    from _datetime import __doc__

