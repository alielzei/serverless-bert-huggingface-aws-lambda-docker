"""Contains well known classes.

This files defines well known classes which need extra maintenance including:
  - Any
  - Duration
  - FieldMask
  - Struct
  - Timestamp
"""

__author__ = 'jieluo@google.com (Jie Luo)'
import calendar
import collections.abc
import datetime
from google.protobuf.descriptor import FieldDescriptor
_TIMESTAMPFOMAT = '%Y-%m-%dT%H:%M:%S'
_NANOS_PER_SECOND = 1000000000
_NANOS_PER_MILLISECOND = 1000000
_NANOS_PER_MICROSECOND = 1000
_MILLIS_PER_SECOND = 1000
_MICROS_PER_SECOND = 1000000
_SECONDS_PER_DAY = 24 * 3600
_DURATION_SECONDS_MAX = 315576000000


class Any(object):
    """Class for Any Message type."""
    __slots__ = ()
    
    def Pack(self, msg, type_url_prefix='type.googleapis.com/', deterministic=None):
        """Packs the specified message into current Any message."""
        if (len(type_url_prefix) < 1 or type_url_prefix[-1] != '/'):
            self.type_url = '%s/%s' % (type_url_prefix, msg.DESCRIPTOR.full_name)
        else:
            self.type_url = '%s%s' % (type_url_prefix, msg.DESCRIPTOR.full_name)
        self.value = msg.SerializeToString(deterministic=deterministic)
    
    def Unpack(self, msg):
        """Unpacks the current Any message into specified message."""
        descriptor = msg.DESCRIPTOR
        if not self.Is(descriptor):
            return False
        msg.ParseFromString(self.value)
        return True
    
    def TypeName(self):
        """Returns the protobuf type name of the inner message."""
        return self.type_url.split('/')[-1]
    
    def Is(self, descriptor):
        """Checks if this Any represents the given protobuf type."""
        return ('/' in self.type_url and self.TypeName() == descriptor.full_name)

_EPOCH_DATETIME_NAIVE = datetime.datetime.utcfromtimestamp(0)
_EPOCH_DATETIME_AWARE = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)


class Timestamp(object):
    """Class for Timestamp message type."""
    __slots__ = ()
    
    def ToJsonString(self):
        """Converts Timestamp to RFC 3339 date string format.

    Returns:
      A string converted from timestamp. The string is always Z-normalized
      and uses 3, 6 or 9 fractional digits as required to represent the
      exact time. Example of the return format: '1972-01-01T10:00:20.021Z'
    """
        nanos = self.nanos % _NANOS_PER_SECOND
        total_sec = self.seconds + (self.nanos - nanos) // _NANOS_PER_SECOND
        seconds = total_sec % _SECONDS_PER_DAY
        days = (total_sec - seconds) // _SECONDS_PER_DAY
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(days, seconds)
        result = dt.isoformat()
        if nanos % 1000000000.0 == 0:
            return result + 'Z'
        if nanos % 1000000.0 == 0:
            return result + '.%03dZ' % (nanos / 1000000.0)
        if nanos % 1000.0 == 0:
            return result + '.%06dZ' % (nanos / 1000.0)
        return result + '.%09dZ' % nanos
    
    def FromJsonString(self, value):
        """Parse a RFC 3339 date string format to Timestamp.

    Args:
      value: A date string. Any fractional digits (or none) and any offset are
          accepted as long as they fit into nano-seconds precision.
          Example of accepted format: '1972-01-01T10:00:20.021-05:00'

    Raises:
      ValueError: On parsing problems.
    """
        if not isinstance(value, str):
            raise ValueError('Timestamp JSON value not a string: {!r}'.format(value))
        timezone_offset = value.find('Z')
        if timezone_offset == -1:
            timezone_offset = value.find('+')
        if timezone_offset == -1:
            timezone_offset = value.rfind('-')
        if timezone_offset == -1:
            raise ValueError('Failed to parse timestamp: missing valid timezone offset.')
        time_value = value[0:timezone_offset]
        point_position = time_value.find('.')
        if point_position == -1:
            second_value = time_value
            nano_value = ''
        else:
            second_value = time_value[:point_position]
            nano_value = time_value[point_position + 1:]
        if 't' in second_value:
            raise ValueError("time data '{0}' does not match format '%Y-%m-%dT%H:%M:%S', lowercase 't' is not accepted".format(second_value))
        date_object = datetime.datetime.strptime(second_value, _TIMESTAMPFOMAT)
        td = date_object - datetime.datetime(1970, 1, 1)
        seconds = td.seconds + td.days * _SECONDS_PER_DAY
        if len(nano_value) > 9:
            raise ValueError('Failed to parse Timestamp: nanos {0} more than 9 fractional digits.'.format(nano_value))
        if nano_value:
            nanos = round(float('0.' + nano_value) * 1000000000.0)
        else:
            nanos = 0
        if value[timezone_offset] == 'Z':
            if len(value) != timezone_offset + 1:
                raise ValueError('Failed to parse timestamp: invalid trailing data {0}.'.format(value))
        else:
            timezone = value[timezone_offset:]
            pos = timezone.find(':')
            if pos == -1:
                raise ValueError('Invalid timezone offset value: {0}.'.format(timezone))
            if timezone[0] == '+':
                seconds -= (int(timezone[1:pos]) * 60 + int(timezone[pos + 1:])) * 60
            else:
                seconds += (int(timezone[1:pos]) * 60 + int(timezone[pos + 1:])) * 60
        self.seconds = int(seconds)
        self.nanos = int(nanos)
    
    def GetCurrentTime(self):
        """Get the current UTC into Timestamp."""
        self.FromDatetime(datetime.datetime.utcnow())
    
    def ToNanoseconds(self):
        """Converts Timestamp to nanoseconds since epoch."""
        return self.seconds * _NANOS_PER_SECOND + self.nanos
    
    def ToMicroseconds(self):
        """Converts Timestamp to microseconds since epoch."""
        return self.seconds * _MICROS_PER_SECOND + self.nanos // _NANOS_PER_MICROSECOND
    
    def ToMilliseconds(self):
        """Converts Timestamp to milliseconds since epoch."""
        return self.seconds * _MILLIS_PER_SECOND + self.nanos // _NANOS_PER_MILLISECOND
    
    def ToSeconds(self):
        """Converts Timestamp to seconds since epoch."""
        return self.seconds
    
    def FromNanoseconds(self, nanos):
        """Converts nanoseconds since epoch to Timestamp."""
        self.seconds = nanos // _NANOS_PER_SECOND
        self.nanos = nanos % _NANOS_PER_SECOND
    
    def FromMicroseconds(self, micros):
        """Converts microseconds since epoch to Timestamp."""
        self.seconds = micros // _MICROS_PER_SECOND
        self.nanos = micros % _MICROS_PER_SECOND * _NANOS_PER_MICROSECOND
    
    def FromMilliseconds(self, millis):
        """Converts milliseconds since epoch to Timestamp."""
        self.seconds = millis // _MILLIS_PER_SECOND
        self.nanos = millis % _MILLIS_PER_SECOND * _NANOS_PER_MILLISECOND
    
    def FromSeconds(self, seconds):
        """Converts seconds since epoch to Timestamp."""
        self.seconds = seconds
        self.nanos = 0
    
    def ToDatetime(self, tzinfo=None):
        """Converts Timestamp to a datetime.

    Args:
      tzinfo: A datetime.tzinfo subclass; defaults to None.

    Returns:
      If tzinfo is None, returns a timezone-naive UTC datetime (with no timezone
      information, i.e. not aware that it's UTC).

      Otherwise, returns a timezone-aware datetime in the input timezone.
    """
        delta = datetime.timedelta(seconds=self.seconds, microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND))
        if tzinfo is None:
            return _EPOCH_DATETIME_NAIVE + delta
        else:
            return _EPOCH_DATETIME_AWARE.astimezone(tzinfo) + delta
    
    def FromDatetime(self, dt):
        """Converts datetime to Timestamp.

    Args:
      dt: A datetime. If it's timezone-naive, it's assumed to be in UTC.
    """
        self.seconds = calendar.timegm(dt.utctimetuple())
        self.nanos = dt.microsecond * _NANOS_PER_MICROSECOND



class Duration(object):
    """Class for Duration message type."""
    __slots__ = ()
    
    def ToJsonString(self):
        """Converts Duration to string format.

    Returns:
      A string converted from self. The string format will contains
      3, 6, or 9 fractional digits depending on the precision required to
      represent the exact Duration value. For example: "1s", "1.010s",
      "1.000000100s", "-3.100s"
    """
        _CheckDurationValid(self.seconds, self.nanos)
        if (self.seconds < 0 or self.nanos < 0):
            result = '-'
            seconds = -self.seconds + int((0 - self.nanos) // 1000000000.0)
            nanos = (0 - self.nanos) % 1000000000.0
        else:
            result = ''
            seconds = self.seconds + int(self.nanos // 1000000000.0)
            nanos = self.nanos % 1000000000.0
        result += '%d' % seconds
        if nanos % 1000000000.0 == 0:
            return result + 's'
        if nanos % 1000000.0 == 0:
            return result + '.%03ds' % (nanos / 1000000.0)
        if nanos % 1000.0 == 0:
            return result + '.%06ds' % (nanos / 1000.0)
        return result + '.%09ds' % nanos
    
    def FromJsonString(self, value):
        """Converts a string to Duration.

    Args:
      value: A string to be converted. The string must end with 's'. Any
          fractional digits (or none) are accepted as long as they fit into
          precision. For example: "1s", "1.01s", "1.0000001s", "-3.100s

    Raises:
      ValueError: On parsing problems.
    """
        if not isinstance(value, str):
            raise ValueError('Duration JSON value not a string: {!r}'.format(value))
        if (len(value) < 1 or value[-1] != 's'):
            raise ValueError('Duration must end with letter "s": {0}.'.format(value))
        try:
            pos = value.find('.')
            if pos == -1:
                seconds = int(value[:-1])
                nanos = 0
            else:
                seconds = int(value[:pos])
                if value[0] == '-':
                    nanos = int(round(float('-0{0}'.format(value[pos:-1])) * 1000000000.0))
                else:
                    nanos = int(round(float('0{0}'.format(value[pos:-1])) * 1000000000.0))
            _CheckDurationValid(seconds, nanos)
            self.seconds = seconds
            self.nanos = nanos
        except ValueError as e:
            raise ValueError("Couldn't parse duration: {0} : {1}.".format(value, e))
    
    def ToNanoseconds(self):
        """Converts a Duration to nanoseconds."""
        return self.seconds * _NANOS_PER_SECOND + self.nanos
    
    def ToMicroseconds(self):
        """Converts a Duration to microseconds."""
        micros = _RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND)
        return self.seconds * _MICROS_PER_SECOND + micros
    
    def ToMilliseconds(self):
        """Converts a Duration to milliseconds."""
        millis = _RoundTowardZero(self.nanos, _NANOS_PER_MILLISECOND)
        return self.seconds * _MILLIS_PER_SECOND + millis
    
    def ToSeconds(self):
        """Converts a Duration to seconds."""
        return self.seconds
    
    def FromNanoseconds(self, nanos):
        """Converts nanoseconds to Duration."""
        self._NormalizeDuration(nanos // _NANOS_PER_SECOND, nanos % _NANOS_PER_SECOND)
    
    def FromMicroseconds(self, micros):
        """Converts microseconds to Duration."""
        self._NormalizeDuration(micros // _MICROS_PER_SECOND, micros % _MICROS_PER_SECOND * _NANOS_PER_MICROSECOND)
    
    def FromMilliseconds(self, millis):
        """Converts milliseconds to Duration."""
        self._NormalizeDuration(millis // _MILLIS_PER_SECOND, millis % _MILLIS_PER_SECOND * _NANOS_PER_MILLISECOND)
    
    def FromSeconds(self, seconds):
        """Converts seconds to Duration."""
        self.seconds = seconds
        self.nanos = 0
    
    def ToTimedelta(self):
        """Converts Duration to timedelta."""
        return datetime.timedelta(seconds=self.seconds, microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND))
    
    def FromTimedelta(self, td):
        """Converts timedelta to Duration."""
        self._NormalizeDuration(td.seconds + td.days * _SECONDS_PER_DAY, td.microseconds * _NANOS_PER_MICROSECOND)
    
    def _NormalizeDuration(self, seconds, nanos):
        """Set Duration by seconds and nanos."""
        if (seconds < 0 and nanos > 0):
            seconds += 1
            nanos -= _NANOS_PER_SECOND
        self.seconds = seconds
        self.nanos = nanos


def _CheckDurationValid(seconds, nanos):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._CheckDurationValid', '_CheckDurationValid(seconds, nanos)', {'_DURATION_SECONDS_MAX': _DURATION_SECONDS_MAX, '_NANOS_PER_SECOND': _NANOS_PER_SECOND, 'seconds': seconds, 'nanos': nanos}, 0)

def _RoundTowardZero(value, divider):
    """Truncates the remainder part after division."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._RoundTowardZero', '_RoundTowardZero(value, divider)', {'value': value, 'divider': divider}, 1)


class FieldMask(object):
    """Class for FieldMask message type."""
    __slots__ = ()
    
    def ToJsonString(self):
        """Converts FieldMask to string according to proto3 JSON spec."""
        camelcase_paths = []
        for path in self.paths:
            camelcase_paths.append(_SnakeCaseToCamelCase(path))
        return ','.join(camelcase_paths)
    
    def FromJsonString(self, value):
        """Converts string to FieldMask according to proto3 JSON spec."""
        if not isinstance(value, str):
            raise ValueError('FieldMask JSON value not a string: {!r}'.format(value))
        self.Clear()
        if value:
            for path in value.split(','):
                self.paths.append(_CamelCaseToSnakeCase(path))
    
    def IsValidForDescriptor(self, message_descriptor):
        """Checks whether the FieldMask is valid for Message Descriptor."""
        for path in self.paths:
            if not _IsValidPath(message_descriptor, path):
                return False
        return True
    
    def AllFieldsFromDescriptor(self, message_descriptor):
        """Gets all direct fields of Message Descriptor to FieldMask."""
        self.Clear()
        for field in message_descriptor.fields:
            self.paths.append(field.name)
    
    def CanonicalFormFromMask(self, mask):
        """Converts a FieldMask to the canonical form.

    Removes paths that are covered by another path. For example,
    "foo.bar" is covered by "foo" and will be removed if "foo"
    is also in the FieldMask. Then sorts all paths in alphabetical order.

    Args:
      mask: The original FieldMask to be converted.
    """
        tree = _FieldMaskTree(mask)
        tree.ToFieldMask(self)
    
    def Union(self, mask1, mask2):
        """Merges mask1 and mask2 into this FieldMask."""
        _CheckFieldMaskMessage(mask1)
        _CheckFieldMaskMessage(mask2)
        tree = _FieldMaskTree(mask1)
        tree.MergeFromFieldMask(mask2)
        tree.ToFieldMask(self)
    
    def Intersect(self, mask1, mask2):
        """Intersects mask1 and mask2 into this FieldMask."""
        _CheckFieldMaskMessage(mask1)
        _CheckFieldMaskMessage(mask2)
        tree = _FieldMaskTree(mask1)
        intersection = _FieldMaskTree()
        for path in mask2.paths:
            tree.IntersectPath(path, intersection)
        intersection.ToFieldMask(self)
    
    def MergeMessage(self, source, destination, replace_message_field=False, replace_repeated_field=False):
        """Merges fields specified in FieldMask from source to destination.

    Args:
      source: Source message.
      destination: The destination message to be merged into.
      replace_message_field: Replace message field if True. Merge message
          field if False.
      replace_repeated_field: Replace repeated field if True. Append
          elements of repeated field if False.
    """
        tree = _FieldMaskTree(self)
        tree.MergeMessage(source, destination, replace_message_field, replace_repeated_field)


def _IsValidPath(message_descriptor, path):
    """Checks whether the path is valid for Message Descriptor."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._IsValidPath', '_IsValidPath(message_descriptor, path)', {'FieldDescriptor': FieldDescriptor, 'message_descriptor': message_descriptor, 'path': path}, 1)

def _CheckFieldMaskMessage(message):
    """Raises ValueError if message is not a FieldMask."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._CheckFieldMaskMessage', '_CheckFieldMaskMessage(message)', {'message': message}, 0)

def _SnakeCaseToCamelCase(path_name):
    """Converts a path name from snake_case to camelCase."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._SnakeCaseToCamelCase', '_SnakeCaseToCamelCase(path_name)', {'path_name': path_name}, 1)

def _CamelCaseToSnakeCase(path_name):
    """Converts a field name from camelCase to snake_case."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._CamelCaseToSnakeCase', '_CamelCaseToSnakeCase(path_name)', {'path_name': path_name}, 1)


class _FieldMaskTree(object):
    """Represents a FieldMask in a tree structure.

  For example, given a FieldMask "foo.bar,foo.baz,bar.baz",
  the FieldMaskTree will be:
      [_root] -+- foo -+- bar
            |       |
            |       +- baz
            |
            +- bar --- baz
  In the tree, each leaf node represents a field path.
  """
    __slots__ = ('_root', )
    
    def __init__(self, field_mask=None):
        """Initializes the tree by FieldMask."""
        self._root = {}
        if field_mask:
            self.MergeFromFieldMask(field_mask)
    
    def MergeFromFieldMask(self, field_mask):
        """Merges a FieldMask to the tree."""
        for path in field_mask.paths:
            self.AddPath(path)
    
    def AddPath(self, path):
        """Adds a field path into the tree.

    If the field path to add is a sub-path of an existing field path
    in the tree (i.e., a leaf node), it means the tree already matches
    the given path so nothing will be added to the tree. If the path
    matches an existing non-leaf node in the tree, that non-leaf node
    will be turned into a leaf node with all its children removed because
    the path matches all the node's children. Otherwise, a new path will
    be added.

    Args:
      path: The field path to add.
    """
        node = self._root
        for name in path.split('.'):
            if name not in node:
                node[name] = {}
            elif not node[name]:
                return
            node = node[name]
        node.clear()
    
    def ToFieldMask(self, field_mask):
        """Converts the tree to a FieldMask."""
        field_mask.Clear()
        _AddFieldPaths(self._root, '', field_mask)
    
    def IntersectPath(self, path, intersection):
        """Calculates the intersection part of a field path with this tree.

    Args:
      path: The field path to calculates.
      intersection: The out tree to record the intersection part.
    """
        node = self._root
        for name in path.split('.'):
            if name not in node:
                return
            elif not node[name]:
                intersection.AddPath(path)
                return
            node = node[name]
        intersection.AddLeafNodes(path, node)
    
    def AddLeafNodes(self, prefix, node):
        """Adds leaf nodes begin with prefix to this tree."""
        if not node:
            self.AddPath(prefix)
        for name in node:
            child_path = prefix + '.' + name
            self.AddLeafNodes(child_path, node[name])
    
    def MergeMessage(self, source, destination, replace_message, replace_repeated):
        """Merge all fields specified by this tree from source to destination."""
        _MergeMessage(self._root, source, destination, replace_message, replace_repeated)


def _StrConvert(value):
    """Converts value to str if it is not."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._StrConvert', '_StrConvert(value)', {'value': value}, 1)

def _MergeMessage(node, source, destination, replace_message, replace_repeated):
    """Merge all fields specified by a sub-tree from source to destination."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._MergeMessage', '_MergeMessage(node, source, destination, replace_message, replace_repeated)', {'FieldDescriptor': FieldDescriptor, '_MergeMessage': _MergeMessage, '_StrConvert': _StrConvert, 'node': node, 'source': source, 'destination': destination, 'replace_message': replace_message, 'replace_repeated': replace_repeated}, 0)

def _AddFieldPaths(node, prefix, field_mask):
    """Adds the field paths descended from node to field_mask."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._AddFieldPaths', '_AddFieldPaths(node, prefix, field_mask)', {'_AddFieldPaths': _AddFieldPaths, 'node': node, 'prefix': prefix, 'field_mask': field_mask}, 1)

def _SetStructValue(struct_value, value):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._SetStructValue', '_SetStructValue(struct_value, value)', {'Struct': Struct, 'ListValue': ListValue, 'struct_value': struct_value, 'value': value}, 0)

def _GetStructValue(struct_value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.well_known_types._GetStructValue', '_GetStructValue(struct_value)', {'struct_value': struct_value}, 1)


class Struct(object):
    """Class for Struct message type."""
    __slots__ = ()
    
    def __getitem__(self, key):
        return _GetStructValue(self.fields[key])
    
    def __contains__(self, item):
        return item in self.fields
    
    def __setitem__(self, key, value):
        _SetStructValue(self.fields[key], value)
    
    def __delitem__(self, key):
        del self.fields[key]
    
    def __len__(self):
        return len(self.fields)
    
    def __iter__(self):
        return iter(self.fields)
    
    def keys(self):
        return self.fields.keys()
    
    def values(self):
        return [self[key] for key in self]
    
    def items(self):
        return [(key, self[key]) for key in self]
    
    def get_or_create_list(self, key):
        """Returns a list for this key, creating if it didn't exist already."""
        if not self.fields[key].HasField('list_value'):
            self.fields[key].list_value.Clear()
        return self.fields[key].list_value
    
    def get_or_create_struct(self, key):
        """Returns a struct for this key, creating if it didn't exist already."""
        if not self.fields[key].HasField('struct_value'):
            self.fields[key].struct_value.Clear()
        return self.fields[key].struct_value
    
    def update(self, dictionary):
        for (key, value) in dictionary.items():
            _SetStructValue(self.fields[key], value)

collections.abc.MutableMapping.register(Struct)


class ListValue(object):
    """Class for ListValue message type."""
    __slots__ = ()
    
    def __len__(self):
        return len(self.values)
    
    def append(self, value):
        _SetStructValue(self.values.add(), value)
    
    def extend(self, elem_seq):
        for value in elem_seq:
            self.append(value)
    
    def __getitem__(self, index):
        """Retrieves item by the specified index."""
        return _GetStructValue(self.values.__getitem__(index))
    
    def __setitem__(self, index, value):
        _SetStructValue(self.values.__getitem__(index), value)
    
    def __delitem__(self, key):
        del self.values[key]
    
    def items(self):
        for i in range(len(self)):
            yield self[i]
    
    def add_struct(self):
        """Appends and returns a struct value as the next value in the list."""
        struct_value = self.values.add().struct_value
        struct_value.Clear()
        return struct_value
    
    def add_list(self):
        """Appends and returns a list value as the next value in the list."""
        list_value = self.values.add().list_value
        list_value.Clear()
        return list_value

collections.abc.MutableSequence.register(ListValue)
WKTBASES = {'google.protobuf.Any': Any, 'google.protobuf.Duration': Duration, 'google.protobuf.FieldMask': FieldMask, 'google.protobuf.ListValue': ListValue, 'google.protobuf.Struct': Struct, 'google.protobuf.Timestamp': Timestamp}

