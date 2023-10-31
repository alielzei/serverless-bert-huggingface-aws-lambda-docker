from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six

def guess_content_type(filename, default='application/octet-stream'):
    """
    Guess the "Content-Type" of a file.

    :param filename:
        The filename to guess the "Content-Type" of using :mod:`mimetypes`.
    :param default:
        If no "Content-Type" can be guessed, default to `default`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.fields.guess_content_type', "guess_content_type(filename, default='application/octet-stream')", {'mimetypes': mimetypes, 'filename': filename, 'default': default}, 1)

def format_header_param_rfc2231(name, value):
    """
    Helper function to format and quote a single header parameter using the
    strategy defined in RFC 2231.

    Particularly useful for header parameters which might contain
    non-ASCII values, like file names. This follows
    `RFC 2388 Section 4.4 <https://tools.ietf.org/html/rfc2388#section-4.4>`_.

    :param name:
        The name of the parameter, a string expected to be ASCII only.
    :param value:
        The value of the parameter, provided as ``bytes`` or `str``.
    :ret:
        An RFC-2231-formatted unicode string.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.fields.format_header_param_rfc2231', 'format_header_param_rfc2231(name, value)', {'six': six, 'email': email, 'name': name, 'value': value}, 1)
_HTML5_REPLACEMENTS = {'"': '%22', '\\': '\\\\'}
_HTML5_REPLACEMENTS.update({six.unichr(cc): '%{:02X}'.format(cc) for cc in range(0, 31 + 1) if cc not in (27, )})

def _replace_multiple(value, needles_and_replacements):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.fields._replace_multiple', '_replace_multiple(value, needles_and_replacements)', {'re': re, 'value': value, 'needles_and_replacements': needles_and_replacements}, 1)

def format_header_param_html5(name, value):
    """
    Helper function to format and quote a single header parameter using the
    HTML5 strategy.

    Particularly useful for header parameters which might contain
    non-ASCII values, like file names. This follows the `HTML5 Working Draft
    Section 4.10.22.7`_ and matches the behavior of curl and modern browsers.

    .. _HTML5 Working Draft Section 4.10.22.7:
        https://w3c.github.io/html/sec-forms.html#multipart-form-data

    :param name:
        The name of the parameter, a string expected to be ASCII only.
    :param value:
        The value of the parameter, provided as ``bytes`` or `str``.
    :ret:
        A unicode string, stripped of troublesome characters.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.fields.format_header_param_html5', 'format_header_param_html5(name, value)', {'six': six, '_replace_multiple': _replace_multiple, '_HTML5_REPLACEMENTS': _HTML5_REPLACEMENTS, 'name': name, 'value': value}, 1)
format_header_param = format_header_param_html5


class RequestField(object):
    """
    A data container for request body parameters.

    :param name:
        The name of this request field. Must be unicode.
    :param data:
        The data/value body.
    :param filename:
        An optional filename of the request field. Must be unicode.
    :param headers:
        An optional dict-like object of headers to initially use for the field.
    :param header_formatter:
        An optional callable that is used to encode and format the headers. By
        default, this is :func:`format_header_param_html5`.
    """
    
    def __init__(self, name, data, filename=None, headers=None, header_formatter=format_header_param_html5):
        self._name = name
        self._filename = filename
        self.data = data
        self.headers = {}
        if headers:
            self.headers = dict(headers)
        self.header_formatter = header_formatter
    
    @classmethod
    def from_tuples(cls, fieldname, value, header_formatter=format_header_param_html5):
        """
        A :class:`~urllib3.fields.RequestField` factory from old-style tuple parameters.

        Supports constructing :class:`~urllib3.fields.RequestField` from
        parameter of key/value strings AND key/filetuple. A filetuple is a
        (filename, data, MIME type) tuple where the MIME type is optional.
        For example::

            'foo': 'bar',
            'fakefile': ('foofile.txt', 'contents of foofile'),
            'realfile': ('barfile.txt', open('realfile').read()),
            'typedfile': ('bazfile.bin', open('bazfile').read(), 'image/jpeg'),
            'nonamefile': 'contents of nonamefile field',

        Field names and filenames must be unicode.
        """
        if isinstance(value, tuple):
            if len(value) == 3:
                (filename, data, content_type) = value
            else:
                (filename, data) = value
                content_type = guess_content_type(filename)
        else:
            filename = None
            content_type = None
            data = value
        request_param = cls(fieldname, data, filename=filename, header_formatter=header_formatter)
        request_param.make_multipart(content_type=content_type)
        return request_param
    
    def _render_part(self, name, value):
        """
        Overridable helper function to format a single header parameter. By
        default, this calls ``self.header_formatter``.

        :param name:
            The name of the parameter, a string expected to be ASCII only.
        :param value:
            The value of the parameter, provided as a unicode string.
        """
        return self.header_formatter(name, value)
    
    def _render_parts(self, header_parts):
        """
        Helper function to format and quote a single header.

        Useful for single headers that are composed of multiple items. E.g.,
        'Content-Disposition' fields.

        :param header_parts:
            A sequence of (k, v) tuples or a :class:`dict` of (k, v) to format
            as `k1="v1"; k2="v2"; ...`.
        """
        parts = []
        iterable = header_parts
        if isinstance(header_parts, dict):
            iterable = header_parts.items()
        for (name, value) in iterable:
            if value is not None:
                parts.append(self._render_part(name, value))
        return '; '.join(parts)
    
    def render_headers(self):
        """
        Renders the headers for this request field.
        """
        lines = []
        sort_keys = ['Content-Disposition', 'Content-Type', 'Content-Location']
        for sort_key in sort_keys:
            if self.headers.get(sort_key, False):
                lines.append('%s: %s' % (sort_key, self.headers[sort_key]))
        for (header_name, header_value) in self.headers.items():
            if header_name not in sort_keys:
                if header_value:
                    lines.append('%s: %s' % (header_name, header_value))
        lines.append('\r\n')
        return '\r\n'.join(lines)
    
    def make_multipart(self, content_disposition=None, content_type=None, content_location=None):
        """
        Makes this request field into a multipart request field.

        This method overrides "Content-Disposition", "Content-Type" and
        "Content-Location" headers to the request parameter.

        :param content_type:
            The 'Content-Type' of the request body.
        :param content_location:
            The 'Content-Location' of the request body.

        """
        self.headers['Content-Disposition'] = (content_disposition or 'form-data')
        self.headers['Content-Disposition'] += '; '.join(['', self._render_parts((('name', self._name), ('filename', self._filename)))])
        self.headers['Content-Type'] = content_type
        self.headers['Content-Location'] = content_location


