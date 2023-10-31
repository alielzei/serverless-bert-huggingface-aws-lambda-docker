from __future__ import absolute_import
import binascii
import codecs
import os
from io import BytesIO
from .fields import RequestField
from .packages import six
from .packages.six import b
writer = codecs.lookup('utf-8')[3]

def choose_boundary():
    """
    Our embarrassingly-simple replacement for mimetools.choose_boundary.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.filepost.choose_boundary', 'choose_boundary()', {'binascii': binascii, 'os': os, 'six': six}, 1)

def iter_field_objects(fields):
    """
    Iterate over fields.

    Supports list of (k, v) tuples and dicts, and lists of
    :class:`~urllib3.fields.RequestField`.

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('urllib3.filepost.iter_field_objects', 'iter_field_objects(fields)', {'six': six, 'RequestField': RequestField, 'fields': fields}, 0)

def iter_fields(fields):
    """
    .. deprecated:: 1.6

    Iterate over fields.

    The addition of :class:`~urllib3.fields.RequestField` makes this function
    obsolete. Instead, use :func:`iter_field_objects`, which returns
    :class:`~urllib3.fields.RequestField` objects.

    Supports list of (k, v) tuples and dicts.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.filepost.iter_fields', 'iter_fields(fields)', {'six': six, 'fields': fields}, 1)

def encode_multipart_formdata(fields, boundary=None):
    """
    Encode a dictionary of ``fields`` using the multipart/form-data MIME format.

    :param fields:
        Dictionary of fields or list of (key, :class:`~urllib3.fields.RequestField`).

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`urllib3.filepost.choose_boundary`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.filepost.encode_multipart_formdata', 'encode_multipart_formdata(fields, boundary=None)', {'BytesIO': BytesIO, 'choose_boundary': choose_boundary, 'iter_field_objects': iter_field_objects, 'b': b, 'writer': writer, 'six': six, 'fields': fields, 'boundary': boundary}, 2)

