"""
A place for code to be called from the implementation of np.dtype

String handling is much easier to do correctly in python.
"""

import numpy as np
_kind_to_stem = {'u': 'uint', 'i': 'int', 'c': 'complex', 'f': 'float', 'b': 'bool', 'V': 'void', 'O': 'object', 'M': 'datetime', 'm': 'timedelta', 'S': 'bytes', 'U': 'str'}

def _kind_name(dtype):
    try:
        return _kind_to_stem[dtype.kind]
    except KeyError as e:
        raise RuntimeError('internal dtype error, unknown kind {!r}'.format(dtype.kind)) from None

def __str__(dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype.__str__', '__str__(dtype)', {'_struct_str': _struct_str, '_subarray_str': _subarray_str, 'np': np, 'dtype': dtype}, 1)

def __repr__(dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype.__repr__', '__repr__(dtype)', {'_construction_repr': _construction_repr, 'dtype': dtype}, 1)

def _unpack_field(dtype, offset, title=None):
    """
    Helper function to normalize the items in dtype.fields.

    Call as:

    dtype, offset, title = _unpack_field(*dtype.fields[name])
    """
    return (dtype, offset, title)

def _isunsized(dtype):
    return dtype.itemsize == 0

def _construction_repr(dtype, include_align=False, short=False):
    """
    Creates a string repr of the dtype, excluding the 'dtype()' part
    surrounding the object. This object may be a string, a list, or
    a dict depending on the nature of the dtype. This
    is the object passed as the first parameter to the dtype
    constructor, and if no additional constructor parameters are
    given, will reproduce the exact memory layout.

    Parameters
    ----------
    short : bool
        If true, this creates a shorter repr using 'kind' and 'itemsize', instead
        of the longer type name.

    include_align : bool
        If true, this includes the 'align=True' parameter
        inside the struct dtype construction dict when needed. Use this flag
        if you want a proper repr string without the 'dtype()' part around it.

        If false, this does not preserve the
        'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for
        struct arrays like the regular repr does, because the 'align'
        flag is not part of first dtype constructor parameter. This
        mode is intended for a full 'repr', where the 'align=True' is
        provided as the second parameter.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._construction_repr', '_construction_repr(dtype, include_align=False, short=False)', {'_struct_str': _struct_str, '_subarray_str': _subarray_str, '_scalar_str': _scalar_str, 'dtype': dtype, 'include_align': include_align, 'short': short}, 1)

def _scalar_str(dtype, short):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._scalar_str', '_scalar_str(dtype, short)', {'_byte_order_str': _byte_order_str, 'np': np, '_isunsized': _isunsized, '_datetime_metadata_str': _datetime_metadata_str, '_kind_name': _kind_name, 'dtype': dtype, 'short': short}, 1)

def _byte_order_str(dtype):
    """ Normalize byteorder to '<' or '>' """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._byte_order_str', '_byte_order_str(dtype)', {'np': np, 'dtype': dtype}, 1)

def _datetime_metadata_str(dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._datetime_metadata_str', '_datetime_metadata_str(dtype)', {'np': np, 'dtype': dtype}, 1)

def _struct_dict_str(dtype, includealignedflag):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._struct_dict_str', '_struct_dict_str(dtype, includealignedflag)', {'_unpack_field': _unpack_field, 'np': np, '_construction_repr': _construction_repr, 'dtype': dtype, 'includealignedflag': includealignedflag}, 1)

def _aligned_offset(offset, alignment):
    return -(-offset // alignment) * alignment

def _is_packed(dtype):
    """
    Checks whether the structured data type in 'dtype'
    has a simple layout, where all the fields are in order,
    and follow each other with no alignment padding.

    When this returns true, the dtype can be reconstructed
    from a list of the field names and dtypes with no additional
    dtype parameters.

    Duplicates the C `is_dtype_struct_simple_unaligned_layout` function.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._is_packed', '_is_packed(dtype)', {'_unpack_field': _unpack_field, '_aligned_offset': _aligned_offset, 'dtype': dtype}, 1)

def _struct_list_str(dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._struct_list_str', '_struct_list_str(dtype)', {'_unpack_field': _unpack_field, '_construction_repr': _construction_repr, 'dtype': dtype}, 1)

def _struct_str(dtype, include_align):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._struct_str', '_struct_str(dtype, include_align)', {'_is_packed': _is_packed, '_struct_list_str': _struct_list_str, '_struct_dict_str': _struct_dict_str, 'np': np, 'dtype': dtype, 'include_align': include_align}, 1)

def _subarray_str(dtype):
    (base, shape) = dtype.subdtype
    return '({}, {})'.format(_construction_repr(base, short=True), shape)

def _name_includes_bit_suffix(dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._name_includes_bit_suffix', '_name_includes_bit_suffix(dtype)', {'np': np, '_isunsized': _isunsized, 'dtype': dtype}, 1)

def _name_get(dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype._name_get', '_name_get(dtype)', {'np': np, '_kind_name': _kind_name, '_name_includes_bit_suffix': _name_includes_bit_suffix, '_datetime_metadata_str': _datetime_metadata_str, 'dtype': dtype}, 1)

