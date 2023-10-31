"""Test the numpy pickler as a replacement of the standard pickler."""

import copy
import os
import random
import re
import io
import sys
import warnings
import gzip
import zlib
import bz2
import pickle
import socket
from contextlib import closing
import mmap
from pathlib import Path
try:
    import lzma
except ImportError:
    lzma = None
import pytest
from joblib.test.common import np, with_numpy, with_lz4, without_lz4
from joblib.test.common import with_memory_profiler, memory_used
from joblib.testing import parametrize, raises, warns
from joblib import numpy_pickle, register_compressor
from joblib.test import data
from joblib.numpy_pickle_utils import _IO_BUFFER_SIZE
from joblib.numpy_pickle_utils import _detect_compressor
from joblib.numpy_pickle_utils import _is_numpy_array_byte_order_mismatch
from joblib.numpy_pickle_utils import _ensure_native_byte_order
from joblib.compressor import _COMPRESSORS, _LZ4_PREFIX, CompressorWrapper, LZ4_NOT_INSTALLED_ERROR, BinaryZlibFile
typelist = []
_none = None
typelist.append(_none)
_type = type
typelist.append(_type)
_bool = bool(1)
typelist.append(_bool)
_int = int(1)
typelist.append(_int)
_float = float(1)
typelist.append(_float)
_complex = complex(1)
typelist.append(_complex)
_string = str(1)
typelist.append(_string)
_tuple = ()
typelist.append(_tuple)
_list = []
typelist.append(_list)
_dict = {}
typelist.append(_dict)
_builtin = len
typelist.append(_builtin)

def _function(x):
    yield x


class _class:
    
    def _method(self):
        pass



class _newclass(object):
    
    def _method(self):
        pass

typelist.append(_function)
typelist.append(_class)
typelist.append(_newclass)
_instance = _class()
typelist.append(_instance)
_object = _newclass()
typelist.append(_object)

@parametrize('compress', [0, 1])
@parametrize('member', typelist)
def test_standard_types(tmpdir, compress, member):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_standard_types', 'test_standard_types(tmpdir, compress, member)', {'numpy_pickle': numpy_pickle, 'copy': copy, 'parametrize': parametrize, 'typelist': typelist, 'tmpdir': tmpdir, 'compress': compress, 'member': member}, 0)

def test_value_error():
    with raises(ValueError):
        numpy_pickle.dump('foo', dict())

@parametrize('wrong_compress', [-1, 10, dict()])
def test_compress_level_error(wrong_compress):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_compress_level_error', 'test_compress_level_error(wrong_compress)', {'raises': raises, 'numpy_pickle': numpy_pickle, 'parametrize': parametrize, 'dict': dict, 'wrong_compress': wrong_compress}, 0)

@with_numpy
@parametrize('compress', [False, True, 0, 3, 'zlib'])
def test_numpy_persistence(tmpdir, compress):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_numpy_persistence', 'test_numpy_persistence(tmpdir, compress)', {'np': np, 'numpy_pickle': numpy_pickle, 'os': os, 'ComplexTestObject': ComplexTestObject, 'with_numpy': with_numpy, 'parametrize': parametrize, 'tmpdir': tmpdir, 'compress': compress}, 0)

@with_numpy
def test_numpy_persistence_bufferred_array_compression(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_numpy_persistence_bufferred_array_compression', 'test_numpy_persistence_bufferred_array_compression(tmpdir)', {'np': np, '_IO_BUFFER_SIZE': _IO_BUFFER_SIZE, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_memmap_persistence(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_memmap_persistence', 'test_memmap_persistence(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'ComplexTestObject': ComplexTestObject, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_memmap_persistence_mixed_dtypes(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_memmap_persistence_mixed_dtypes', 'test_memmap_persistence_mixed_dtypes(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_masked_array_persistence(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_masked_array_persistence', 'test_masked_array_persistence(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_compress_mmap_mode_warning(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_compress_mmap_mode_warning', 'test_compress_mmap_mode_warning(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'warns': warns, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
@parametrize('cache_size', [None, 0, 10])
def test_cache_size_warning(tmpdir, cache_size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_cache_size_warning', 'test_cache_size_warning(tmpdir, cache_size)', {'np': np, 'warnings': warnings, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'parametrize': parametrize, 'tmpdir': tmpdir, 'cache_size': cache_size}, 0)

@with_numpy
@with_memory_profiler
@parametrize('compress', [True, False])
def test_memory_usage(tmpdir, compress):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_memory_usage', 'test_memory_usage(tmpdir, compress)', {'np': np, 'memory_used': memory_used, 'numpy_pickle': numpy_pickle, '_IO_BUFFER_SIZE': _IO_BUFFER_SIZE, 'with_numpy': with_numpy, 'with_memory_profiler': with_memory_profiler, 'parametrize': parametrize, 'tmpdir': tmpdir, 'compress': compress}, 0)

@with_numpy
def test_compressed_pickle_dump_and_load(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_compressed_pickle_dump_and_load', 'test_compressed_pickle_dump_and_load(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, '_ensure_native_byte_order': _ensure_native_byte_order, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

def _check_pickle(filename, expected_list, mmap_mode=None):
    """Helper function to test joblib pickle content.

    Note: currently only pickles containing an iterable are supported
    by this function.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle._check_pickle', '_check_pickle(filename, expected_list, mmap_mode=None)', {'re': re, 'warnings': warnings, 'numpy_pickle': numpy_pickle, 'os': os, 'np': np, '_ensure_native_byte_order': _ensure_native_byte_order, 'with_lz4': with_lz4, 'LZ4_NOT_INSTALLED_ERROR': LZ4_NOT_INSTALLED_ERROR, 'filename': filename, 'expected_list': expected_list, 'mmap_mode': mmap_mode}, 0)

@with_numpy
def test_joblib_pickle_across_python_versions():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_joblib_pickle_across_python_versions', 'test_joblib_pickle_across_python_versions()', {'np': np, 'os': os, 'data': data, 'lzma': lzma, '_check_pickle': _check_pickle, 'with_numpy': with_numpy}, 0)

@with_numpy
def test_joblib_pickle_across_python_versions_with_mmap():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_joblib_pickle_across_python_versions_with_mmap', 'test_joblib_pickle_across_python_versions_with_mmap()', {'np': np, 'os': os, 'data': data, '_check_pickle': _check_pickle, 'with_numpy': with_numpy}, 0)

@with_numpy
def test_numpy_array_byte_order_mismatch_detection():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_numpy_array_byte_order_mismatch_detection', 'test_numpy_array_byte_order_mismatch_detection()', {'np': np, 'sys': sys, '_is_numpy_array_byte_order_mismatch': _is_numpy_array_byte_order_mismatch, '_ensure_native_byte_order': _ensure_native_byte_order, 'with_numpy': with_numpy}, 0)

@parametrize('compress_tuple', [('zlib', 3), ('gzip', 3)])
def test_compress_tuple_argument(tmpdir, compress_tuple):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_compress_tuple_argument', 'test_compress_tuple_argument(tmpdir, compress_tuple)', {'numpy_pickle': numpy_pickle, '_detect_compressor': _detect_compressor, 'parametrize': parametrize, 'tmpdir': tmpdir, 'compress_tuple': compress_tuple}, 0)

@parametrize('compress_tuple,message', [(('zlib', 3, 'extra'), 'Compress argument tuple should contain exactly 2 elements'), (('wrong', 3), 'Non valid compression method given: "{}"'.format('wrong')), (('zlib', 'wrong'), 'Non valid compress level given: "{}"'.format('wrong'))])
def test_compress_tuple_argument_exception(tmpdir, compress_tuple, message):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_compress_tuple_argument_exception', 'test_compress_tuple_argument_exception(tmpdir, compress_tuple, message)', {'raises': raises, 'numpy_pickle': numpy_pickle, 'parametrize': parametrize, 'tmpdir': tmpdir, 'compress_tuple': compress_tuple, 'message': message}, 0)

@parametrize('compress_string', ['zlib', 'gzip'])
def test_compress_string_argument(tmpdir, compress_string):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_compress_string_argument', 'test_compress_string_argument(tmpdir, compress_string)', {'numpy_pickle': numpy_pickle, '_detect_compressor': _detect_compressor, 'parametrize': parametrize, 'tmpdir': tmpdir, 'compress_string': compress_string}, 0)

@with_numpy
@parametrize('compress', [1, 3, 6])
@parametrize('cmethod', _COMPRESSORS)
def test_joblib_compression_formats(tmpdir, compress, cmethod):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_joblib_compression_formats', 'test_joblib_compression_formats(tmpdir, compress, cmethod)', {'np': np, 'lzma': lzma, 'pytest': pytest, 'with_lz4': with_lz4, 'numpy_pickle': numpy_pickle, '_detect_compressor': _detect_compressor, 'with_numpy': with_numpy, 'parametrize': parametrize, '_COMPRESSORS': _COMPRESSORS, 'tmpdir': tmpdir, 'compress': compress, 'cmethod': cmethod}, 0)

def _gzip_file_decompress(source_filename, target_filename):
    """Decompress a gzip file."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle._gzip_file_decompress', '_gzip_file_decompress(source_filename, target_filename)', {'closing': closing, 'gzip': gzip, 'source_filename': source_filename, 'target_filename': target_filename}, 0)

def _zlib_file_decompress(source_filename, target_filename):
    """Decompress a zlib file."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle._zlib_file_decompress', '_zlib_file_decompress(source_filename, target_filename)', {'zlib': zlib, 'source_filename': source_filename, 'target_filename': target_filename}, 0)

@parametrize('extension,decompress', [('.z', _zlib_file_decompress), ('.gz', _gzip_file_decompress)])
def test_load_externally_decompressed_files(tmpdir, extension, decompress):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_load_externally_decompressed_files', 'test_load_externally_decompressed_files(tmpdir, extension, decompress)', {'numpy_pickle': numpy_pickle, 'parametrize': parametrize, '_zlib_file_decompress': _zlib_file_decompress, '_gzip_file_decompress': _gzip_file_decompress, 'tmpdir': tmpdir, 'extension': extension, 'decompress': decompress}, 0)

@parametrize('extension,cmethod', [('.z', 'zlib'), ('.gz', 'gzip'), ('.bz2', 'bz2'), ('.lzma', 'lzma'), ('.xz', 'xz'), ('.pkl', 'not-compressed'), ('', 'not-compressed')])
def test_compression_using_file_extension(tmpdir, extension, cmethod):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_compression_using_file_extension', 'test_compression_using_file_extension(tmpdir, extension, cmethod)', {'lzma': lzma, 'pytest': pytest, 'numpy_pickle': numpy_pickle, '_detect_compressor': _detect_compressor, 'parametrize': parametrize, 'tmpdir': tmpdir, 'extension': extension, 'cmethod': cmethod}, 0)

@with_numpy
def test_file_handle_persistence(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_file_handle_persistence', 'test_file_handle_persistence(tmpdir)', {'np': np, 'bz2': bz2, 'gzip': gzip, 'lzma': lzma, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_in_memory_persistence():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_in_memory_persistence', 'test_in_memory_persistence()', {'np': np, 'io': io, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy}, 0)

@with_numpy
def test_file_handle_persistence_mmap(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_file_handle_persistence_mmap', 'test_file_handle_persistence_mmap(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_file_handle_persistence_compressed_mmap(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_file_handle_persistence_compressed_mmap', 'test_file_handle_persistence_compressed_mmap(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'closing': closing, 'gzip': gzip, 'warns': warns, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_file_handle_persistence_in_memory_mmap():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_file_handle_persistence_in_memory_mmap', 'test_file_handle_persistence_in_memory_mmap()', {'np': np, 'io': io, 'numpy_pickle': numpy_pickle, 'warns': warns, 'with_numpy': with_numpy}, 0)

@parametrize('data', [b'a little data as bytes.', 10000 * '{}'.format(random.randint(0, 1000) * 1000).encode('latin-1')], ids=['a little data as bytes.', 'a large data as bytes.'])
@parametrize('compress_level', [1, 3, 9])
def test_binary_zlibfile(tmpdir, data, compress_level):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_binary_zlibfile', 'test_binary_zlibfile(tmpdir, data, compress_level)', {'BinaryZlibFile': BinaryZlibFile, 'raises': raises, 'io': io, 'parametrize': parametrize, 'random': random, 'tmpdir': tmpdir, 'data': data, 'compress_level': compress_level}, 0)

@parametrize('bad_value', [-1, 10, 15, 'a', (), {}])
def test_binary_zlibfile_bad_compression_levels(tmpdir, bad_value):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_binary_zlibfile_bad_compression_levels', 'test_binary_zlibfile_bad_compression_levels(tmpdir, bad_value)', {'raises': raises, 'BinaryZlibFile': BinaryZlibFile, 're': re, 'parametrize': parametrize, 'tmpdir': tmpdir, 'bad_value': bad_value}, 0)

@parametrize('bad_mode', ['a', 'x', 'r', 'w', 1, 2])
def test_binary_zlibfile_invalid_modes(tmpdir, bad_mode):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_binary_zlibfile_invalid_modes', 'test_binary_zlibfile_invalid_modes(tmpdir, bad_mode)', {'raises': raises, 'BinaryZlibFile': BinaryZlibFile, 'parametrize': parametrize, 'tmpdir': tmpdir, 'bad_mode': bad_mode}, 0)

@parametrize('bad_file', [1, (), {}])
def test_binary_zlibfile_invalid_filename_type(bad_file):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_binary_zlibfile_invalid_filename_type', 'test_binary_zlibfile_invalid_filename_type(bad_file)', {'raises': raises, 'BinaryZlibFile': BinaryZlibFile, 'parametrize': parametrize, 'bad_file': bad_file}, 0)
if np is not None:
    
    
    class SubArray(np.ndarray):
        
        def __reduce__(self):
            return (_load_sub_array, (np.asarray(self), ))
    
    
    def _load_sub_array(arr):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle._load_sub_array', '_load_sub_array(arr)', {'SubArray': SubArray, 'arr': arr}, 1)
    
    
    class ComplexTestObject:
        """A complex object containing numpy arrays as attributes."""
        
        def __init__(self):
            self.array_float = np.arange(100, dtype='float64')
            self.array_int = np.ones(100, dtype='int32')
            self.array_obj = np.array(['a', 10, 20.0], dtype='object')
    

@with_numpy
def test_numpy_subclass(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_numpy_subclass', 'test_numpy_subclass(tmpdir)', {'SubArray': SubArray, 'numpy_pickle': numpy_pickle, 'np': np, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

def test_pathlib(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_pathlib', 'test_pathlib(tmpdir)', {'numpy_pickle': numpy_pickle, 'Path': Path, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_non_contiguous_array_pickling(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_non_contiguous_array_pickling', 'test_non_contiguous_array_pickling(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_pickle_highest_protocol(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_pickle_highest_protocol', 'test_pickle_highest_protocol(tmpdir)', {'np': np, 'numpy_pickle': numpy_pickle, 'pickle': pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_pickle_in_socket():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_pickle_in_socket', 'test_pickle_in_socket()', {'np': np, 'socket': socket, 'numpy_pickle': numpy_pickle, 'io': io, 'with_numpy': with_numpy}, 0)

@with_numpy
def test_load_memmap_with_big_offset(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_load_memmap_with_big_offset', 'test_load_memmap_with_big_offset(tmpdir)', {'mmap': mmap, 'np': np, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

def test_register_compressor(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_register_compressor', 'test_register_compressor(tmpdir)', {'io': io, 'CompressorWrapper': CompressorWrapper, 'register_compressor': register_compressor, '_COMPRESSORS': _COMPRESSORS, 'tmpdir': tmpdir}, 0)

@parametrize('invalid_name', [1, (), {}])
def test_register_compressor_invalid_name(invalid_name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_register_compressor_invalid_name', 'test_register_compressor_invalid_name(invalid_name)', {'raises': raises, 'register_compressor': register_compressor, 'parametrize': parametrize, 'invalid_name': invalid_name}, 0)

def test_register_compressor_invalid_fileobj():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_register_compressor_invalid_fileobj', 'test_register_compressor_invalid_fileobj()', {'CompressorWrapper': CompressorWrapper, 'raises': raises, 'register_compressor': register_compressor}, 0)


class AnotherZlibCompressorWrapper(CompressorWrapper):
    
    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryZlibFile, prefix=b'prefix')



class StandardLibGzipCompressorWrapper(CompressorWrapper):
    
    def __init__(self):
        CompressorWrapper.__init__(self, obj=gzip.GzipFile, prefix=b'prefix')


def test_register_compressor_already_registered():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_register_compressor_already_registered', 'test_register_compressor_already_registered()', {'register_compressor': register_compressor, 'AnotherZlibCompressorWrapper': AnotherZlibCompressorWrapper, 'raises': raises, 'StandardLibGzipCompressorWrapper': StandardLibGzipCompressorWrapper, '_COMPRESSORS': _COMPRESSORS, 'gzip': gzip}, 0)

@with_lz4
def test_lz4_compression(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_lz4_compression', 'test_lz4_compression(tmpdir)', {'_COMPRESSORS': _COMPRESSORS, 'lz4': lz4, 'numpy_pickle': numpy_pickle, '_LZ4_PREFIX': _LZ4_PREFIX, 'with_lz4': with_lz4, 'tmpdir': tmpdir}, 0)

@without_lz4
def test_lz4_compression_without_lz4(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_lz4_compression_without_lz4', 'test_lz4_compression_without_lz4(tmpdir)', {'LZ4_NOT_INSTALLED_ERROR': LZ4_NOT_INSTALLED_ERROR, 'raises': raises, 'numpy_pickle': numpy_pickle, 'without_lz4': without_lz4, 'tmpdir': tmpdir}, 0)
protocols = [pickle.DEFAULT_PROTOCOL]
if pickle.HIGHEST_PROTOCOL != pickle.DEFAULT_PROTOCOL:
    protocols.append(pickle.HIGHEST_PROTOCOL)

@with_numpy
@parametrize('protocol', protocols)
def test_memmap_alignment_padding(tmpdir, protocol):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle.test_memmap_alignment_padding', 'test_memmap_alignment_padding(tmpdir, protocol)', {'np': np, 'numpy_pickle': numpy_pickle, 'with_numpy': with_numpy, 'parametrize': parametrize, 'protocols': protocols, 'tmpdir': tmpdir, 'protocol': protocol}, 0)

