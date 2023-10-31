"""Utilities for fast persistence of big data, with optional compression."""

import pickle
import os
import warnings
import io
from pathlib import Path
from .compressor import lz4, LZ4_NOT_INSTALLED_ERROR
from .compressor import _COMPRESSORS, register_compressor, BinaryZlibFile
from .compressor import ZlibCompressorWrapper, GzipCompressorWrapper, BZ2CompressorWrapper, LZMACompressorWrapper, XZCompressorWrapper, LZ4CompressorWrapper
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import _read_fileobject, _write_fileobject
from .numpy_pickle_utils import _read_bytes, BUFFER_SIZE
from .numpy_pickle_utils import _ensure_native_byte_order
from .numpy_pickle_compat import load_compatibility
from .numpy_pickle_compat import NDArrayWrapper
from .numpy_pickle_compat import ZNDArrayWrapper
from .backports import make_memmap
register_compressor('zlib', ZlibCompressorWrapper())
register_compressor('gzip', GzipCompressorWrapper())
register_compressor('bz2', BZ2CompressorWrapper())
register_compressor('lzma', LZMACompressorWrapper())
register_compressor('xz', XZCompressorWrapper())
register_compressor('lz4', LZ4CompressorWrapper())
NUMPY_ARRAY_ALIGNMENT_BYTES = 16


class NumpyArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    This object is used to hack into the pickle machinery and read numpy
    array data from our custom persistence format.
    More precisely, this object is used for:
    * carrying the information of the persisted array: subclass, shape, order,
    dtype. Those ndarray metadata are used to correctly reconstruct the array
    with low level numpy functions.
    * determining if memmap is allowed on the array.
    * reading the array bytes from a file.
    * reading the array using memorymap from a file.
    * writing the array bytes to a file.

    Attributes
    ----------
    subclass: numpy.ndarray subclass
        Determine the subclass of the wrapped array.
    shape: numpy.ndarray shape
        Determine the shape of the wrapped array.
    order: {'C', 'F'}
        Determine the order of wrapped array data. 'C' is for C order, 'F' is
        for fortran order.
    dtype: numpy.ndarray dtype
        Determine the data type of the wrapped array.
    allow_mmap: bool
        Determine if memory mapping is allowed on the wrapped array.
        Default: False.
    """
    
    def __init__(self, subclass, shape, order, dtype, allow_mmap=False, numpy_array_alignment_bytes=NUMPY_ARRAY_ALIGNMENT_BYTES):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.shape = shape
        self.order = order
        self.dtype = dtype
        self.allow_mmap = allow_mmap
        self.numpy_array_alignment_bytes = numpy_array_alignment_bytes
    
    def safe_get_numpy_array_alignment_bytes(self):
        return getattr(self, 'numpy_array_alignment_bytes', None)
    
    def write_array(self, array, pickler):
        """Write array bytes to pickler file handle.

        This function is an adaptation of the numpy write_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        buffersize = max(16 * 1024**2 // array.itemsize, 1)
        if array.dtype.hasobject:
            pickle.dump(array, pickler.file_handle, protocol=2)
        else:
            numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
            if numpy_array_alignment_bytes is not None:
                current_pos = pickler.file_handle.tell()
                pos_after_padding_byte = current_pos + 1
                padding_length = numpy_array_alignment_bytes - pos_after_padding_byte % numpy_array_alignment_bytes
                padding_length_byte = int.to_bytes(padding_length, length=1, byteorder='little')
                pickler.file_handle.write(padding_length_byte)
                if padding_length != 0:
                    padding = b'\xff' * padding_length
                    pickler.file_handle.write(padding)
            for chunk in pickler.np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'], buffersize=buffersize, order=self.order):
                pickler.file_handle.write(chunk.tobytes('C'))
    
    def read_array(self, unpickler):
        """Read array from unpickler file handle.

        This function is an adaptation of the numpy read_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        if len(self.shape) == 0:
            count = 1
        else:
            shape_int64 = [unpickler.np.int64(x) for x in self.shape]
            count = unpickler.np.multiply.reduce(shape_int64)
        if self.dtype.hasobject:
            array = pickle.load(unpickler.file_handle)
        else:
            numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
            if numpy_array_alignment_bytes is not None:
                padding_byte = unpickler.file_handle.read(1)
                padding_length = int.from_bytes(padding_byte, byteorder='little')
                if padding_length != 0:
                    unpickler.file_handle.read(padding_length)
            max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, self.dtype.itemsize)
            array = unpickler.np.empty(count, dtype=self.dtype)
            for i in range(0, count, max_read_count):
                read_count = min(max_read_count, count - i)
                read_size = int(read_count * self.dtype.itemsize)
                data = _read_bytes(unpickler.file_handle, read_size, 'array data')
                array[i:i + read_count] = unpickler.np.frombuffer(data, dtype=self.dtype, count=read_count)
                del data
            if self.order == 'F':
                array.shape = self.shape[::-1]
                array = array.transpose()
            else:
                array.shape = self.shape
        return _ensure_native_byte_order(array)
    
    def read_mmap(self, unpickler):
        """Read an array using numpy memmap."""
        current_pos = unpickler.file_handle.tell()
        offset = current_pos
        numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
        if numpy_array_alignment_bytes is not None:
            padding_byte = unpickler.file_handle.read(1)
            padding_length = int.from_bytes(padding_byte, byteorder='little')
            offset += padding_length + 1
        if unpickler.mmap_mode == 'w+':
            unpickler.mmap_mode = 'r+'
        marray = make_memmap(unpickler.filename, dtype=self.dtype, shape=self.shape, order=self.order, mode=unpickler.mmap_mode, offset=offset)
        unpickler.file_handle.seek(offset + marray.nbytes)
        if (numpy_array_alignment_bytes is None and current_pos % NUMPY_ARRAY_ALIGNMENT_BYTES != 0):
            message = f'The memmapped array {marray} loaded from the file {unpickler.file_handle.name} is not byte aligned. This may cause segmentation faults if this memmapped array is used in some libraries like BLAS or PyTorch. To get rid of this warning, regenerate your pickle file with joblib >= 1.2.0. See https://github.com/joblib/joblib/issues/563 for more details'
            warnings.warn(message)
        return _ensure_native_byte_order(marray)
    
    def read(self, unpickler):
        """Read the array corresponding to this wrapper.

        Use the unpickler to get all information to correctly read the array.

        Parameters
        ----------
        unpickler: NumpyUnpickler

        Returns
        -------
        array: numpy.ndarray

        """
        if (unpickler.mmap_mode is not None and self.allow_mmap):
            array = self.read_mmap(unpickler)
        else:
            array = self.read_array(unpickler)
        if (hasattr(array, '__array_prepare__') and self.subclass not in (unpickler.np.ndarray, unpickler.np.memmap)):
            new_array = unpickler.np.core.multiarray._reconstruct(self.subclass, (0, ), 'b')
            return new_array.__array_prepare__(array)
        else:
            return array



class NumpyPickler(Pickler):
    """A pickler to persist big data efficiently.

    The main features of this object are:
    * persistence of numpy arrays in a single file.
    * optional compression with a special care on avoiding memory copies.

    Attributes
    ----------
    fp: file
        File object handle used for serializing the input object.
    protocol: int, optional
        Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL.
    """
    dispatch = Pickler.dispatch.copy()
    
    def __init__(self, fp, protocol=None):
        self.file_handle = fp
        self.buffered = isinstance(self.file_handle, BinaryZlibFile)
        if protocol is None:
            protocol = pickle.DEFAULT_PROTOCOL
        Pickler.__init__(self, self.file_handle, protocol=protocol)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np
    
    def _create_array_wrapper(self, array):
        """Create and returns a numpy array wrapper from a numpy array."""
        order = ('F' if (array.flags.f_contiguous and not array.flags.c_contiguous) else 'C')
        allow_mmap = (not self.buffered and not array.dtype.hasobject)
        kwargs = {}
        try:
            self.file_handle.tell()
        except io.UnsupportedOperation:
            kwargs = {'numpy_array_alignment_bytes': None}
        wrapper = NumpyArrayWrapper(type(array), array.shape, order, array.dtype, allow_mmap=allow_mmap, **kwargs)
        return wrapper
    
    def save(self, obj):
        """Subclass the Pickler `save` method.

        This is a total abuse of the Pickler class in order to use the numpy
        persistence function `save` instead of the default pickle
        implementation. The numpy array is replaced by a custom wrapper in the
        pickle persistence stack and the serialized array is written right
        after in the file. Warning: the file produced does not follow the
        pickle format. As such it can not be read with `pickle.load`.
        """
        if (self.np is not None and type(obj) in (self.np.ndarray, self.np.matrix, self.np.memmap)):
            if type(obj) is self.np.memmap:
                obj = self.np.asanyarray(obj)
            wrapper = self._create_array_wrapper(obj)
            Pickler.save(self, wrapper)
            if self.proto >= 4:
                self.framer.commit_frame(force=True)
            wrapper.write_array(obj, self)
            return
        return Pickler.save(self, obj)



class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.

    Attributes
    ----------
    mmap_mode: str
        The memorymap mode to use for reading numpy arrays.
    file_handle: file_like
        File object to unpickle from.
    filename: str
        Name of the file to unpickle from. It should correspond to file_handle.
        This parameter is required when using mmap_mode.
    np: module
        Reference to numpy module if numpy is installed else None.

    """
    dispatch = Unpickler.dispatch.copy()
    
    def __init__(self, filename, file_handle, mmap_mode=None):
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = file_handle
        self.filename = filename
        self.compat_mode = False
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np
    
    def load_build(self):
        """Called to set the state of a newly created object.

        We capture it to replace our place-holder objects, NDArrayWrapper or
        NumpyArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        NDArrayWrapper is used for backward compatibility with joblib <= 0.9.
        """
        Unpickler.load_build(self)
        if isinstance(self.stack[-1], (NDArrayWrapper, NumpyArrayWrapper)):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, but numpy didn't import correctly")
            array_wrapper = self.stack.pop()
            if isinstance(array_wrapper, NDArrayWrapper):
                self.compat_mode = True
            self.stack.append(array_wrapper.read(self))
    dispatch[pickle.BUILD[0]] = load_build


def dump(value, filename, compress=0, protocol=None, cache_size=None):
    """Persist an arbitrary Python object into one file.

    Read more in the :ref:`User Guide <persistence>`.

    Parameters
    ----------
    value: any Python object
        The object to store to disk.
    filename: str, pathlib.Path, or file object.
        The file object or path of the file in which it is to be stored.
        The compression method corresponding to one of the supported filename
        extensions ('.z', '.gz', '.bz2', '.xz' or '.lzma') will be used
        automatically.
    compress: int from 0 to 9 or bool or 2-tuple, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher value means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
        If compress is True, the compression level used is 3.
        If compress is a 2-tuple, the first element must correspond to a string
        between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
        'xz'), the second element must be an integer from 0 to 9, corresponding
        to the compression level.
    protocol: int, optional
        Pickle protocol, see pickle.dump documentation for more details.
    cache_size: positive int, optional
        This option is deprecated in 0.10 and has no effect.

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    See Also
    --------
    joblib.load : corresponding loader

    Notes
    -----
    Memmapping on load cannot be used for compressed files. Thus
    using compression can significantly slow down loading. In
    addition, compressed files take up extra memory during
    dump and load.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.numpy_pickle.dump', 'dump(value, filename, compress=0, protocol=None, cache_size=None)', {'Path': Path, 'lz4': lz4, 'LZ4_NOT_INSTALLED_ERROR': LZ4_NOT_INSTALLED_ERROR, '_COMPRESSORS': _COMPRESSORS, 'warnings': warnings, '_write_fileobject': _write_fileobject, 'NumpyPickler': NumpyPickler, 'value': value, 'filename': filename, 'compress': compress, 'protocol': protocol, 'cache_size': cache_size}, 1)

def _unpickle(fobj, filename='', mmap_mode=None):
    """Internal unpickling function."""
    unpickler = NumpyUnpickler(filename, fobj, mmap_mode=mmap_mode)
    obj = None
    try:
        obj = unpickler.load()
        if unpickler.compat_mode:
            warnings.warn("The file '%s' has been generated with a joblib version less than 0.10. Please regenerate this pickle file." % filename, DeprecationWarning, stacklevel=3)
    except UnicodeDecodeError as exc:
        new_exc = ValueError('You may be trying to read with python 3 a joblib pickle generated with python 2. This feature is not supported by joblib.')
        new_exc.__cause__ = exc
        raise new_exc
    return obj

def load_temporary_memmap(filename, mmap_mode, unlink_on_gc_collect):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.numpy_pickle.load_temporary_memmap', 'load_temporary_memmap(filename, mmap_mode, unlink_on_gc_collect)', {'load': load, 'filename': filename, 'mmap_mode': mmap_mode, 'unlink_on_gc_collect': unlink_on_gc_collect}, 1)

def load(filename, mmap_mode=None):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Read more in the :ref:`User Guide <persistence>`.

    WARNING: joblib.load relies on the pickle module and can therefore
    execute arbitrary Python code. It should therefore never be used
    to load files from untrusted sources.

    Parameters
    ----------
    filename: str, pathlib.Path, or file object.
        The file object or path of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has no effect for compressed files. Note that in this
        case the reconstructed object might no longer match exactly
        the originally pickled object.

    Returns
    -------
    result: any Python object
        The object stored in the file.

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump. If the mmap_mode argument is given, it is passed to np.load and
    arrays are loaded as memmaps. As a consequence, the reconstructed
    object might not match the original pickled object. Note that if the
    file was saved with compression, the arrays cannot be memmapped.
    """
    if (Path is not None and isinstance(filename, Path)):
        filename = str(filename)
    if hasattr(filename, 'read'):
        fobj = filename
        filename = getattr(fobj, 'name', '')
        with _read_fileobject(fobj, filename, mmap_mode) as fobj:
            obj = _unpickle(fobj)
    else:
        with open(filename, 'rb') as f:
            with _read_fileobject(f, filename, mmap_mode) as fobj:
                if isinstance(fobj, str):
                    return load_compatibility(fobj)
                obj = _unpickle(fobj, filename, mmap_mode)
    return obj

