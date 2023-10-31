"""
Fast cryptographic hash of Python objects, with a special case for fast
hashing of numpy arrays.
"""

import pickle
import hashlib
import sys
import types
import struct
import io
import decimal
Pickler = pickle._Pickler


class _ConsistentSet(object):
    """ Class used to ensure the hash of Sets is preserved
        whatever the order of its items.
    """
    
    def __init__(self, set_sequence):
        try:
            self._sequence = sorted(set_sequence)
        except (TypeError, decimal.InvalidOperation):
            self._sequence = sorted((hash(e) for e in set_sequence))



class _MyHash(object):
    """ Class used to hash objects that won't normally pickle """
    
    def __init__(self, *args):
        self.args = args



class Hasher(Pickler):
    """ A subclass of pickler, to do cryptographic hashing, rather than
        pickling.
    """
    
    def __init__(self, hash_name='md5'):
        self.stream = io.BytesIO()
        protocol = 3
        Pickler.__init__(self, self.stream, protocol=protocol)
        self._hash = hashlib.new(hash_name)
    
    def hash(self, obj, return_digest=True):
        try:
            self.dump(obj)
        except pickle.PicklingError as e:
            e.args += ('PicklingError while hashing %r: %r' % (obj, e), )
            raise
        dumps = self.stream.getvalue()
        self._hash.update(dumps)
        if return_digest:
            return self._hash.hexdigest()
    
    def save(self, obj):
        if isinstance(obj, (types.MethodType, type({}.pop))):
            if hasattr(obj, '__func__'):
                func_name = obj.__func__.__name__
            else:
                func_name = obj.__name__
            inst = obj.__self__
            if type(inst) is type(pickle):
                obj = _MyHash(func_name, inst.__name__)
            elif inst is None:
                obj = _MyHash(func_name, inst)
            else:
                cls = obj.__self__.__class__
                obj = _MyHash(func_name, inst, cls)
        Pickler.save(self, obj)
    
    def memoize(self, obj):
        if isinstance(obj, (bytes, str)):
            return
        Pickler.memoize(self, obj)
    
    def save_global(self, obj, name=None, pack=struct.pack):
        kwargs = dict(name=name, pack=pack)
        del kwargs['pack']
        try:
            Pickler.save_global(self, obj, **kwargs)
        except pickle.PicklingError:
            Pickler.save_global(self, obj, **kwargs)
            module = getattr(obj, '__module__', None)
            if module == '__main__':
                my_name = name
                if my_name is None:
                    my_name = obj.__name__
                mod = sys.modules[module]
                if not hasattr(mod, my_name):
                    setattr(mod, my_name, obj)
    dispatch = Pickler.dispatch.copy()
    dispatch[type(len)] = save_global
    dispatch[type(object)] = save_global
    dispatch[type(Pickler)] = save_global
    dispatch[type(pickle.dump)] = save_global
    
    def _batch_setitems(self, items):
        try:
            Pickler._batch_setitems(self, iter(sorted(items)))
        except TypeError:
            Pickler._batch_setitems(self, iter(sorted(((hash(k), v) for (k, v) in items))))
    
    def save_set(self, set_items):
        Pickler.save(self, _ConsistentSet(set_items))
    dispatch[type(set())] = save_set



class NumpyHasher(Hasher):
    """ Special case the hasher for when numpy is loaded.
    """
    
    def __init__(self, hash_name='md5', coerce_mmap=False):
        """
            Parameters
            ----------
            hash_name: string
                The hash algorithm to be used
            coerce_mmap: boolean
                Make no difference between np.memmap and np.ndarray
                objects.
        """
        self.coerce_mmap = coerce_mmap
        Hasher.__init__(self, hash_name=hash_name)
        import numpy as np
        self.np = np
        if hasattr(np, 'getbuffer'):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview
    
    def save(self, obj):
        """ Subclass the save method, to hash ndarray subclass, rather
            than pickling them. Off course, this is a total abuse of
            the Pickler class.
        """
        if (isinstance(obj, self.np.ndarray) and not obj.dtype.hasobject):
            if obj.shape == ():
                obj_c_contiguous = obj.flatten()
            elif obj.flags.c_contiguous:
                obj_c_contiguous = obj
            elif obj.flags.f_contiguous:
                obj_c_contiguous = obj.T
            else:
                obj_c_contiguous = obj.flatten()
            self._hash.update(self._getbuffer(obj_c_contiguous.view(self.np.uint8)))
            if (self.coerce_mmap and isinstance(obj, self.np.memmap)):
                klass = self.np.ndarray
            else:
                klass = obj.__class__
            obj = (klass, ('HASHED', obj.dtype, obj.shape, obj.strides))
        elif isinstance(obj, self.np.dtype):
            self._hash.update('_HASHED_DTYPE'.encode('utf-8'))
            self._hash.update(pickle.dumps(obj))
            return
        Hasher.save(self, obj)


def hash(obj, hash_name='md5', coerce_mmap=False):
    """ Quick calculation of a hash to identify uniquely Python objects
        containing numpy arrays.

        Parameters
        ----------
        hash_name: 'md5' or 'sha1'
            Hashing algorithm used. sha1 is supposedly safer, but md5 is
            faster.
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.hashing.hash', "hash(obj, hash_name='md5', coerce_mmap=False)", {'sys': sys, 'NumpyHasher': NumpyHasher, 'Hasher': Hasher, 'obj': obj, 'hash_name': hash_name, 'coerce_mmap': coerce_mmap}, 1)

