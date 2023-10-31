import numpy as np
import numba as nb
from numpy.random import PCG64
from timeit import timeit
bit_gen = PCG64()
next_d = bit_gen.cffi.next_double
state_addr = bit_gen.cffi.state_address

def normals(n, state):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.random._examples.numba.extending.normals', 'normals(n, state)', {'np': np, 'next_d': next_d, 'n': n, 'state': state}, 1)
normalsj = nb.jit(normals, nopython=True)
n = 10000

def numbacall():
    return normalsj(n, state_addr)
rg = np.random.Generator(PCG64())

def numpycall():
    return rg.normal(size=n)
r1 = numbacall()
r2 = numpycall()
assert r1.shape == (n, )
assert r1.shape == r2.shape
t1 = timeit(numbacall, number=1000)
print(f'{t1:.2f} secs for {n} PCG64 (Numba/PCG64) gaussian randoms')
t2 = timeit(numpycall, number=1000)
print(f'{t2:.2f} secs for {n} PCG64 (NumPy/PCG64) gaussian randoms')
next_u32 = bit_gen.ctypes.next_uint32
ctypes_state = bit_gen.ctypes.state

@nb.jit(nopython=True)
def bounded_uint(lb, ub, state):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.random._examples.numba.extending.bounded_uint', 'bounded_uint(lb, ub, state)', {'next_u32': next_u32, 'nb': nb, 'lb': lb, 'ub': ub, 'state': state}, 1)
print(bounded_uint(323, 2394691, ctypes_state.value))

@nb.jit(nopython=True)
def bounded_uints(lb, ub, n, state):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.random._examples.numba.extending.bounded_uints', 'bounded_uints(lb, ub, n, state)', {'np': np, 'bounded_uint': bounded_uint, 'nb': nb, 'lb': lb, 'ub': ub, 'n': n, 'state': state}, 0)
bounded_uints(323, 2394691, 10000000, ctypes_state.value)

