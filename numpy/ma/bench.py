import timeit
import numpy
xs = numpy.random.uniform(-1, 1, 6).reshape(2, 3)
ys = numpy.random.uniform(-1, 1, 6).reshape(2, 3)
zs = xs + 1j * ys
m1 = [[True, False, False], [False, False, True]]
m2 = [[True, False, True], [False, False, True]]
nmxs = numpy.ma.array(xs, mask=m1)
nmys = numpy.ma.array(ys, mask=m2)
nmzs = numpy.ma.array(zs, mask=m1)
xl = numpy.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
yl = numpy.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
zl = xl + 1j * yl
maskx = xl > 0.8
masky = yl < -0.8
nmxl = numpy.ma.array(xl, mask=maskx)
nmyl = numpy.ma.array(yl, mask=masky)
nmzl = numpy.ma.array(zl, mask=maskx)

def timer(s, v='', nloop=500, nrep=3):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.ma.bench.timer', "timer(s, v='', nloop=500, nrep=3)", {'timeit': timeit, 'numpy': numpy, 's': s, 'v': v, 'nloop': nloop, 'nrep': nrep}, 0)

def compare_functions_1v(func, nloop=500, xs=xs, nmxs=nmxs, xl=xl, nmxl=nmxl):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.bench.compare_functions_1v', 'compare_functions_1v(func, nloop=500, xs=xs, nmxs=nmxs, xl=xl, nmxl=nmxl)', {'timer': timer, 'func': func, 'nloop': nloop, 'xs': xs, 'nmxs': nmxs, 'xl': xl, 'nmxl': nmxl, 'xs': xs, 'nmxs': nmxs, 'xl': xl, 'nmxl': nmxl}, 1)

def compare_methods(methodname, args, vars='x', nloop=500, test=True, xs=xs, nmxs=nmxs, xl=xl, nmxl=nmxl):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.bench.compare_methods', "compare_methods(methodname, args, vars='x', nloop=500, test=True, xs=xs, nmxs=nmxs, xl=xl, nmxl=nmxl)", {'timer': timer, 'methodname': methodname, 'args': args, 'vars': vars, 'nloop': nloop, 'test': test, 'xs': xs, 'nmxs': nmxs, 'xl': xl, 'nmxl': nmxl, 'xs': xs, 'nmxs': nmxs, 'xl': xl, 'nmxl': nmxl}, 1)

def compare_functions_2v(func, nloop=500, test=True, xs=xs, nmxs=nmxs, ys=ys, nmys=nmys, xl=xl, nmxl=nmxl, yl=yl, nmyl=nmyl):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.bench.compare_functions_2v', 'compare_functions_2v(func, nloop=500, test=True, xs=xs, nmxs=nmxs, ys=ys, nmys=nmys, xl=xl, nmxl=nmxl, yl=yl, nmyl=nmyl)', {'timer': timer, 'func': func, 'nloop': nloop, 'test': test, 'xs': xs, 'nmxs': nmxs, 'ys': ys, 'nmys': nmys, 'xl': xl, 'nmxl': nmxl, 'yl': yl, 'nmyl': nmyl, 'xs': xs, 'nmxs': nmxs, 'ys': ys, 'nmys': nmys, 'xl': xl, 'nmxl': nmxl, 'yl': yl, 'nmyl': nmyl}, 1)
if __name__ == '__main__':
    compare_functions_1v(numpy.sin)
    compare_functions_1v(numpy.log)
    compare_functions_1v(numpy.sqrt)
    compare_functions_2v(numpy.multiply)
    compare_functions_2v(numpy.divide)
    compare_functions_2v(numpy.power)
    compare_methods('ravel', '', nloop=1000)
    compare_methods('conjugate', '', 'z', nloop=1000)
    compare_methods('transpose', '', nloop=1000)
    compare_methods('compressed', '', nloop=1000)
    compare_methods('__getitem__', '0', nloop=1000)
    compare_methods('__getitem__', '(0,0)', nloop=1000)
    compare_methods('__getitem__', '[0,-1]', nloop=1000)
    compare_methods('__setitem__', '0, 17', nloop=1000, test=False)
    compare_methods('__setitem__', '(0,0), 17', nloop=1000, test=False)
    print('-' * 50)
    print('__setitem__ on small arrays')
    timer('nmxs.__setitem__((-1,0),numpy.ma.masked)', 'numpy.ma   ', nloop=10000)
    print('-' * 50)
    print('__setitem__ on large arrays')
    timer('nmxl.__setitem__((-1,0),numpy.ma.masked)', 'numpy.ma   ', nloop=10000)
    print('-' * 50)
    print('where on small arrays')
    timer('numpy.ma.where(nmxs>2,nmxs,nmys)', 'numpy.ma   ', nloop=1000)
    print('-' * 50)
    print('where on large arrays')
    timer('numpy.ma.where(nmxl>2,nmxl,nmyl)', 'numpy.ma   ', nloop=100)

