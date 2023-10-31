"""

Rules for building C/API module with f2py2e.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2004/11/26 11:13:06 $
Pearu Peterson

"""

import copy
from .auxfuncs import getfortranname, isexternal, isfunction, isfunction_wrap, isintent_in, isintent_out, islogicalfunction, ismoduleroutine, isscalar, issubroutine, issubroutine_wrap, outmess, show

def var2fixfortran(vars, a, fa=None, f90mode=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.func2subr.var2fixfortran', 'var2fixfortran(vars, a, fa=None, f90mode=None)', {'show': show, 'outmess': outmess, 'vars': vars, 'a': a, 'fa': fa, 'f90mode': f90mode}, 1)

def createfuncwrapper(rout, signature=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.func2subr.createfuncwrapper', 'createfuncwrapper(rout, signature=0)', {'isfunction': isfunction, 'getfortranname': getfortranname, 'ismoduleroutine': ismoduleroutine, 'var2fixfortran': var2fixfortran, 'isexternal': isexternal, 'isscalar': isscalar, 'isintent_in': isintent_in, 'islogicalfunction': islogicalfunction, 'rout': rout, 'signature': signature}, 1)

def createsubrwrapper(rout, signature=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.func2subr.createsubrwrapper', 'createsubrwrapper(rout, signature=0)', {'issubroutine': issubroutine, 'getfortranname': getfortranname, 'ismoduleroutine': ismoduleroutine, 'isexternal': isexternal, 'isscalar': isscalar, 'var2fixfortran': var2fixfortran, 'rout': rout, 'signature': signature}, 1)

def assubr(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.func2subr.assubr', 'assubr(rout)', {'isfunction_wrap': isfunction_wrap, 'getfortranname': getfortranname, 'outmess': outmess, 'copy': copy, 'isintent_out': isintent_out, 'createfuncwrapper': createfuncwrapper, 'issubroutine_wrap': issubroutine_wrap, 'createsubrwrapper': createsubrwrapper, 'rout': rout}, 2)

