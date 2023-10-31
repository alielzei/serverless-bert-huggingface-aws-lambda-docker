"""

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 10:57:33 $
Pearu Peterson

"""

from . import __version__
f2py_version = __version__.version
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from .auxfuncs import *
__all__ = ['getctype', 'getstrlength', 'getarrdims', 'getpydocsign', 'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map', 'cb_sign2map', 'cb_routsign2map', 'common_sign2map']
using_newcore = True
depargs = []
lcb_map = {}
lcb2_map = {}
c2py_map = {'double': 'float', 'float': 'float', 'long_double': 'float', 'char': 'int', 'signed_char': 'int', 'unsigned_char': 'int', 'short': 'int', 'unsigned_short': 'int', 'int': 'int', 'long': 'int', 'long_long': 'long', 'unsigned': 'int', 'complex_float': 'complex', 'complex_double': 'complex', 'complex_long_double': 'complex', 'string': 'string', 'character': 'bytes'}
c2capi_map = {'double': 'NPY_DOUBLE', 'float': 'NPY_FLOAT', 'long_double': 'NPY_DOUBLE', 'char': 'NPY_STRING', 'unsigned_char': 'NPY_UBYTE', 'signed_char': 'NPY_BYTE', 'short': 'NPY_SHORT', 'unsigned_short': 'NPY_USHORT', 'int': 'NPY_INT', 'unsigned': 'NPY_UINT', 'long': 'NPY_LONG', 'long_long': 'NPY_LONG', 'complex_float': 'NPY_CFLOAT', 'complex_double': 'NPY_CDOUBLE', 'complex_long_double': 'NPY_CDOUBLE', 'string': 'NPY_STRING', 'character': 'NPY_CHAR'}
if using_newcore:
    c2capi_map = {'double': 'NPY_DOUBLE', 'float': 'NPY_FLOAT', 'long_double': 'NPY_LONGDOUBLE', 'char': 'NPY_BYTE', 'unsigned_char': 'NPY_UBYTE', 'signed_char': 'NPY_BYTE', 'short': 'NPY_SHORT', 'unsigned_short': 'NPY_USHORT', 'int': 'NPY_INT', 'unsigned': 'NPY_UINT', 'long': 'NPY_LONG', 'unsigned_long': 'NPY_ULONG', 'long_long': 'NPY_LONGLONG', 'unsigned_long_long': 'NPY_ULONGLONG', 'complex_float': 'NPY_CFLOAT', 'complex_double': 'NPY_CDOUBLE', 'complex_long_double': 'NPY_CDOUBLE', 'string': 'NPY_STRING', 'character': 'NPY_STRING'}
c2pycode_map = {'double': 'd', 'float': 'f', 'long_double': 'd', 'char': '1', 'signed_char': '1', 'unsigned_char': 'b', 'short': 's', 'unsigned_short': 'w', 'int': 'i', 'unsigned': 'u', 'long': 'l', 'long_long': 'L', 'complex_float': 'F', 'complex_double': 'D', 'complex_long_double': 'D', 'string': 'c', 'character': 'c'}
if using_newcore:
    c2pycode_map = {'double': 'd', 'float': 'f', 'long_double': 'g', 'char': 'b', 'unsigned_char': 'B', 'signed_char': 'b', 'short': 'h', 'unsigned_short': 'H', 'int': 'i', 'unsigned': 'I', 'long': 'l', 'unsigned_long': 'L', 'long_long': 'q', 'unsigned_long_long': 'Q', 'complex_float': 'F', 'complex_double': 'D', 'complex_long_double': 'G', 'string': 'S', 'character': 'c'}
c2buildvalue_map = {'double': 'd', 'float': 'f', 'char': 'b', 'signed_char': 'b', 'short': 'h', 'int': 'i', 'long': 'l', 'long_long': 'L', 'complex_float': 'N', 'complex_double': 'N', 'complex_long_double': 'N', 'string': 'y', 'character': 'c'}
f2cmap_all = {'real': {'': 'float', '4': 'float', '8': 'double', '12': 'long_double', '16': 'long_double'}, 'integer': {'': 'int', '1': 'signed_char', '2': 'short', '4': 'int', '8': 'long_long', '-1': 'unsigned_char', '-2': 'unsigned_short', '-4': 'unsigned', '-8': 'unsigned_long_long'}, 'complex': {'': 'complex_float', '8': 'complex_float', '16': 'complex_double', '24': 'complex_long_double', '32': 'complex_long_double'}, 'complexkind': {'': 'complex_float', '4': 'complex_float', '8': 'complex_double', '12': 'complex_long_double', '16': 'complex_long_double'}, 'logical': {'': 'int', '1': 'char', '2': 'short', '4': 'int', '8': 'long_long'}, 'double complex': {'': 'complex_double'}, 'double precision': {'': 'double'}, 'byte': {'': 'char'}}
f2cmap_default = copy.deepcopy(f2cmap_all)
f2cmap_mapped = []

def load_f2cmap_file(f2cmap_file):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.load_f2cmap_file', 'load_f2cmap_file(f2cmap_file)', {'copy': copy, 'f2cmap_default': f2cmap_default, 'os': os, 'outmess': outmess, 'c2py_map': c2py_map, 'f2cmap_mapped': f2cmap_mapped, 'errmess': errmess, 'f2cmap_file': f2cmap_file}, 1)
cformat_map = {'double': '%g', 'float': '%g', 'long_double': '%Lg', 'char': '%d', 'signed_char': '%d', 'unsigned_char': '%hhu', 'short': '%hd', 'unsigned_short': '%hu', 'int': '%d', 'unsigned': '%u', 'long': '%ld', 'unsigned_long': '%lu', 'long_long': '%ld', 'complex_float': '(%g,%g)', 'complex_double': '(%g,%g)', 'complex_long_double': '(%Lg,%Lg)', 'string': '\\"%s\\"', 'character': "'%c'"}

def getctype(var):
    """
    Determines C type
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.getctype', 'getctype(var)', {'isfunction': isfunction, 'getctype': getctype, 'errmess': errmess, 'issubroutine': issubroutine, 'ischaracter_or_characterarray': ischaracter_or_characterarray, 'isstring_or_stringarray': isstring_or_stringarray, 'f2cmap_all': f2cmap_all, 'os': os, 'isexternal': isexternal, 'var': var}, 1)

def f2cexpr(expr):
    """Rewrite Fortran expression as f2py supported C expression.

    Due to the lack of a proper expression parser in f2py, this
    function uses a heuristic approach that assumes that Fortran
    arithmetic expressions are valid C arithmetic expressions when
    mapping Fortran function calls to the corresponding C function/CPP
    macros calls.

    """
    expr = re.sub('\\blen\\b', 'f2py_slen', expr)
    return expr

def getstrlength(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.getstrlength', 'getstrlength(var)', {'isstringfunction': isstringfunction, 'getstrlength': getstrlength, 'errmess': errmess, 'isstring': isstring, 'f2cexpr': f2cexpr, 're': re, 'isintent_hide': isintent_hide, 'var': var}, 1)

def getarrdims(a, var, verbose=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.getarrdims', 'getarrdims(a, var, verbose=0)', {'isstring': isstring, 'isarray': isarray, 'getstrlength': getstrlength, 'isscalar': isscalar, 'copy': copy, 'depargs': depargs, 're': re, 'isintent_in': isintent_in, 'outmess': outmess, 'errmess': errmess, 'a': a, 'var': var, 'verbose': verbose}, 1)

def getpydocsign(a, var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.getpydocsign', 'getpydocsign(a, var)', {'isfunction': isfunction, 'getpydocsign': getpydocsign, 'errmess': errmess, 'isintent_in': isintent_in, 'isintent_inout': isintent_inout, 'isintent_out': isintent_out, 'getctype': getctype, 'hasinitvalue': hasinitvalue, 'getinit': getinit, 'isscalar': isscalar, 'c2py_map': c2py_map, 'c2pycode_map': c2pycode_map, 'isstring': isstring, 'getstrlength': getstrlength, 'isarray': isarray, 'isexternal': isexternal, 'lcb_map': lcb_map, 'lcb2_map': lcb2_map, 'a': a, 'var': var}, 1)

def getarrdocsign(a, var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.getarrdocsign', 'getarrdocsign(a, var)', {'getctype': getctype, 'isstring': isstring, 'isarray': isarray, 'getstrlength': getstrlength, 'isscalar': isscalar, 'c2py_map': c2py_map, 'c2pycode_map': c2pycode_map, 'a': a, 'var': var}, 1)

def getinit(a, var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.getinit', 'getinit(a, var)', {'isstring': isstring, 'hasinitvalue': hasinitvalue, 'iscomplex': iscomplex, 'iscomplexarray': iscomplexarray, 'markoutercomma': markoutercomma, 'isarray': isarray, 'a': a, 'var': var}, 2)

def get_elsize(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.get_elsize', 'get_elsize(var)', {'isstring': isstring, 'isstringarray': isstringarray, 'getstrlength': getstrlength, 'ischaracter': ischaracter, 'ischaracterarray': ischaracterarray, 'var': var}, 1)

def sign2map(a, var):
    """
    varname,ctype,atype
    init,init.r,init.i,pytype
    vardebuginfo,vardebugshowvalue,varshowvalue
    varrformat

    intent
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.sign2map', 'sign2map(a, var)', {'isintent_out': isintent_out, 'getctype': getctype, 'isintent_dict': isintent_dict, 'isarray': isarray, 'c2buildvalue_map': c2buildvalue_map, 'getinit': getinit, 'hasinitvalue': hasinitvalue, 'iscomplex': iscomplex, 'markoutercomma': markoutercomma, 'isexternal': isexternal, 'lcb_map': lcb_map, 'lcb2_map': lcb2_map, 'errmess': errmess, 'isstring': isstring, 'getstrlength': getstrlength, 'dictappend': dictappend, 'getarrdims': getarrdims, 'copy': copy, 'c2capi_map': c2capi_map, 'get_elsize': get_elsize, 'debugcapi': debugcapi, 'isintent_in': isintent_in, 'isintent_inout': isintent_inout, 'isrequired': isrequired, 'isoptional': isoptional, 'isintent_hide': isintent_hide, 'l_and': l_and, 'isscalar': isscalar, 'l_not': l_not, 'iscomplexarray': iscomplexarray, 'isstringarray': isstringarray, 'iscomplexfunction': iscomplexfunction, 'isfunction': isfunction, 'isintent_callback': isintent_callback, 'isintent_aux': isintent_aux, 'cformat_map': cformat_map, 'getpydocsign': getpydocsign, 'hasnote': hasnote, 'a': a, 'var': var}, 1)

def routsign2map(rout):
    """
    name,NAME,begintitle,endtitle
    rname,ctype,rformat
    routdebugshowvalue
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.routsign2map', 'routsign2map(rout)', {'getfortranname': getfortranname, 'gentitle': gentitle, 'getcallstatement': getcallstatement, 'getusercode': getusercode, 'getusercode1': getusercode1, 'cb_rules': cb_rules, 'errmess': errmess, 'getcallprotoargument': getcallprotoargument, 'isfunction': isfunction, 'getpydocsign': getpydocsign, 'getctype': getctype, 'hasresultnote': hasresultnote, 'c2buildvalue_map': c2buildvalue_map, 'debugcapi': debugcapi, 'cformat_map': cformat_map, 'isstringfunction': isstringfunction, 'getstrlength': getstrlength, 'hasnote': hasnote, 'rout': rout}, 1)

def modsign2map(m):
    """
    modulename
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.modsign2map', 'modsign2map(m)', {'ismodule': ismodule, 'getrestdoc': getrestdoc, 'hasnote': hasnote, 'getusercode': getusercode, 'getusercode1': getusercode1, 'getpymethoddef': getpymethoddef, 'm': m}, 1)

def cb_sign2map(a, var, index=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.cb_sign2map', 'cb_sign2map(a, var, index=None)', {'getctype': getctype, 'c2capi_map': c2capi_map, 'get_elsize': get_elsize, 'cformat_map': cformat_map, 'isarray': isarray, 'dictappend': dictappend, 'getarrdims': getarrdims, 'getpydocsign': getpydocsign, 'hasnote': hasnote, 'a': a, 'var': var, 'index': index}, 1)

def cb_routsign2map(rout, um):
    """
    name,begintitle,endtitle,argname
    ctype,rctype,maxnofargs,nofoptargs,returncptr
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.cb_routsign2map', 'cb_routsign2map(rout, um)', {'isintent_callback': isintent_callback, 'gentitle': gentitle, 'getctype': getctype, 'iscomplexfunction': iscomplexfunction, 'cformat_map': cformat_map, 'isstringfunction': isstringfunction, 'getstrlength': getstrlength, 'isfunction': isfunction, 'hasnote': hasnote, 'getpydocsign': getpydocsign, 'l_or': l_or, 'isintent_in': isintent_in, 'isintent_inout': isintent_inout, 'isoptional': isoptional, 'rout': rout, 'um': um}, 1)

def common_sign2map(a, var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.capi_maps.common_sign2map', 'common_sign2map(a, var)', {'getctype': getctype, 'isstringarray': isstringarray, 'c2capi_map': c2capi_map, 'get_elsize': get_elsize, 'cformat_map': cformat_map, 'isarray': isarray, 'dictappend': dictappend, 'getarrdims': getarrdims, 'isstring': isstring, 'getstrlength': getstrlength, 'getpydocsign': getpydocsign, 'hasnote': hasnote, 'getarrdocsign': getarrdocsign, 'a': a, 'var': var}, 1)

