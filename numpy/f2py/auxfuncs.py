"""

Auxiliary functions for f2py2e.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) LICENSE.


NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/07/24 19:01:55 $
Pearu Peterson

"""

import pprint
import sys
import types
from functools import reduce
from . import __version__
from . import cfuncs
__all__ = ['applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle', 'getargs2', 'getcallprotoargument', 'getcallstatement', 'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode', 'getusercode1', 'hasbody', 'hascallstatement', 'hascommon', 'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote', 'isallocatable', 'isarray', 'isarrayofstrings', 'ischaracter', 'ischaracterarray', 'ischaracter_or_characterarray', 'iscomplex', 'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn', 'isdouble', 'isdummyroutine', 'isexternal', 'isfunction', 'isfunction_wrap', 'isint1', 'isint1array', 'isinteger', 'isintent_aux', 'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict', 'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace', 'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical', 'islogicalfunction', 'islong_complex', 'islong_double', 'islong_doublefunction', 'islong_long', 'islong_longfunction', 'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isrequired', 'isroutine', 'isscalar', 'issigned_long_longarray', 'isstring', 'isstringarray', 'isstring_or_stringarray', 'isstringfunction', 'issubroutine', 'issubroutine_wrap', 'isthreadsafe', 'isunsigned', 'isunsigned_char', 'isunsigned_chararray', 'isunsigned_long_long', 'isunsigned_long_longarray', 'isunsigned_short', 'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess', 'replace', 'show', 'stripcomma', 'throw_error', 'isattr_value']
f2py_version = __version__.version
errmess = sys.stderr.write
show = pprint.pprint
options = {}
debugoptions = []
wrapfuncs = 1

def outmess(t):
    if options.get('verbose', 1):
        sys.stdout.write(t)

def debugcapi(var):
    return 'capi' in debugoptions

def _ischaracter(var):
    return ('typespec' in var and var['typespec'] == 'character' and not isexternal(var))

def _isstring(var):
    return ('typespec' in var and var['typespec'] == 'character' and not isexternal(var))

def ischaracter_or_characterarray(var):
    return (_ischaracter(var) and 'charselector' not in var)

def ischaracter(var):
    return (ischaracter_or_characterarray(var) and not isarray(var))

def ischaracterarray(var):
    return (ischaracter_or_characterarray(var) and isarray(var))

def isstring_or_stringarray(var):
    return (_ischaracter(var) and 'charselector' in var)

def isstring(var):
    return (isstring_or_stringarray(var) and not isarray(var))

def isstringarray(var):
    return (isstring_or_stringarray(var) and isarray(var))

def isarrayofstrings(var):
    return (isstringarray(var) and var['dimension'][-1] == '(*)')

def isarray(var):
    return ('dimension' in var and not isexternal(var))

def isscalar(var):
    return not ((isarray(var) or isstring(var) or isexternal(var)))

def iscomplex(var):
    return (isscalar(var) and var.get('typespec') in ['complex', 'double complex'])

def islogical(var):
    return (isscalar(var) and var.get('typespec') == 'logical')

def isinteger(var):
    return (isscalar(var) and var.get('typespec') == 'integer')

def isreal(var):
    return (isscalar(var) and var.get('typespec') == 'real')

def get_kind(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.get_kind', 'get_kind(var)', {'var': var}, 1)

def isint1(var):
    return (var.get('typespec') == 'integer' and get_kind(var) == '1' and not isarray(var))

def islong_long(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.islong_long', 'islong_long(var)', {'isscalar': isscalar, 'get_kind': get_kind, 'var': var}, 1)

def isunsigned_char(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isunsigned_char', 'isunsigned_char(var)', {'isscalar': isscalar, 'get_kind': get_kind, 'var': var}, 1)

def isunsigned_short(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isunsigned_short', 'isunsigned_short(var)', {'isscalar': isscalar, 'get_kind': get_kind, 'var': var}, 1)

def isunsigned(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isunsigned', 'isunsigned(var)', {'isscalar': isscalar, 'get_kind': get_kind, 'var': var}, 1)

def isunsigned_long_long(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isunsigned_long_long', 'isunsigned_long_long(var)', {'isscalar': isscalar, 'get_kind': get_kind, 'var': var}, 1)

def isdouble(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isdouble', 'isdouble(var)', {'isscalar': isscalar, 'get_kind': get_kind, 'var': var}, 1)

def islong_double(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.islong_double', 'islong_double(var)', {'isscalar': isscalar, 'get_kind': get_kind, 'var': var}, 1)

def islong_complex(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.islong_complex', 'islong_complex(var)', {'iscomplex': iscomplex, 'get_kind': get_kind, 'var': var}, 1)

def iscomplexarray(var):
    return (isarray(var) and var.get('typespec') in ['complex', 'double complex'])

def isint1array(var):
    return (isarray(var) and var.get('typespec') == 'integer' and get_kind(var) == '1')

def isunsigned_chararray(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '-1')

def isunsigned_shortarray(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '-2')

def isunsignedarray(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '-4')

def isunsigned_long_longarray(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '-8')

def issigned_chararray(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '1')

def issigned_shortarray(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '2')

def issigned_array(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '4')

def issigned_long_longarray(var):
    return (isarray(var) and var.get('typespec') in ['integer', 'logical'] and get_kind(var) == '8')

def isallocatable(var):
    return ('attrspec' in var and 'allocatable' in var['attrspec'])

def ismutable(var):
    return not (('dimension' not in var or isstring(var)))

def ismoduleroutine(rout):
    return 'modulename' in rout

def ismodule(rout):
    return ('block' in rout and 'module' == rout['block'])

def isfunction(rout):
    return ('block' in rout and 'function' == rout['block'])

def isfunction_wrap(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isfunction_wrap', 'isfunction_wrap(rout)', {'isintent_c': isintent_c, 'wrapfuncs': wrapfuncs, 'isfunction': isfunction, 'isexternal': isexternal, 'rout': rout}, 1)

def issubroutine(rout):
    return ('block' in rout and 'subroutine' == rout['block'])

def issubroutine_wrap(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.issubroutine_wrap', 'issubroutine_wrap(rout)', {'isintent_c': isintent_c, 'issubroutine': issubroutine, 'hasassumedshape': hasassumedshape, 'rout': rout}, 1)

def isattr_value(var):
    return 'value' in var.get('attrspec', [])

def hasassumedshape(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.hasassumedshape', 'hasassumedshape(rout)', {'rout': rout}, 1)

def requiresf90wrapper(rout):
    return (ismoduleroutine(rout) or hasassumedshape(rout))

def isroutine(rout):
    return (isfunction(rout) or issubroutine(rout))

def islogicalfunction(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.islogicalfunction', 'islogicalfunction(rout)', {'isfunction': isfunction, 'islogical': islogical, 'rout': rout}, 1)

def islong_longfunction(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.islong_longfunction', 'islong_longfunction(rout)', {'isfunction': isfunction, 'islong_long': islong_long, 'rout': rout}, 1)

def islong_doublefunction(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.islong_doublefunction', 'islong_doublefunction(rout)', {'isfunction': isfunction, 'islong_double': islong_double, 'rout': rout}, 1)

def iscomplexfunction(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.iscomplexfunction', 'iscomplexfunction(rout)', {'isfunction': isfunction, 'iscomplex': iscomplex, 'rout': rout}, 1)

def iscomplexfunction_warn(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.iscomplexfunction_warn', 'iscomplexfunction_warn(rout)', {'iscomplexfunction': iscomplexfunction, 'outmess': outmess, 'rout': rout}, 1)

def isstringfunction(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isstringfunction', 'isstringfunction(rout)', {'isfunction': isfunction, 'isstring': isstring, 'rout': rout}, 1)

def hasexternals(rout):
    return ('externals' in rout and rout['externals'])

def isthreadsafe(rout):
    return ('f2pyenhancements' in rout and 'threadsafe' in rout['f2pyenhancements'])

def hasvariables(rout):
    return ('vars' in rout and rout['vars'])

def isoptional(var):
    return (('attrspec' in var and 'optional' in var['attrspec'] and 'required' not in var['attrspec']) and isintent_nothide(var))

def isexternal(var):
    return ('attrspec' in var and 'external' in var['attrspec'])

def isrequired(var):
    return (not isoptional(var) and isintent_nothide(var))

def isintent_in(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isintent_in', 'isintent_in(var)', {'var': var}, 1)

def isintent_inout(var):
    return ('intent' in var and (('inout' in var['intent'] or 'outin' in var['intent'])) and 'in' not in var['intent'] and 'hide' not in var['intent'] and 'inplace' not in var['intent'])

def isintent_out(var):
    return 'out' in var.get('intent', [])

def isintent_hide(var):
    return ('intent' in var and (('hide' in var['intent'] or ('out' in var['intent'] and 'in' not in var['intent'] and not l_or(isintent_inout, isintent_inplace)(var)))))

def isintent_nothide(var):
    return not isintent_hide(var)

def isintent_c(var):
    return 'c' in var.get('intent', [])

def isintent_cache(var):
    return 'cache' in var.get('intent', [])

def isintent_copy(var):
    return 'copy' in var.get('intent', [])

def isintent_overwrite(var):
    return 'overwrite' in var.get('intent', [])

def isintent_callback(var):
    return 'callback' in var.get('intent', [])

def isintent_inplace(var):
    return 'inplace' in var.get('intent', [])

def isintent_aux(var):
    return 'aux' in var.get('intent', [])

def isintent_aligned4(var):
    return 'aligned4' in var.get('intent', [])

def isintent_aligned8(var):
    return 'aligned8' in var.get('intent', [])

def isintent_aligned16(var):
    return 'aligned16' in var.get('intent', [])
isintent_dict = {isintent_in: 'INTENT_IN', isintent_inout: 'INTENT_INOUT', isintent_out: 'INTENT_OUT', isintent_hide: 'INTENT_HIDE', isintent_cache: 'INTENT_CACHE', isintent_c: 'INTENT_C', isoptional: 'OPTIONAL', isintent_inplace: 'INTENT_INPLACE', isintent_aligned4: 'INTENT_ALIGNED4', isintent_aligned8: 'INTENT_ALIGNED8', isintent_aligned16: 'INTENT_ALIGNED16'}

def isprivate(var):
    return ('attrspec' in var and 'private' in var['attrspec'])

def hasinitvalue(var):
    return '=' in var

def hasinitvalueasstring(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.hasinitvalueasstring', 'hasinitvalueasstring(var)', {'hasinitvalue': hasinitvalue, 'var': var}, 1)

def hasnote(var):
    return 'note' in var

def hasresultnote(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.hasresultnote', 'hasresultnote(rout)', {'isfunction': isfunction, 'hasnote': hasnote, 'rout': rout}, 1)

def hascommon(rout):
    return 'common' in rout

def containscommon(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.containscommon', 'containscommon(rout)', {'hascommon': hascommon, 'hasbody': hasbody, 'containscommon': containscommon, 'rout': rout}, 1)

def containsmodule(block):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.containsmodule', 'containsmodule(block)', {'ismodule': ismodule, 'hasbody': hasbody, 'containsmodule': containsmodule, 'block': block}, 1)

def hasbody(rout):
    return 'body' in rout

def hascallstatement(rout):
    return getcallstatement(rout) is not None

def istrue(var):
    return 1

def isfalse(var):
    return 0


class F2PYError(Exception):
    pass



class throw_error:
    
    def __init__(self, mess):
        self.mess = mess
    
    def __call__(self, var):
        mess = '\n\n  var = %s\n  Message: %s\n' % (var, self.mess)
        raise F2PYError(mess)


def l_and(*f):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.l_and', 'l_and(*f)', {'f': f}, 1)

def l_or(*f):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.l_or', 'l_or(*f)', {'f': f}, 1)

def l_not(f):
    return eval('lambda v,f=f:not f(v)')

def isdummyroutine(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.isdummyroutine', 'isdummyroutine(rout)', {'rout': rout}, 1)

def getfortranname(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.getfortranname', 'getfortranname(rout)', {'errmess': errmess, 'rout': rout}, 1)

def getmultilineblock(rout, blockname, comment=1, counter=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.getmultilineblock', 'getmultilineblock(rout, blockname, comment=1, counter=0)', {'errmess': errmess, 'rout': rout, 'blockname': blockname, 'comment': comment, 'counter': counter}, 1)

def getcallstatement(rout):
    return getmultilineblock(rout, 'callstatement')

def getcallprotoargument(rout, cb_map={}):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.getcallprotoargument', 'getcallprotoargument(rout, cb_map={})', {'getmultilineblock': getmultilineblock, 'hascallstatement': hascallstatement, 'outmess': outmess, 'l_and': l_and, 'isstringfunction': isstringfunction, 'l_not': l_not, 'isfunction_wrap': isfunction_wrap, 'isintent_callback': isintent_callback, 'isintent_c': isintent_c, 'l_or': l_or, 'isscalar': isscalar, 'iscomplex': iscomplex, 'isstring': isstring, 'isattr_value': isattr_value, 'isarrayofstrings': isarrayofstrings, 'isstringarray': isstringarray, 'rout': rout, 'cb_map': cb_map}, 1)

def getusercode(rout):
    return getmultilineblock(rout, 'usercode')

def getusercode1(rout):
    return getmultilineblock(rout, 'usercode', counter=1)

def getpymethoddef(rout):
    return getmultilineblock(rout, 'pymethoddef')

def getargs(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.getargs', 'getargs(rout)', {'rout': rout}, 2)

def getargs2(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.getargs2', 'getargs2(rout)', {'isintent_aux': isintent_aux, 'rout': rout}, 2)

def getrestdoc(rout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.getrestdoc', 'getrestdoc(rout)', {'rout': rout}, 1)

def gentitle(name):
    ln = (80 - len(name) - 6) // 2
    return '/*%s %s %s*/' % (ln * '*', name, ln * '*')

def flatlist(lst):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.flatlist', 'flatlist(lst)', {'reduce': reduce, 'flatlist': flatlist, 'lst': lst}, 1)

def stripcomma(s):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.stripcomma', 'stripcomma(s)', {'s': s}, 1)

def replace(str, d, defaultsep=''):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.replace', "replace(str, d, defaultsep='')", {'flatlist': flatlist, 'str': str, 'd': d, 'defaultsep': defaultsep}, 1)

def dictappend(rd, ar):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.dictappend', 'dictappend(rd, ar)', {'dictappend': dictappend, 'rd': rd, 'ar': ar}, 1)

def applyrules(rules, d, var={}):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.auxfuncs.applyrules', 'applyrules(rules, d, var={})', {'applyrules': applyrules, 'dictappend': dictappend, 'cfuncs': cfuncs, 'types': types, 'errmess': errmess, 'rules': rules, 'd': d, 'var': var}, 1)

