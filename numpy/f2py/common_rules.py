"""

Build common block mechanism for f2py2e.

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 10:57:33 $
Pearu Peterson

"""

from . import __version__
f2py_version = __version__.version
from .auxfuncs import hasbody, hascommon, hasnote, isintent_hide, outmess
from . import capi_maps
from . import func2subr
from .crackfortran import rmbadname

def findcommonblocks(block, top=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.common_rules.findcommonblocks', 'findcommonblocks(block, top=1)', {'hascommon': hascommon, 'hasbody': hasbody, 'findcommonblocks': findcommonblocks, 'block': block, 'top': top}, 1)

def buildhooks(m):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.common_rules.buildhooks', 'buildhooks(m)', {'findcommonblocks': findcommonblocks, 'isintent_hide': isintent_hide, 'outmess': outmess, 'func2subr': func2subr, 'capi_maps': capi_maps, 'rmbadname': rmbadname, 'hasnote': hasnote, 'm': m}, 2)

