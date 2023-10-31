"""

Build F90 module support for f2py2e.

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/02/03 19:30:23 $
Pearu Peterson

"""

__version__ = '$Revision: 1.27 $'[10:-1]
f2py_version = 'See `f2py -v`'
import numpy as np
from . import capi_maps
from . import func2subr
from .crackfortran import undo_rmbadname, undo_rmbadname1
from .auxfuncs import *
options = {}

def findf90modules(m):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f90mod_rules.findf90modules', 'findf90modules(m)', {'ismodule': ismodule, 'hasbody': hasbody, 'findf90modules': findf90modules, 'm': m}, 1)
fgetdims1 = '      external f2pysetdata\n      logical ns\n      integer r,i\n      integer(%d) s(*)\n      ns = .FALSE.\n      if (allocated(d)) then\n         do i=1,r\n            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then\n               ns = .TRUE.\n            end if\n         end do\n         if (ns) then\n            deallocate(d)\n         end if\n      end if\n      if ((.not.allocated(d)).and.(s(1).ge.1)) then' % np.intp().itemsize
fgetdims2 = '      end if\n      if (allocated(d)) then\n         do i=1,r\n            s(i) = size(d,i)\n         end do\n      end if\n      flag = 1\n      call f2pysetdata(d,allocated(d))'
fgetdims2_sa = '      end if\n      if (allocated(d)) then\n         do i=1,r\n            s(i) = size(d,i)\n         end do\n         !s(r) must be equal to len(d(1))\n      end if\n      flag = 2\n      call f2pysetdata(d,allocated(d))'

def buildhooks(pymod):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f90mod_rules.buildhooks', 'buildhooks(pymod)', {'findf90modules': findf90modules, 'hasbody': hasbody, 'l_or': l_or, 'isintent_hide': isintent_hide, 'isprivate': isprivate, 'outmess': outmess, 'capi_maps': capi_maps, 'hasnote': hasnote, 'fgetdims2': fgetdims2, 'undo_rmbadname1': undo_rmbadname1, 'isallocatable': isallocatable, 'fgetdims1': fgetdims1, 'isroutine': isroutine, 'isfunction': isfunction, 'func2subr': func2subr, 'applyrules': applyrules, 'dictappend': dictappend, 'undo_rmbadname': undo_rmbadname, 'pymod': pymod}, 2)

