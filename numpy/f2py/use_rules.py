"""

Build 'use others module data' mechanism for f2py2e.

Unfinished.

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2000/09/10 12:35:43 $
Pearu Peterson

"""

__version__ = '$Revision: 1.3 $'[10:-1]
f2py_version = 'See `f2py -v`'
from .auxfuncs import applyrules, dictappend, gentitle, hasnote, outmess
usemodule_rules = {'body': '\n#begintitle#\nstatic char doc_#apiname#[] = "\\\nVariable wrapper signature:\\n\\\n\t #name# = get_#name#()\\n\\\nArguments:\\n\\\n#docstr#";\nextern F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#);\nstatic PyObject *#apiname#(PyObject *capi_self, PyObject *capi_args) {\n/*#decl#*/\n\tif (!PyArg_ParseTuple(capi_args, "")) goto capi_fail;\nprintf("c: %d\\n",F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#));\n\treturn Py_BuildValue("");\ncapi_fail:\n\treturn NULL;\n}\n', 'method': '\t{"get_#name#",#apiname#,METH_VARARGS|METH_KEYWORDS,doc_#apiname#},', 'need': ['F_MODFUNC']}

def buildusevars(m, r):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.use_rules.buildusevars', 'buildusevars(m, r)', {'outmess': outmess, 'dictappend': dictappend, 'buildusevar': buildusevar, 'm': m, 'r': r}, 1)

def buildusevar(name, realname, vars, usemodulename):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.use_rules.buildusevar', 'buildusevar(name, realname, vars, usemodulename)', {'outmess': outmess, 'gentitle': gentitle, 'hasnote': hasnote, 'dictappend': dictappend, 'applyrules': applyrules, 'usemodule_rules': usemodule_rules, 'name': name, 'realname': realname, 'vars': vars, 'usemodulename': usemodulename}, 1)

