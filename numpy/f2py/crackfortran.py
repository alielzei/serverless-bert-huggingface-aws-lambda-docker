"""
crackfortran --- read fortran (77,90) code and extract declaration information.

Copyright 1999-2004 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/09/27 07:13:49 $
Pearu Peterson


Usage of crackfortran:
======================
Command line keys: -quiet,-verbose,-fix,-f77,-f90,-show,-h <pyffilename>
                   -m <module name for f77 routines>,--ignore-contains
Functions: crackfortran, crack2fortran
The following Fortran statements/constructions are supported
(or will be if needed):
   block data,byte,call,character,common,complex,contains,data,
   dimension,double complex,double precision,end,external,function,
   implicit,integer,intent,interface,intrinsic,
   logical,module,optional,parameter,private,public,
   program,real,(sequence?),subroutine,type,use,virtual,
   include,pythonmodule
Note: 'virtual' is mapped to 'dimension'.
Note: 'implicit integer (z) static (z)' is 'implicit static (z)' (this is minor bug).
Note: code after 'contains' will be ignored until its scope ends.
Note: 'common' statement is extended: dimensions are moved to variable definitions
Note: f2py directive: <commentchar>f2py<line> is read as <line>
Note: pythonmodule is introduced to represent Python module

Usage:
  `postlist=crackfortran(files)`
  `postlist` contains declaration information read from the list of files `files`.
  `crack2fortran(postlist)` returns a fortran code to be saved to pyf-file

  `postlist` has the following structure:
 *** it is a list of dictionaries containing `blocks':
     B = {'block','body','vars','parent_block'[,'name','prefix','args','result',
          'implicit','externals','interfaced','common','sortvars',
          'commonvars','note']}
     B['block'] = 'interface' | 'function' | 'subroutine' | 'module' |
                  'program' | 'block data' | 'type' | 'pythonmodule' |
                  'abstract interface'
     B['body'] --- list containing `subblocks' with the same structure as `blocks'
     B['parent_block'] --- dictionary of a parent block:
                             C['body'][<index>]['parent_block'] is C
     B['vars'] --- dictionary of variable definitions
     B['sortvars'] --- dictionary of variable definitions sorted by dependence (independent first)
     B['name'] --- name of the block (not if B['block']=='interface')
     B['prefix'] --- prefix string (only if B['block']=='function')
     B['args'] --- list of argument names if B['block']== 'function' | 'subroutine'
     B['result'] --- name of the return value (only if B['block']=='function')
     B['implicit'] --- dictionary {'a':<variable definition>,'b':...} | None
     B['externals'] --- list of variables being external
     B['interfaced'] --- list of variables being external and defined
     B['common'] --- dictionary of common blocks (list of objects)
     B['commonvars'] --- list of variables used in common blocks (dimensions are moved to variable definitions)
     B['from'] --- string showing the 'parents' of the current block
     B['use'] --- dictionary of modules used in current block:
         {<modulename>:{['only':<0|1>],['map':{<local_name1>:<use_name1>,...}]}}
     B['note'] --- list of LaTeX comments on the block
     B['f2pyenhancements'] --- optional dictionary
          {'threadsafe':'','fortranname':<name>,
           'callstatement':<C-expr>|<multi-line block>,
           'callprotoargument':<C-expr-list>,
           'usercode':<multi-line block>|<list of multi-line blocks>,
           'pymethoddef:<multi-line block>'
           }
     B['entry'] --- dictionary {entryname:argslist,..}
     B['varnames'] --- list of variable names given in the order of reading the
                       Fortran code, useful for derived types.
     B['saved_interface'] --- a string of scanned routine signature, defines explicit interface
 *** Variable definition is a dictionary
     D = B['vars'][<variable name>] =
     {'typespec'[,'attrspec','kindselector','charselector','=','typename']}
     D['typespec'] = 'byte' | 'character' | 'complex' | 'double complex' |
                     'double precision' | 'integer' | 'logical' | 'real' | 'type'
     D['attrspec'] --- list of attributes (e.g. 'dimension(<arrayspec>)',
                       'external','intent(in|out|inout|hide|c|callback|cache|aligned4|aligned8|aligned16)',
                       'optional','required', etc)
     K = D['kindselector'] = {['*','kind']} (only if D['typespec'] =
                         'complex' | 'integer' | 'logical' | 'real' )
     C = D['charselector'] = {['*','len','kind','f2py_len']}
                             (only if D['typespec']=='character')
     D['='] --- initialization expression string
     D['typename'] --- name of the type if D['typespec']=='type'
     D['dimension'] --- list of dimension bounds
     D['intent'] --- list of intent specifications
     D['depend'] --- list of variable names on which current variable depends on
     D['check'] --- list of C-expressions; if C-expr returns zero, exception is raised
     D['note'] --- list of LaTeX comments on the variable
 *** Meaning of kind/char selectors (few examples):
     D['typespec>']*K['*']
     D['typespec'](kind=K['kind'])
     character*C['*']
     character(len=C['len'],kind=C['kind'], f2py_len=C['f2py_len'])
     (see also fortran type declaration statement formats below)

Fortran 90 type declaration statement format (F77 is subset of F90)
====================================================================
(Main source: IBM XL Fortran 5.1 Language Reference Manual)
type declaration = <typespec> [[<attrspec>]::] <entitydecl>
<typespec> = byte                          |
             character[<charselector>]     |
             complex[<kindselector>]       |
             double complex                |
             double precision              |
             integer[<kindselector>]       |
             logical[<kindselector>]       |
             real[<kindselector>]          |
             type(<typename>)
<charselector> = * <charlen>               |
             ([len=]<len>[,[kind=]<kind>]) |
             (kind=<kind>[,len=<len>])
<kindselector> = * <intlen>                |
             ([kind=]<kind>)
<attrspec> = comma separated list of attributes.
             Only the following attributes are used in
             building up the interface:
                external
                (parameter --- affects '=' key)
                optional
                intent
             Other attributes are ignored.
<intentspec> = in | out | inout
<arrayspec> = comma separated list of dimension bounds.
<entitydecl> = <name> [[*<charlen>][(<arrayspec>)] | [(<arrayspec>)]*<charlen>]
                      [/<init_expr>/ | =<init_expr>] [,<entitydecl>]

In addition, the following attributes are used: check,depend,note

TODO:
    * Apply 'parameter' attribute (e.g. 'integer parameter :: i=2' 'real x(i)'
                                   -> 'real x(2)')
    The above may be solved by creating appropriate preprocessor program, for example.

"""

import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
try:
    import charset_normalizer
except ImportError:
    charset_normalizer = None
from . import __version__
from .auxfuncs import *
from . import symbolic
f2py_version = __version__.version
strictf77 = 1
sourcecodeform = 'fix'
quiet = 0
verbose = 1
tabchar = 4 * ' '
pyffilename = ''
f77modulename = ''
skipemptyends = 0
ignorecontains = 1
dolowercase = 1
debug = []
beginpattern = ''
currentfilename = ''
expectbegin = 1
f90modulevars = {}
filepositiontext = ''
gotnextfile = 1
groupcache = None
groupcounter = 0
grouplist = {groupcounter: []}
groupname = ''
include_paths = []
neededmodule = -1
onlyfuncs = []
previous_context = None
skipblocksuntil = -1
skipfuncs = []
skipfunctions = []
usermodules = []

def reset_global_f2py_vars():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.reset_global_f2py_vars', 'reset_global_f2py_vars()', {}, 0)

def outmess(line, flag=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.outmess', 'outmess(line, flag=1)', {'verbose': verbose, 'quiet': quiet, 'sys': sys, 'filepositiontext': filepositiontext, 'line': line, 'flag': flag}, 1)
re._MAXCACHE = 50
defaultimplicitrules = {}
for c in 'abcdefghopqrstuvwxyz$_':
    defaultimplicitrules[c] = {'typespec': 'real'}
for c in 'ijklmn':
    defaultimplicitrules[c] = {'typespec': 'integer'}
badnames = {}
invbadnames = {}
for n in ['int', 'double', 'float', 'char', 'short', 'long', 'void', 'case', 'while', 'return', 'signed', 'unsigned', 'if', 'for', 'typedef', 'sizeof', 'union', 'struct', 'static', 'register', 'new', 'break', 'do', 'goto', 'switch', 'continue', 'else', 'inline', 'extern', 'delete', 'const', 'auto', 'len', 'rank', 'shape', 'index', 'slen', 'size', '_i', 'max', 'min', 'flen', 'fshape', 'string', 'complex_double', 'float_double', 'stdin', 'stderr', 'stdout', 'type', 'default']:
    badnames[n] = n + '_bn'
    invbadnames[n + '_bn'] = n

def rmbadname1(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.rmbadname1', 'rmbadname1(name)', {'badnames': badnames, 'errmess': errmess, 'name': name}, 1)

def rmbadname(names):
    return [rmbadname1(_m) for _m in names]

def undo_rmbadname1(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.undo_rmbadname1', 'undo_rmbadname1(name)', {'invbadnames': invbadnames, 'errmess': errmess, 'name': name}, 1)

def undo_rmbadname(names):
    return [undo_rmbadname1(_m) for _m in names]

def getextension(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.getextension', 'getextension(name)', {'name': name}, 1)
is_f_file = re.compile('.*\\.(for|ftn|f77|f)\\Z', re.I).match
_has_f_header = re.compile('-\\*-\\s*fortran\\s*-\\*-', re.I).search
_has_f90_header = re.compile('-\\*-\\s*f90\\s*-\\*-', re.I).search
_has_fix_header = re.compile('-\\*-\\s*fix\\s*-\\*-', re.I).search
_free_f90_start = re.compile('[^c*]\\s*[^\\s\\d\\t]', re.I).match

def openhook(filename, mode):
    """Ensures that filename is opened with correct encoding parameter.

    This function uses charset_normalizer package, when available, for
    determining the encoding of the file to be opened. When charset_normalizer
    is not available, the function detects only UTF encodings, otherwise, ASCII
    encoding is used as fallback.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.openhook', 'openhook(filename, mode)', {'charset_normalizer': charset_normalizer, 'os': os, 'codecs': codecs, 'filename': filename, 'mode': mode}, 1)

def is_free_format(file):
    """Check if file is in free format Fortran."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.is_free_format', 'is_free_format(file)', {'openhook': openhook, '_has_f_header': _has_f_header, '_has_f90_header': _has_f90_header, '_free_f90_start': _free_f90_start, 'file': file}, 1)

def readfortrancode(ffile, dowithline=show, istop=1):
    """
    Read fortran codes from files and
     1) Get rid of comments, line continuations, and empty lines; lower cases.
     2) Call dowithline(line) on every line.
     3) Recursively call itself when statement "include '<filename>'" is met.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.readfortrancode', 'readfortrancode(ffile, dowithline=show, istop=1)', {'re': re, 'fileinput': fileinput, 'openhook': openhook, 'os': os, 'is_f_file': is_f_file, '_has_f90_header': _has_f90_header, '_has_fix_header': _has_fix_header, 'is_free_format': is_free_format, 'beginpattern77': beginpattern77, 'beginpattern90': beginpattern90, 'outmess': outmess, 'split_by_unquoted': split_by_unquoted, 'errmess': errmess, 'readfortrancode': readfortrancode, 'include_paths': include_paths, 'ffile': ffile, 'dowithline': dowithline, 'istop': istop, 'show': show}, 1)
beforethisafter = '\\s*(?P<before>%s(?=\\s*(\\b(%s)\\b)))' + '\\s*(?P<this>(\\b(%s)\\b))' + '\\s*(?P<after>%s)\\s*\\Z'
fortrantypes = 'character|logical|integer|real|complex|double\\s*(precision\\s*(complex|)|complex)|type(?=\\s*\\([\\w\\s,=(*)]*\\))|byte'
typespattern = (re.compile(beforethisafter % ('', fortrantypes, fortrantypes, '.*'), re.I), 'type')
typespattern4implicit = re.compile(beforethisafter % ('', fortrantypes + '|static|automatic|undefined', fortrantypes + '|static|automatic|undefined', '.*'), re.I)
functionpattern = (re.compile(beforethisafter % ('([a-z]+[\\w\\s(=*+-/)]*?|)', 'function', 'function', '.*'), re.I), 'begin')
subroutinepattern = (re.compile(beforethisafter % ('[a-z\\s]*?', 'subroutine', 'subroutine', '.*'), re.I), 'begin')
groupbegins77 = 'program|block\\s*data'
beginpattern77 = (re.compile(beforethisafter % ('', groupbegins77, groupbegins77, '.*'), re.I), 'begin')
groupbegins90 = groupbegins77 + '|module(?!\\s*procedure)|python\\s*module|(abstract|)\\s*interface|' + 'type(?!\\s*\\()'
beginpattern90 = (re.compile(beforethisafter % ('', groupbegins90, groupbegins90, '.*'), re.I), 'begin')
groupends = 'end|endprogram|endblockdata|endmodule|endpythonmodule|endinterface|endsubroutine|endfunction'
endpattern = (re.compile(beforethisafter % ('', groupends, groupends, '.*'), re.I), 'end')
endifs = 'end\\s*(if|do|where|select|while|forall|associate|block|' + 'critical|enum|team)'
endifpattern = (re.compile(beforethisafter % ('[\\w]*?', endifs, endifs, '[\\w\\s]*'), re.I), 'endif')
moduleprocedures = 'module\\s*procedure'
moduleprocedurepattern = (re.compile(beforethisafter % ('', moduleprocedures, moduleprocedures, '.*'), re.I), 'moduleprocedure')
implicitpattern = (re.compile(beforethisafter % ('', 'implicit', 'implicit', '.*'), re.I), 'implicit')
dimensionpattern = (re.compile(beforethisafter % ('', 'dimension|virtual', 'dimension|virtual', '.*'), re.I), 'dimension')
externalpattern = (re.compile(beforethisafter % ('', 'external', 'external', '.*'), re.I), 'external')
optionalpattern = (re.compile(beforethisafter % ('', 'optional', 'optional', '.*'), re.I), 'optional')
requiredpattern = (re.compile(beforethisafter % ('', 'required', 'required', '.*'), re.I), 'required')
publicpattern = (re.compile(beforethisafter % ('', 'public', 'public', '.*'), re.I), 'public')
privatepattern = (re.compile(beforethisafter % ('', 'private', 'private', '.*'), re.I), 'private')
intrinsicpattern = (re.compile(beforethisafter % ('', 'intrinsic', 'intrinsic', '.*'), re.I), 'intrinsic')
intentpattern = (re.compile(beforethisafter % ('', 'intent|depend|note|check', 'intent|depend|note|check', '\\s*\\(.*?\\).*'), re.I), 'intent')
parameterpattern = (re.compile(beforethisafter % ('', 'parameter', 'parameter', '\\s*\\(.*'), re.I), 'parameter')
datapattern = (re.compile(beforethisafter % ('', 'data', 'data', '.*'), re.I), 'data')
callpattern = (re.compile(beforethisafter % ('', 'call', 'call', '.*'), re.I), 'call')
entrypattern = (re.compile(beforethisafter % ('', 'entry', 'entry', '.*'), re.I), 'entry')
callfunpattern = (re.compile(beforethisafter % ('', 'callfun', 'callfun', '.*'), re.I), 'callfun')
commonpattern = (re.compile(beforethisafter % ('', 'common', 'common', '.*'), re.I), 'common')
usepattern = (re.compile(beforethisafter % ('', 'use', 'use', '.*'), re.I), 'use')
containspattern = (re.compile(beforethisafter % ('', 'contains', 'contains', ''), re.I), 'contains')
formatpattern = (re.compile(beforethisafter % ('', 'format', 'format', '.*'), re.I), 'format')
f2pyenhancementspattern = (re.compile(beforethisafter % ('', 'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef', 'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef', '.*'), re.I | re.S), 'f2pyenhancements')
multilinepattern = (re.compile("\\s*(?P<before>''')(?P<this>.*?)(?P<after>''')\\s*\\Z", re.S), 'multiline')

def split_by_unquoted(line, characters):
    """
    Splits the line into (line[:i], line[i:]),
    where i is the index of first occurrence of one of the characters
    not within quotes, or len(line) if no such index exists
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.split_by_unquoted', 'split_by_unquoted(line, characters)', {'re': re, 'line': line, 'characters': characters}, 2)

def _simplifyargs(argsline):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._simplifyargs', '_simplifyargs(argsline)', {'markoutercomma': markoutercomma, 'argsline': argsline}, 1)
crackline_re_1 = re.compile('\\s*(?P<result>\\b[a-z]+\\w*\\b)\\s*=.*', re.I)

def crackline(line, reset=0):
    """
    reset=-1  --- initialize
    reset=0   --- crack the line
    reset=1   --- final check if mismatch of blocks occurred

    Cracked data is saved in grouplist[0].
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.crackline', 'crackline(line, reset=0)', {'split_by_unquoted': split_by_unquoted, 'f2pyenhancementspattern': f2pyenhancementspattern, 'multilinepattern': multilinepattern, 'crackline': crackline, 'f77modulename': f77modulename, 'outmess': outmess, 'dimensionpattern': dimensionpattern, 'externalpattern': externalpattern, 'intentpattern': intentpattern, 'optionalpattern': optionalpattern, 'requiredpattern': requiredpattern, 'parameterpattern': parameterpattern, 'datapattern': datapattern, 'publicpattern': publicpattern, 'privatepattern': privatepattern, 'intrinsicpattern': intrinsicpattern, 'endifpattern': endifpattern, 'endpattern': endpattern, 'formatpattern': formatpattern, 'beginpattern': beginpattern, 'functionpattern': functionpattern, 'subroutinepattern': subroutinepattern, 'implicitpattern': implicitpattern, 'typespattern': typespattern, 'commonpattern': commonpattern, 'callpattern': callpattern, 'usepattern': usepattern, 'containspattern': containspattern, 'entrypattern': entrypattern, 'moduleprocedurepattern': moduleprocedurepattern, 'crackline_re_1': crackline_re_1, 'invbadnames': invbadnames, 're': re, 'markouterparen': markouterparen, '_simplifyargs': _simplifyargs, 'callfunpattern': callfunpattern, 'analyzeline': analyzeline, 'verbose': verbose, 'currentfilename': currentfilename, 'filepositiontext': filepositiontext, 'skipemptyends': skipemptyends, 'ignorecontains': ignorecontains, 'line': line, 'reset': reset}, 1)

def markouterparen(line):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.markouterparen', 'markouterparen(line)', {'line': line}, 1)

def markoutercomma(line, comma=','):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.markoutercomma', "markoutercomma(line, comma=',')", {'split_by_unquoted': split_by_unquoted, 'line': line, 'comma': comma}, 1)

def unmarkouterparen(line):
    r = line.replace('@(@', '(').replace('@)@', ')')
    return r

def appenddecl(decl, decl2, force=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.appenddecl', 'appenddecl(decl, decl2, force=1)', {'setattrspec': setattrspec, 'setkindselector': setkindselector, 'setcharselector': setcharselector, 'errmess': errmess, 'decl': decl, 'decl2': decl2, 'force': force}, 1)
selectpattern = re.compile('\\s*(?P<this>(@\\(@.*?@\\)@|\\*[\\d*]+|\\*\\s*@\\(@.*?@\\)@|))(?P<after>.*)\\Z', re.I)
typedefpattern = re.compile('(?:,(?P<attributes>[\\w(),]+))?(::)?(?P<name>\\b[a-z$_][\\w$]*\\b)(?:\\((?P<params>[\\w,]*)\\))?\\Z', re.I)
nameargspattern = re.compile('\\s*(?P<name>\\b[\\w$]+\\b)\\s*(@\\(@\\s*(?P<args>[\\w\\s,]*)\\s*@\\)@|)\\s*((result(\\s*@\\(@\\s*(?P<result>\\b[\\w$]+\\b)\\s*@\\)@|))|(bind\\s*@\\(@\\s*(?P<bind>.*)\\s*@\\)@))*\\s*\\Z', re.I)
operatorpattern = re.compile('\\s*(?P<scheme>(operator|assignment))@\\(@\\s*(?P<name>[^)]+)\\s*@\\)@\\s*\\Z', re.I)
callnameargspattern = re.compile('\\s*(?P<name>\\b[\\w$]+\\b)\\s*@\\(@\\s*(?P<args>.*)\\s*@\\)@\\s*\\Z', re.I)
real16pattern = re.compile('([-+]?(?:\\d+(?:\\.\\d*)?|\\d*\\.\\d+))[dD]((?:[-+]?\\d+)?)')
real8pattern = re.compile('([-+]?((?:\\d+(?:\\.\\d*)?|\\d*\\.\\d+))[eE]((?:[-+]?\\d+)?)|(\\d+\\.\\d*))')
_intentcallbackpattern = re.compile('intent\\s*\\(.*?\\bcallback\\b', re.I)

def _is_intent_callback(vdecl):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._is_intent_callback', '_is_intent_callback(vdecl)', {'_intentcallbackpattern': _intentcallbackpattern, 'vdecl': vdecl}, 1)

def _resolvetypedefpattern(line):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._resolvetypedefpattern', '_resolvetypedefpattern(line)', {'typedefpattern': typedefpattern, 'line': line}, 3)

def _resolvenameargspattern(line):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._resolvenameargspattern', '_resolvenameargspattern(line)', {'markouterparen': markouterparen, 'nameargspattern': nameargspattern, 'operatorpattern': operatorpattern, 'callnameargspattern': callnameargspattern, 'line': line}, 4)

def analyzeline(m, case, line):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.analyzeline', 'analyzeline(m, case, line)', {'skipemptyends': skipemptyends, 'os': os, 'currentfilename': currentfilename, 'outmess': outmess, 'groupname': groupname, 'groupcache': groupcache, 'grouplist': grouplist, 're': re, '_resolvetypedefpattern': _resolvetypedefpattern, '_resolvenameargspattern': _resolvenameargspattern, 'rmbadname': rmbadname, 'markoutercomma': markoutercomma, 'f77modulename': f77modulename, 'verbose': verbose, 'rmbadname1': rmbadname1, 'copy': copy, 'appenddecl': appenddecl, 'typespattern': typespattern, 'cracktypespec0': cracktypespec0, 'updatevars': updatevars, 'markouterparen': markouterparen, 'namepattern': namepattern, '_intentcallbackpattern': _intentcallbackpattern, 'errmess': errmess, 'get_parameters': get_parameters, 'determineexprtype': determineexprtype, 'real16pattern': real16pattern, 'typespattern4implicit': typespattern4implicit, 'cracktypespec': cracktypespec, 'appendmultiline': appendmultiline, 'm': m, 'case': case, 'line': line}, 1)

def appendmultiline(group, context_name, ml):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.appendmultiline', 'appendmultiline(group, context_name, ml)', {'group': group, 'context_name': context_name, 'ml': ml}, 1)

def cracktypespec0(typespec, ll):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.cracktypespec0', 'cracktypespec0(typespec, ll)', {'re': re, 'selectpattern': selectpattern, 'markouterparen': markouterparen, 'outmess': outmess, 'unmarkouterparen': unmarkouterparen, 'typespec': typespec, 'll': ll}, 1)
namepattern = re.compile('\\s*(?P<name>\\b\\w+\\b)\\s*(?P<after>.*)\\s*\\Z', re.I)
kindselector = re.compile('\\s*(\\(\\s*(kind\\s*=)?\\s*(?P<kind>.*)\\s*\\)|\\*\\s*(?P<kind2>.*?))\\s*\\Z', re.I)
charselector = re.compile('\\s*(\\((?P<lenkind>.*)\\)|\\*\\s*(?P<charlen>.*))\\s*\\Z', re.I)
lenkindpattern = re.compile('\\s*(kind\\s*=\\s*(?P<kind>.*?)\\s*(@,@\\s*len\\s*=\\s*(?P<len>.*)|)|(len\\s*=\\s*|)(?P<len2>.*?)\\s*(@,@\\s*(kind\\s*=\\s*|)(?P<kind2>.*)|(f2py_len\\s*=\\s*(?P<f2py_len>.*))|))\\s*\\Z', re.I)
lenarraypattern = re.compile('\\s*(@\\(@\\s*(?!/)\\s*(?P<array>.*?)\\s*@\\)@\\s*\\*\\s*(?P<len>.*?)|(\\*\\s*(?P<len2>.*?)|)\\s*(@\\(@\\s*(?!/)\\s*(?P<array2>.*?)\\s*@\\)@|))\\s*(=\\s*(?P<init>.*?)|(@\\(@|)/\\s*(?P<init2>.*?)\\s*/(@\\)@|)|)\\s*\\Z', re.I)

def removespaces(expr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.removespaces', 'removespaces(expr)', {'expr': expr}, 1)

def markinnerspaces(line):
    """
    The function replace all spaces in the input variable line which are 
    surrounded with quotation marks, with the triplet "@_@".

    For instance, for the input "a 'b c'" the function returns "a 'b@_@c'"

    Parameters
    ----------
    line : str

    Returns
    -------
    str

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.markinnerspaces', 'markinnerspaces(line)', {'line': line}, 1)

def updatevars(typespec, selector, attrspec, entitydecl):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.updatevars', 'updatevars(typespec, selector, attrspec, entitydecl)', {'cracktypespec': cracktypespec, 'markoutercomma': markoutercomma, 're': re, 'removespaces': removespaces, 'markinnerspaces': markinnerspaces, 'namepattern': namepattern, 'outmess': outmess, 'rmbadname1': rmbadname1, 'groupcache': groupcache, 'groupcounter': groupcounter, 'copy': copy, 'errmess': errmess, 'lenarraypattern': lenarraypattern, 'markouterparen': markouterparen, 'unmarkouterparen': unmarkouterparen, 'typespec': typespec, 'selector': selector, 'attrspec': attrspec, 'entitydecl': entitydecl}, 1)

def cracktypespec(typespec, selector):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.cracktypespec', 'cracktypespec(typespec, selector)', {'kindselector': kindselector, 'outmess': outmess, 'rmbadname1': rmbadname1, 'charselector': charselector, 'lenkindpattern': lenkindpattern, 'markoutercomma': markoutercomma, 're': re, 'typespec': typespec, 'selector': selector}, 1)

def setattrspec(decl, attr, force=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.setattrspec', 'setattrspec(decl, attr, force=0)', {'decl': decl, 'attr': attr, 'force': force}, 1)

def setkindselector(decl, sel, force=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.setkindselector', 'setkindselector(decl, sel, force=0)', {'decl': decl, 'sel': sel, 'force': force}, 1)

def setcharselector(decl, sel, force=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.setcharselector', 'setcharselector(decl, sel, force=0)', {'decl': decl, 'sel': sel, 'force': force}, 1)

def getblockname(block, unknown='unknown'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.getblockname', "getblockname(block, unknown='unknown')", {'block': block, 'unknown': unknown}, 1)

def setmesstext(block):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.setmesstext', 'setmesstext(block)', {'block': block}, 0)

def get_usedict(block):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.get_usedict', 'get_usedict(block)', {'get_usedict': get_usedict, 'block': block}, 1)

def get_useparameters(block, param_map=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.get_useparameters', 'get_useparameters(block, param_map=None)', {'get_usedict': get_usedict, 'f90modulevars': f90modulevars, 'outmess': outmess, 'get_parameters': get_parameters, 'errmess': errmess, 'block': block, 'param_map': param_map}, 1)

def postcrack2(block, tab='', param_map=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.postcrack2', "postcrack2(block, tab='', param_map=None)", {'f90modulevars': f90modulevars, 'postcrack2': postcrack2, 'setmesstext': setmesstext, 'outmess': outmess, 'get_useparameters': get_useparameters, 'block': block, 'tab': tab, 'param_map': param_map}, 1)

def postcrack(block, args=None, tab=''):
    """
    TODO:
          function return values
          determine expression types if in argument list
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.postcrack', "postcrack(block, args=None, tab='')", {'setmesstext': setmesstext, 'postcrack': postcrack, 'outmess': outmess, 'analyzeargs': analyzeargs, 'analyzecommon': analyzecommon, 'analyzevars': analyzevars, 'sortvarnames': sortvarnames, 'analyzebody': analyzebody, 'copy': copy, 'isexternal': isexternal, 'usermodules': usermodules, 'block': block, 'args': args, 'tab': tab}, 1)

def sortvarnames(vars):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.sortvarnames', 'sortvarnames(vars)', {'errmess': errmess, 'vars': vars}, 1)

def analyzecommon(block):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.analyzecommon', 'analyzecommon(block)', {'hascommon': hascommon, 're': re, 'markoutercomma': markoutercomma, 'rmbadname1': rmbadname1, 'errmess': errmess, 'block': block}, 1)

def analyzebody(block, args, tab=''):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.analyzebody', "analyzebody(block, args, tab='')", {'setmesstext': setmesstext, 'skipfuncs': skipfuncs, 'onlyfuncs': onlyfuncs, 'crack2fortrangen': crack2fortrangen, 'postcrack': postcrack, 'usermodules': usermodules, 'f90modulevars': f90modulevars, 'block': block, 'args': args, 'tab': tab}, 1)

def buildimplicitrules(block):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.buildimplicitrules', 'buildimplicitrules(block)', {'setmesstext': setmesstext, 'defaultimplicitrules': defaultimplicitrules, 'verbose': verbose, 'outmess': outmess, 'block': block}, 2)

def myeval(e, g=None, l=None):
    """ Like `eval` but returns only integers and floats """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.myeval', 'myeval(e, g=None, l=None)', {'e': e, 'g': g, 'l': l}, 1)
getlincoef_re_1 = re.compile('\\A\\b\\w+\\b\\Z', re.I)

def getlincoef(e, xset):
    """
    Obtain ``a`` and ``b`` when ``e == "a*x+b"``, where ``x`` is a symbol in
    xset.

    >>> getlincoef('2*x + 1', {'x'})
    (2, 1, 'x')
    >>> getlincoef('3*x + x*2 + 2 + 1', {'x'})
    (5, 3, 'x')
    >>> getlincoef('0', {'x'})
    (0, 0, None)
    >>> getlincoef('0*x', {'x'})
    (0, 0, 'x')
    >>> getlincoef('x*x', {'x'})
    (None, None, None)

    This can be tricked by sufficiently complex expressions

    >>> getlincoef('(x - 0.5)*(x - 1.5)*(x - 1)*x + 2*x + 3', {'x'})
    (2.0, 3.0, 'x')
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.getlincoef', 'getlincoef(e, xset)', {'myeval': myeval, 'getlincoef_re_1': getlincoef_re_1, 're': re, 'e': e, 'xset': xset}, 3)
word_pattern = re.compile('\\b[a-z][\\w$]*\\b', re.I)

def _get_depend_dict(name, vars, deps):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._get_depend_dict', '_get_depend_dict(name, vars, deps)', {'isstring': isstring, 'word_pattern': word_pattern, '_get_depend_dict': _get_depend_dict, 'outmess': outmess, 'name': name, 'vars': vars, 'deps': deps}, 1)

def _calc_depend_dict(vars):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._calc_depend_dict', '_calc_depend_dict(vars)', {'_get_depend_dict': _get_depend_dict, 'vars': vars}, 1)

def get_sorted_names(vars):
    """
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.get_sorted_names', 'get_sorted_names(vars)', {'_calc_depend_dict': _calc_depend_dict, 'vars': vars}, 1)

def _kind_func(string):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._kind_func', '_kind_func(string)', {'real16pattern': real16pattern, 'real8pattern': real8pattern, 'string': string}, 1)

def _selected_int_kind_func(r):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._selected_int_kind_func', '_selected_int_kind_func(r)', {'r': r}, 1)

def _selected_real_kind_func(p, r=0, radix=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._selected_real_kind_func', '_selected_real_kind_func(p, r=0, radix=0)', {'platform': platform, 'p': p, 'r': r, 'radix': radix}, 1)

def get_parameters(vars, global_params={}):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.get_parameters', 'get_parameters(vars, global_params={})', {'copy': copy, '_kind_func': _kind_func, '_selected_int_kind_func': _selected_int_kind_func, '_selected_real_kind_func': _selected_real_kind_func, 'get_sorted_names': get_sorted_names, 're': re, 'islogical': islogical, 'isdouble': isdouble, 'real16pattern': real16pattern, 'iscomplex': iscomplex, 'outmess': outmess, 'real8pattern': real8pattern, 'isstring': isstring, 'vars': vars, 'global_params': global_params}, 1)

def _eval_length(length, params):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._eval_length', '_eval_length(length, params)', {'_eval_scalar': _eval_scalar, 'length': length, 'params': params}, 1)
_is_kind_number = re.compile('\\d+_').match

def _eval_scalar(value, params):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._eval_scalar', '_eval_scalar(value, params)', {'_is_kind_number': _is_kind_number, 'errmess': errmess, 'value': value, 'params': params}, 1)

def analyzevars(block):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.analyzevars', 'analyzevars(block)', {'setmesstext': setmesstext, 'buildimplicitrules': buildimplicitrules, 'copy': copy, 'setattrspec': setattrspec, 'get_parameters': get_parameters, 'get_useparameters': get_useparameters, 're': re, 'outmess': outmess, 'markoutercomma': markoutercomma, 'rmbadname': rmbadname, 'symbolic': symbolic, 'l_or': l_or, 'isintent_in': isintent_in, 'isintent_inout': isintent_inout, 'isintent_inplace': isintent_inplace, 'isarray': isarray, 'isstring': isstring, '_eval_length': _eval_length, 'isscalar': isscalar, '_eval_scalar': _eval_scalar, 'appenddecl': appenddecl, 'typespattern': typespattern, 'cracktypespec0': cracktypespec0, 'cracktypespec': cracktypespec, 'isintent_callback': isintent_callback, 'isintent_aux': isintent_aux, 'block': block}, 1)
analyzeargs_re_1 = re.compile('\\A[a-z]+[\\w$]*\\Z', re.I)

def expr2name(a, block, args=[]):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.expr2name', 'expr2name(a, block, args=[])', {'analyzeargs_re_1': analyzeargs_re_1, 'buildimplicitrules': buildimplicitrules, 'determineexprtype': determineexprtype, 'string': string, 'setattrspec': setattrspec, 'a': a, 'block': block, 'args': args}, 1)

def analyzeargs(block):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.analyzeargs', 'analyzeargs(block)', {'setmesstext': setmesstext, 'buildimplicitrules': buildimplicitrules, 'expr2name': expr2name, 'block': block}, 1)
determineexprtype_re_1 = re.compile('\\A\\(.+?,.+?\\)\\Z', re.I)
determineexprtype_re_2 = re.compile('\\A[+-]?\\d+(_(?P<name>\\w+)|)\\Z', re.I)
determineexprtype_re_3 = re.compile('\\A[+-]?[\\d.]+[-\\d+de.]*(_(?P<name>\\w+)|)\\Z', re.I)
determineexprtype_re_4 = re.compile('\\A\\(.*\\)\\Z', re.I)
determineexprtype_re_5 = re.compile('\\A(?P<name>\\w+)\\s*\\(.*?\\)\\s*\\Z', re.I)

def _ensure_exprdict(r):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran._ensure_exprdict', '_ensure_exprdict(r)', {'r': r}, 1)

def determineexprtype(expr, vars, rules={}):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.determineexprtype', 'determineexprtype(expr, vars, rules={})', {'_ensure_exprdict': _ensure_exprdict, 'determineexprtype_re_1': determineexprtype_re_1, 'determineexprtype_re_2': determineexprtype_re_2, 'outmess': outmess, 'determineexprtype_re_3': determineexprtype_re_3, 'markoutercomma': markoutercomma, 'determineexprtype_re_4': determineexprtype_re_4, 'determineexprtype': determineexprtype, 'determineexprtype_re_5': determineexprtype_re_5, 'expr': expr, 'vars': vars, 'rules': rules}, 1)

def crack2fortrangen(block, tab='\n', as_interface=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.crack2fortrangen', "crack2fortrangen(block, tab='\n', as_interface=False)", {'setmesstext': setmesstext, 'skipfuncs': skipfuncs, 'onlyfuncs': onlyfuncs, 'crack2fortrangen': crack2fortrangen, 'expr2name': expr2name, 'isintent_callback': isintent_callback, 'tabchar': tabchar, 'use2fortran': use2fortran, 'common2fortran': common2fortran, 'vars2fortran': vars2fortran, 'block': block, 'tab': tab, 'as_interface': as_interface}, 1)

def common2fortran(common, tab=''):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.common2fortran', "common2fortran(common, tab='')", {'common': common, 'tab': tab}, 1)

def use2fortran(use, tab=''):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.use2fortran', "use2fortran(use, tab='')", {'use': use, 'tab': tab}, 1)

def true_intent_list(var):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.true_intent_list', 'true_intent_list(var)', {'var': var}, 1)

def vars2fortran(block, vars, args, tab='', as_interface=False):
    """
    TODO:
    public sub
    ...
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.vars2fortran', "vars2fortran(block, vars, args, tab='', as_interface=False)", {'setmesstext': setmesstext, 'errmess': errmess, 'isintent_callback': isintent_callback, 'isoptional': isoptional, 'show': show, 'outmess': outmess, 'true_intent_list': true_intent_list, 'block': block, 'vars': vars, 'args': args, 'tab': tab, 'as_interface': as_interface}, 1)
post_processing_hooks = []

def crackfortran(files):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.crackfortran', 'crackfortran(files)', {'outmess': outmess, 'readfortrancode': readfortrancode, 'crackline': crackline, 'postcrack': postcrack, 'grouplist': grouplist, 'post_processing_hooks': post_processing_hooks, 'traverse': traverse, 'postcrack2': postcrack2, 'files': files}, 1)

def crack2fortran(block):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.crack2fortran', 'crack2fortran(block)', {'crack2fortrangen': crack2fortrangen, 'f2py_version': f2py_version, 'block': block}, 1)

def _is_visit_pair(obj):
    return (isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], (int, str)))

def traverse(obj, visit, parents=[], result=None, *args, **kwargs):
    """Traverse f2py data structure with the following visit function:

    def visit(item, parents, result, *args, **kwargs):
        "

        parents is a list of key-"f2py data structure" pairs from which
        items are taken from.

        result is a f2py data structure that is filled with the
        return value of the visit function.

        item is 2-tuple (index, value) if parents[-1][1] is a list
        item is 2-tuple (key, value) if parents[-1][1] is a dict

        The return value of visit must be None, or of the same kind as
        item, that is, if parents[-1] is a list, the return value must
        be 2-tuple (new_index, new_value), or if parents[-1] is a
        dict, the return value must be 2-tuple (new_key, new_value).

        If new_index or new_value is None, the return value of visit
        is ignored, that is, it will not be added to the result.

        If the return value is None, the content of obj will be
        traversed, otherwise not.
        "
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.traverse', 'traverse(obj, visit, parents=[], result=None, *args, **kwargs)', {'_is_visit_pair': _is_visit_pair, 'traverse': traverse, 'obj': obj, 'visit': visit, 'parents': parents, 'result': result, 'args': args, 'kwargs': kwargs}, 1)

def character_backward_compatibility_hook(item, parents, result, *args, **kwargs):
    """Previously, Fortran character was incorrectly treated as
    character*1. This hook fixes the usage of the corresponding
    variables in `check`, `dimension`, `=`, and `callstatement`
    expressions.

    The usage of `char*` in `callprotoargument` expression can be left
    unchanged because C `character` is C typedef of `char`, although,
    new implementations should use `character*` in the corresponding
    expressions.

    See https://github.com/numpy/numpy/pull/19388 for more information.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.crackfortran.character_backward_compatibility_hook', 'character_backward_compatibility_hook(item, parents, result, *args, **kwargs)', {'re': re, 'ischaracter': ischaracter, 'outmess': outmess, 'item': item, 'parents': parents, 'result': result, 'args': args, 'kwargs': kwargs}, 1)
post_processing_hooks.append(character_backward_compatibility_hook)
if __name__ == '__main__':
    files = []
    funcs = []
    f = 1
    f2 = 0
    f3 = 0
    showblocklist = 0
    for l in sys.argv[1:]:
        if l == '':
            pass
        elif l[0] == ':':
            f = 0
        elif l == '-quiet':
            quiet = 1
            verbose = 0
        elif l == '-verbose':
            verbose = 2
            quiet = 0
        elif l == '-fix':
            if strictf77:
                outmess('Use option -f90 before -fix if Fortran 90 code is in fix form.\n', 0)
            skipemptyends = 1
            sourcecodeform = 'fix'
        elif l == '-skipemptyends':
            skipemptyends = 1
        elif l == '--ignore-contains':
            ignorecontains = 1
        elif l == '-f77':
            strictf77 = 1
            sourcecodeform = 'fix'
        elif l == '-f90':
            strictf77 = 0
            sourcecodeform = 'free'
            skipemptyends = 1
        elif l == '-h':
            f2 = 1
        elif l == '-show':
            showblocklist = 1
        elif l == '-m':
            f3 = 1
        elif l[0] == '-':
            errmess('Unknown option %s\n' % repr(l))
        elif f2:
            f2 = 0
            pyffilename = l
        elif f3:
            f3 = 0
            f77modulename = l
        elif f:
            try:
                open(l).close()
                files.append(l)
            except OSError as detail:
                errmess(f'OSError: {detail!s}\n')
        else:
            funcs.append(l)
    if (not strictf77 and f77modulename and not skipemptyends):
        outmess('  Warning: You have specified module name for non Fortran 77 code that\n  should not need one (expect if you are scanning F90 code for non\n  module blocks but then you should use flag -skipemptyends and also\n  be sure that the files do not contain programs without program\n  statement).\n', 0)
    postlist = crackfortran(files)
    if pyffilename:
        outmess('Writing fortran code to file %s\n' % repr(pyffilename), 0)
        pyf = crack2fortran(postlist)
        with open(pyffilename, 'w') as f:
            f.write(pyf)
    if showblocklist:
        show(postlist)

