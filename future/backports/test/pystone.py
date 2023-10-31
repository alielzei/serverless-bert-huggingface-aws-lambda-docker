"""
"PYSTONE" Benchmark Program

Version:        Python/1.1 (corresponds to C/1.1 plus 2 Pystone fixes)

Author:         Reinhold P. Weicker,  CACM Vol 27, No 10, 10/84 pg. 1013.

                Translated from ADA to C by Rick Richardson.
                Every method to preserve ADA-likeness has been used,
                at the expense of C-ness.

                Translated from C to Python by Guido van Rossum.

Version History:

                Version 1.1 corrects two bugs in version 1.0:

                First, it leaked memory: in Proc1(), NextRecord ends
                up having a pointer to itself.  I have corrected this
                by zapping NextRecord.PtrComp at the end of Proc1().

                Second, Proc3() used the operator != to compare a
                record to None.  This is rather inefficient and not
                true to the intention of the original benchmark (where
                a pointer comparison to None is intended; the !=
                operator attempts to find a method __cmp__ to do value
                comparison of the record).  Version 1.1 runs 5-10
                percent faster than version 1.0, so benchmark figures
                of different versions can't be compared directly.

"""

from __future__ import print_function
from time import clock
LOOPS = 50000
__version__ = '1.1'
[Ident1, Ident2, Ident3, Ident4, Ident5] = range(1, 6)


class Record(object):
    
    def __init__(self, PtrComp=None, Discr=0, EnumComp=0, IntComp=0, StringComp=0):
        self.PtrComp = PtrComp
        self.Discr = Discr
        self.EnumComp = EnumComp
        self.IntComp = IntComp
        self.StringComp = StringComp
    
    def copy(self):
        return Record(self.PtrComp, self.Discr, self.EnumComp, self.IntComp, self.StringComp)

TRUE = 1
FALSE = 0

def main(loops=LOOPS):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.pystone.main', 'main(loops=LOOPS)', {'pystones': pystones, '__version__': __version__, 'loops': loops, 'LOOPS': LOOPS}, 0)

def pystones(loops=LOOPS):
    return Proc0(loops)
IntGlob = 0
BoolGlob = FALSE
Char1Glob = '\x00'
Char2Glob = '\x00'
Array1Glob = [0] * 51
Array2Glob = [x[:] for x in [Array1Glob] * 51]
PtrGlb = None
PtrGlbNext = None

def Proc0(loops=LOOPS):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc0', 'Proc0(loops=LOOPS)', {'clock': clock, 'Record': Record, 'Ident1': Ident1, 'Ident3': Ident3, 'Array2Glob': Array2Glob, 'Proc5': Proc5, 'Proc4': Proc4, 'Ident2': Ident2, 'Func2': Func2, 'Proc7': Proc7, 'Proc8': Proc8, 'Array1Glob': Array1Glob, 'Proc1': Proc1, 'Char2Glob': Char2Glob, 'Func1': Func1, 'Proc6': Proc6, 'Proc2': Proc2, 'loops': loops, 'LOOPS': LOOPS}, 2)

def Proc1(PtrParIn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc1', 'Proc1(PtrParIn)', {'PtrGlb': PtrGlb, 'Proc3': Proc3, 'Ident1': Ident1, 'Proc6': Proc6, 'Proc7': Proc7, 'PtrParIn': PtrParIn}, 1)

def Proc2(IntParIO):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc2', 'Proc2(IntParIO)', {'Char1Glob': Char1Glob, 'IntGlob': IntGlob, 'Ident1': Ident1, 'IntParIO': IntParIO}, 1)

def Proc3(PtrParOut):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc3', 'Proc3(PtrParOut)', {'PtrGlb': PtrGlb, 'Proc7': Proc7, 'PtrParOut': PtrParOut}, 1)

def Proc4():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc4', 'Proc4()', {'Char1Glob': Char1Glob, 'BoolGlob': BoolGlob}, 0)

def Proc5():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc5', 'Proc5()', {'FALSE': FALSE}, 0)

def Proc6(EnumParIn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc6', 'Proc6(EnumParIn)', {'Func3': Func3, 'Ident4': Ident4, 'Ident1': Ident1, 'Ident2': Ident2, 'IntGlob': IntGlob, 'Ident3': Ident3, 'Ident5': Ident5, 'EnumParIn': EnumParIn}, 1)

def Proc7(IntParI1, IntParI2):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc7', 'Proc7(IntParI1, IntParI2)', {'IntParI1': IntParI1, 'IntParI2': IntParI2}, 1)

def Proc8(Array1Par, Array2Par, IntParI1, IntParI2):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.pystone.Proc8', 'Proc8(Array1Par, Array2Par, IntParI1, IntParI2)', {'Array1Par': Array1Par, 'Array2Par': Array2Par, 'IntParI1': IntParI1, 'IntParI2': IntParI2}, 0)

def Func1(CharPar1, CharPar2):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Func1', 'Func1(CharPar1, CharPar2)', {'Ident1': Ident1, 'Ident2': Ident2, 'CharPar1': CharPar1, 'CharPar2': CharPar2}, 1)

def Func2(StrParI1, StrParI2):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Func2', 'Func2(StrParI1, StrParI2)', {'Func1': Func1, 'Ident1': Ident1, 'TRUE': TRUE, 'FALSE': FALSE, 'StrParI1': StrParI1, 'StrParI2': StrParI2}, 1)

def Func3(EnumParIn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.pystone.Func3', 'Func3(EnumParIn)', {'Ident3': Ident3, 'TRUE': TRUE, 'FALSE': FALSE, 'EnumParIn': EnumParIn}, 1)
if __name__ == '__main__':
    import sys
    
    def error(msg):
        import custom_funtemplate
        custom_funtemplate.rewrite_template('future.backports.test.pystone.error', 'error(msg)', {'sys': sys, 'msg': msg}, 0)
    nargs = len(sys.argv) - 1
    if nargs > 1:
        error('%d arguments are too many;' % nargs)
    elif nargs == 1:
        try:
            loops = int(sys.argv[1])
        except ValueError:
            error('Invalid argument %r;' % sys.argv[1])
    else:
        loops = LOOPS
    main(loops)

