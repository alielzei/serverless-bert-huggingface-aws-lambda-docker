import os

def parse_distributions_h(ffi, inc_dir):
    """
    Parse distributions.h located in inc_dir for CFFI, filling in the ffi.cdef

    Read the function declarations without the "#define ..." macros that will
    be filled in when loading the library.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.random._examples.cffi.parse.parse_distributions_h', 'parse_distributions_h(ffi, inc_dir)', {'os': os, 'ffi': ffi, 'inc_dir': inc_dir}, 0)

