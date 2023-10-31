"""

process_file(filename)

  takes templated file .xxx.src and produces .xxx file where .xxx
  is .pyf .f90 or .f using the following template rules:

  '<..>' denotes a template.

  All function and subroutine blocks in a source file with names that
  contain '<..>' will be replicated according to the rules in '<..>'.

  The number of comma-separated words in '<..>' will determine the number of
  replicates.

  '<..>' may have two different forms, named and short. For example,

  named:
   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with
   'd', 's', 'z', and 'c' for each replicate of the block.

   <_c>  is already defined: <_c=s,d,c,z>
   <_t>  is already defined: <_t=real,double precision,complex,double complex>

  short:
   <s,d,c,z>, a short form of the named, useful when no <p> appears inside
   a block.

  In general, '<..>' contains a comma separated list of arbitrary
  expressions. If these expression must contain a comma|leftarrow|rightarrow,
  then prepend the comma|leftarrow|rightarrow with a backslash.

  If an expression matches '\<index>' then it will be replaced
  by <index>-th expression.

  Note that all '<..>' forms in a block must have the same number of
  comma-separated entries.

 Predefined named template rules:
  <prefix=s,d,c,z>
  <ftype=real,double precision,complex,double complex>
  <ftypereal=real,double precision,
,>
  <ctype=float,double,complex_float,complex_double>
  <ctypereal=float,double,
,>

"""

__all__ = ['process_str', 'process_file']
import os
import sys
import re
routine_start_re = re.compile('(\\n|\\A)((     (\\$|\\*))|)\\s*(subroutine|function)\\b', re.I)
routine_end_re = re.compile('\\n\\s*end\\s*(subroutine|function)\\b.*(\\n|\\Z)', re.I)
function_start_re = re.compile('\\n     (\\$|\\*)\\s*function\\b', re.I)

def parse_structure(astr):
    """ Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.parse_structure', 'parse_structure(astr)', {'routine_start_re': routine_start_re, 'function_start_re': function_start_re, 'routine_end_re': routine_end_re, 'astr': astr}, 1)
template_re = re.compile('<\\s*(\\w[\\w\\d]*)\\s*>')
named_re = re.compile('<\\s*(\\w[\\w\\d]*)\\s*=\\s*(.*?)\\s*>')
list_re = re.compile('<\\s*((.*?))\\s*>')

def find_repl_patterns(astr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.find_repl_patterns', 'find_repl_patterns(astr)', {'named_re': named_re, 'unique_key': unique_key, 'conv': conv, 'astr': astr}, 1)

def find_and_remove_repl_patterns(astr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.find_and_remove_repl_patterns', 'find_and_remove_repl_patterns(astr)', {'find_repl_patterns': find_repl_patterns, 're': re, 'named_re': named_re, 'astr': astr}, 2)
item_re = re.compile('\\A\\\\(?P<index>\\d+)\\Z')

def conv(astr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.conv', 'conv(astr)', {'item_re': item_re, 'astr': astr}, 1)

def unique_key(adict):
    """ Obtain a unique key given a dictionary."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.unique_key', 'unique_key(adict)', {'adict': adict}, 1)
template_name_re = re.compile('\\A\\s*(\\w[\\w\\d]*)\\s*\\Z')

def expand_sub(substr, names):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.expand_sub', 'expand_sub(substr, names)', {'find_repl_patterns': find_repl_patterns, 'named_re': named_re, 'conv': conv, 'template_name_re': template_name_re, 'unique_key': unique_key, 'list_re': list_re, 'template_re': template_re, 'substr': substr, 'names': names}, 1)

def process_str(allstr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.process_str', 'process_str(allstr)', {'parse_structure': parse_structure, '_special_names': _special_names, 'find_and_remove_repl_patterns': find_and_remove_repl_patterns, 'expand_sub': expand_sub, 'allstr': allstr}, 1)
include_src_re = re.compile('(\\n|\\A)\\s*include\\s*[\'\\"](?P<name>[\\w\\d./\\\\]+\\.src)[\'\\"]', re.I)

def resolve_includes(source):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.from_template.resolve_includes', 'resolve_includes(source)', {'os': os, 'include_src_re': include_src_re, 'resolve_includes': resolve_includes, 'source': source}, 1)

def process_file(source):
    lines = resolve_includes(source)
    return process_str(''.join(lines))
_special_names = find_repl_patterns('\n<_c=s,d,c,z>\n<_t=real,double precision,complex,double complex>\n<prefix=s,d,c,z>\n<ftype=real,double precision,complex,double complex>\n<ctype=float,double,complex_float,complex_double>\n<ftypereal=real,double precision,\\0,\\1>\n<ctypereal=float,double,\\0,\\1>\n')

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.from_template.main', 'main()', {'sys': sys, 'os': os, 'process_str': process_str}, 0)
if __name__ == '__main__':
    main()

