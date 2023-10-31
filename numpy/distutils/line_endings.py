""" Functions for converting from DOS to UNIX line endings

"""

import os
import re
import sys

def dos2unix(file):
    """Replace CRLF with LF in argument files.  Print names of changed files."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.line_endings.dos2unix', 'dos2unix(file)', {'os': os, 're': re, 'file': file}, 1)

def dos2unix_one_dir(modified_files, dir_name, file_names):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.line_endings.dos2unix_one_dir', 'dos2unix_one_dir(modified_files, dir_name, file_names)', {'os': os, 'dos2unix': dos2unix, 'modified_files': modified_files, 'dir_name': dir_name, 'file_names': file_names}, 0)

def dos2unix_dir(dir_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.line_endings.dos2unix_dir', 'dos2unix_dir(dir_name)', {'os': os, 'dos2unix_one_dir': dos2unix_one_dir, 'dir_name': dir_name}, 1)

def unix2dos(file):
    """Replace LF with CRLF in argument files.  Print names of changed files."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.line_endings.unix2dos', 'unix2dos(file)', {'os': os, 're': re, 'file': file}, 1)

def unix2dos_one_dir(modified_files, dir_name, file_names):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.line_endings.unix2dos_one_dir', 'unix2dos_one_dir(modified_files, dir_name, file_names)', {'os': os, 'unix2dos': unix2dos, 'modified_files': modified_files, 'dir_name': dir_name, 'file_names': file_names}, 0)

def unix2dos_dir(dir_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.line_endings.unix2dos_dir', 'unix2dos_dir(dir_name)', {'os': os, 'unix2dos_one_dir': unix2dos_one_dir, 'dir_name': dir_name}, 1)
if __name__ == '__main__':
    dos2unix_dir(sys.argv[1])

