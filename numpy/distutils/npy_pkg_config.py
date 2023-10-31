import sys
import re
import os
from configparser import RawConfigParser
__all__ = ['FormatError', 'PkgNotFound', 'LibraryInfo', 'VariableSet', 'read_config', 'parse_flags']
_VAR = re.compile('\\$\\{([a-zA-Z0-9_-]+)\\}')


class FormatError(OSError):
    """
    Exception thrown when there is a problem parsing a configuration file.

    """
    
    def __init__(self, msg):
        self.msg = msg
    
    def __str__(self):
        return self.msg



class PkgNotFound(OSError):
    """Exception raised when a package can not be located."""
    
    def __init__(self, msg):
        self.msg = msg
    
    def __str__(self):
        return self.msg


def parse_flags(line):
    """
    Parse a line from a config file containing compile flags.

    Parameters
    ----------
    line : str
        A single line containing one or more compile flags.

    Returns
    -------
    d : dict
        Dictionary of parsed flags, split into relevant categories.
        These categories are the keys of `d`:

        * 'include_dirs'
        * 'library_dirs'
        * 'libraries'
        * 'macros'
        * 'ignored'

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.npy_pkg_config.parse_flags', 'parse_flags(line)', {'line': line}, 1)

def _escape_backslash(val):
    return val.replace('\\', '\\\\')


class LibraryInfo:
    """
    Object containing build information about a library.

    Parameters
    ----------
    name : str
        The library name.
    description : str
        Description of the library.
    version : str
        Version string.
    sections : dict
        The sections of the configuration file for the library. The keys are
        the section headers, the values the text under each header.
    vars : class instance
        A `VariableSet` instance, which contains ``(name, value)`` pairs for
        variables defined in the configuration file for the library.
    requires : sequence, optional
        The required libraries for the library to be installed.

    Notes
    -----
    All input parameters (except "sections" which is a method) are available as
    attributes of the same name.

    """
    
    def __init__(self, name, description, version, sections, vars, requires=None):
        self.name = name
        self.description = description
        if requires:
            self.requires = requires
        else:
            self.requires = []
        self.version = version
        self._sections = sections
        self.vars = vars
    
    def sections(self):
        """
        Return the section headers of the config file.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list of str
            The list of section headers.

        """
        return list(self._sections.keys())
    
    def cflags(self, section='default'):
        val = self.vars.interpolate(self._sections[section]['cflags'])
        return _escape_backslash(val)
    
    def libs(self, section='default'):
        val = self.vars.interpolate(self._sections[section]['libs'])
        return _escape_backslash(val)
    
    def __str__(self):
        m = ['Name: %s' % self.name, 'Description: %s' % self.description]
        if self.requires:
            m.append('Requires:')
        else:
            m.append('Requires: %s' % ','.join(self.requires))
        m.append('Version: %s' % self.version)
        return '\n'.join(m)



class VariableSet:
    """
    Container object for the variables defined in a config file.

    `VariableSet` can be used as a plain dictionary, with the variable names
    as keys.

    Parameters
    ----------
    d : dict
        Dict of items in the "variables" section of the configuration file.

    """
    
    def __init__(self, d):
        self._raw_data = dict([(k, v) for (k, v) in d.items()])
        self._re = {}
        self._re_sub = {}
        self._init_parse()
    
    def _init_parse(self):
        for (k, v) in self._raw_data.items():
            self._init_parse_var(k, v)
    
    def _init_parse_var(self, name, value):
        self._re[name] = re.compile('\\$\\{%s\\}' % name)
        self._re_sub[name] = value
    
    def interpolate(self, value):
        
        def _interpolate(value):
            for k in self._re.keys():
                value = self._re[k].sub(self._re_sub[k], value)
            return value
        while _VAR.search(value):
            nvalue = _interpolate(value)
            if nvalue == value:
                break
            value = nvalue
        return value
    
    def variables(self):
        """
        Return the list of variable names.

        Parameters
        ----------
        None

        Returns
        -------
        names : list of str
            The names of all variables in the `VariableSet` instance.

        """
        return list(self._raw_data.keys())
    
    def __getitem__(self, name):
        return self._raw_data[name]
    
    def __setitem__(self, name, value):
        self._raw_data[name] = value
        self._init_parse_var(name, value)


def parse_meta(config):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.npy_pkg_config.parse_meta', 'parse_meta(config)', {'FormatError': FormatError, 'config': config}, 1)

def parse_variables(config):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.npy_pkg_config.parse_variables', 'parse_variables(config)', {'FormatError': FormatError, 'VariableSet': VariableSet, 'config': config}, 1)

def parse_sections(config):
    return (meta_d, r)

def pkg_to_filename(pkg_name):
    return '%s.ini' % pkg_name

def parse_config(filename, dirs=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.npy_pkg_config.parse_config', 'parse_config(filename, dirs=None)', {'os': os, 'RawConfigParser': RawConfigParser, 'PkgNotFound': PkgNotFound, 'parse_meta': parse_meta, '_escape_backslash': _escape_backslash, 'filename': filename, 'dirs': dirs}, 4)

def _read_config_imp(filenames, dirs=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.npy_pkg_config._read_config_imp', '_read_config_imp(filenames, dirs=None)', {'parse_config': parse_config, 'pkg_to_filename': pkg_to_filename, 'sys': sys, '_escape_backslash': _escape_backslash, 'os': os, 'LibraryInfo': LibraryInfo, 'VariableSet': VariableSet, 'filenames': filenames, 'dirs': dirs}, 4)
_CACHE = {}

def read_config(pkgname, dirs=None):
    """
    Return library info for a package from its configuration file.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of directories - usually including
        the NumPy base directory - where to look for npy-pkg-config files.

    Returns
    -------
    pkginfo : class instance
        The `LibraryInfo` instance containing the build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    misc_util.get_info, misc_util.get_pkg_info

    Examples
    --------
    >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')
    >>> type(npymath_info)
    <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>
    >>> print(npymath_info)
    Name: npymath
    Description: Portable, core math library implementing C99 standard
    Requires:
    Version: 0.1  #random

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.npy_pkg_config.read_config', 'read_config(pkgname, dirs=None)', {'_CACHE': _CACHE, '_read_config_imp': _read_config_imp, 'pkg_to_filename': pkg_to_filename, 'pkgname': pkgname, 'dirs': dirs}, 1)
if __name__ == '__main__':
    from optparse import OptionParser
    import glob
    parser = OptionParser()
    parser.add_option('--cflags', dest='cflags', action='store_true', help='output all preprocessor and compiler flags')
    parser.add_option('--libs', dest='libs', action='store_true', help='output all linker flags')
    parser.add_option('--use-section', dest='section', help='use this section instead of default for options')
    parser.add_option('--version', dest='version', action='store_true', help='output version')
    parser.add_option('--atleast-version', dest='min_version', help='Minimal version')
    parser.add_option('--list-all', dest='list_all', action='store_true', help='Minimal version')
    parser.add_option('--define-variable', dest='define_variable', help='Replace variable with the given value')
    (options, args) = parser.parse_args(sys.argv)
    if len(args) < 2:
        raise ValueError('Expect package name on the command line:')
    if options.list_all:
        files = glob.glob('*.ini')
        for f in files:
            info = read_config(f)
            print('%s\t%s - %s' % (info.name, info.name, info.description))
    pkg_name = args[1]
    d = os.environ.get('NPY_PKG_CONFIG_PATH')
    if d:
        info = read_config(pkg_name, ['numpy/core/lib/npy-pkg-config', '.', d])
    else:
        info = read_config(pkg_name, ['numpy/core/lib/npy-pkg-config', '.'])
    if options.section:
        section = options.section
    else:
        section = 'default'
    if options.define_variable:
        m = re.search('([\\S]+)=([\\S]+)', options.define_variable)
        if not m:
            raise ValueError('--define-variable option should be of the form --define-variable=foo=bar')
        else:
            name = m.group(1)
            value = m.group(2)
        info.vars[name] = value
    if options.cflags:
        print(info.cflags(section))
    if options.libs:
        print(info.libs(section))
    if options.version:
        print(info.version)
    if options.min_version:
        print(info.version >= options.min_version)

