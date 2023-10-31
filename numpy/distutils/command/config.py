import os
import signal
import subprocess
import sys
import textwrap
import warnings
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist import check_gcc_function_attribute, check_gcc_function_attribute_with_intrinsics, check_gcc_variable_attribute, check_gcc_version_at_least, check_inline, check_restrict, check_compiler_gcc
LANG_EXT['f77'] = '.f'
LANG_EXT['f90'] = '.f90'


class config(old_config):
    old_config.user_options += [('fcompiler=', None, 'specify the Fortran compiler type')]
    
    def initialize_options(self):
        self.fcompiler = None
        old_config.initialize_options(self)
    
    def _check_compiler(self):
        old_config._check_compiler(self)
        from numpy.distutils.fcompiler import FCompiler, new_fcompiler
        if (sys.platform == 'win32' and self.compiler.compiler_type in ('msvc', 'intelw', 'intelemw')):
            if not self.compiler.initialized:
                try:
                    self.compiler.initialize()
                except IOError as e:
                    msg = textwrap.dedent('                        Could not initialize compiler instance: do you have Visual Studio\n                        installed?  If you are trying to build with MinGW, please use "python setup.py\n                        build -c mingw32" instead.  If you have Visual Studio installed, check it is\n                        correctly installed, and the right version (VS 2015 as of this writing).\n\n                        Original exception was: %s, and the Compiler class was %s\n                        ============================================================================') % (e, self.compiler.__class__.__name__)
                    print(textwrap.dedent('                        ============================================================================'))
                    raise distutils.errors.DistutilsPlatformError(msg) from e
            from distutils import msvc9compiler
            if msvc9compiler.get_build_version() >= 10:
                for ldflags in [self.compiler.ldflags_shared, self.compiler.ldflags_shared_debug]:
                    if '/MANIFEST' not in ldflags:
                        ldflags.append('/MANIFEST')
        if not isinstance(self.fcompiler, FCompiler):
            self.fcompiler = new_fcompiler(compiler=self.fcompiler, dry_run=self.dry_run, force=1, c_compiler=self.compiler)
            if self.fcompiler is not None:
                self.fcompiler.customize(self.distribution)
                if self.fcompiler.get_version():
                    self.fcompiler.customize_cmd(self)
                    self.fcompiler.show_customization()
    
    def _wrap_method(self, mth, lang, args):
        from distutils.ccompiler import CompileError
        from distutils.errors import DistutilsExecError
        save_compiler = self.compiler
        if lang in ['f77', 'f90']:
            self.compiler = self.fcompiler
        if self.compiler is None:
            raise CompileError('%s compiler is not set' % (lang, ))
        try:
            ret = mth(*(self, ) + args)
        except (DistutilsExecError, CompileError) as e:
            self.compiler = save_compiler
            raise CompileError from e
        self.compiler = save_compiler
        return ret
    
    def _compile(self, body, headers, include_dirs, lang):
        (src, obj) = self._wrap_method(old_config._compile, lang, (body, headers, include_dirs, lang))
        self.temp_files.append(obj + '.d')
        return (src, obj)
    
    def _link(self, body, headers, include_dirs, libraries, library_dirs, lang):
        if self.compiler.compiler_type == 'msvc':
            libraries = ((libraries or []))[:]
            library_dirs = ((library_dirs or []))[:]
            if lang in ['f77', 'f90']:
                lang = 'c'
                if self.fcompiler:
                    for d in (self.fcompiler.library_dirs or []):
                        if d.startswith('/usr/lib'):
                            try:
                                d = subprocess.check_output(['cygpath', '-w', d])
                            except (OSError, subprocess.CalledProcessError):
                                pass
                            else:
                                d = filepath_from_subprocess_output(d)
                        library_dirs.append(d)
                    for libname in (self.fcompiler.libraries or []):
                        if libname not in libraries:
                            libraries.append(libname)
            for libname in libraries:
                if libname.startswith('msvc'):
                    continue
                fileexists = False
                for libdir in (library_dirs or []):
                    libfile = os.path.join(libdir, '%s.lib' % libname)
                    if os.path.isfile(libfile):
                        fileexists = True
                        break
                if fileexists:
                    continue
                fileexists = False
                for libdir in library_dirs:
                    libfile = os.path.join(libdir, 'lib%s.a' % libname)
                    if os.path.isfile(libfile):
                        libfile2 = os.path.join(libdir, '%s.lib' % libname)
                        copy_file(libfile, libfile2)
                        self.temp_files.append(libfile2)
                        fileexists = True
                        break
                if fileexists:
                    continue
                log.warn('could not find library %r in directories %s' % (libname, library_dirs))
        elif self.compiler.compiler_type == 'mingw32':
            generate_manifest(self)
        return self._wrap_method(old_config._link, lang, (body, headers, include_dirs, libraries, library_dirs, lang))
    
    def check_header(self, header, include_dirs=None, library_dirs=None, lang='c'):
        self._check_compiler()
        return self.try_compile('/* we need a dummy line to make distutils happy */', [header], include_dirs)
    
    def check_decl(self, symbol, headers=None, include_dirs=None):
        self._check_compiler()
        body = textwrap.dedent('\n            int main(void)\n            {\n            #ifndef %s\n                (void) %s;\n            #endif\n                ;\n                return 0;\n            }') % (symbol, symbol)
        return self.try_compile(body, headers, include_dirs)
    
    def check_macro_true(self, symbol, headers=None, include_dirs=None):
        self._check_compiler()
        body = textwrap.dedent('\n            int main(void)\n            {\n            #if %s\n            #else\n            #error false or undefined macro\n            #endif\n                ;\n                return 0;\n            }') % (symbol, )
        return self.try_compile(body, headers, include_dirs)
    
    def check_type(self, type_name, headers=None, include_dirs=None, library_dirs=None):
        """Check type availability. Return True if the type can be compiled,
        False otherwise"""
        self._check_compiler()
        body = textwrap.dedent('\n            int main(void) {\n              if ((%(name)s *) 0)\n                return 0;\n              if (sizeof (%(name)s))\n                return 0;\n            }\n            ') % {'name': type_name}
        st = False
        try:
            try:
                self._compile(body % {'type': type_name}, headers, include_dirs, 'c')
                st = True
            except distutils.errors.CompileError:
                st = False
        finally:
            self._clean()
        return st
    
    def check_type_size(self, type_name, headers=None, include_dirs=None, library_dirs=None, expected=None):
        """Check size of a given type."""
        self._check_compiler()
        body = textwrap.dedent('\n            typedef %(type)s npy_check_sizeof_type;\n            int main (void)\n            {\n                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];\n                test_array [0] = 0\n\n                ;\n                return 0;\n            }\n            ')
        self._compile(body % {'type': type_name}, headers, include_dirs, 'c')
        self._clean()
        if expected:
            body = textwrap.dedent('\n                typedef %(type)s npy_check_sizeof_type;\n                int main (void)\n                {\n                    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)s)];\n                    test_array [0] = 0\n\n                    ;\n                    return 0;\n                }\n                ')
            for size in expected:
                try:
                    self._compile(body % {'type': type_name, 'size': size}, headers, include_dirs, 'c')
                    self._clean()
                    return size
                except CompileError:
                    pass
        body = textwrap.dedent('\n            typedef %(type)s npy_check_sizeof_type;\n            int main (void)\n            {\n                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)s)];\n                test_array [0] = 0\n\n                ;\n                return 0;\n            }\n            ')
        low = 0
        mid = 0
        while True:
            try:
                self._compile(body % {'type': type_name, 'size': mid}, headers, include_dirs, 'c')
                self._clean()
                break
            except CompileError:
                low = mid + 1
                mid = 2 * mid + 1
        high = mid
        while low != high:
            mid = (high - low) // 2 + low
            try:
                self._compile(body % {'type': type_name, 'size': mid}, headers, include_dirs, 'c')
                self._clean()
                high = mid
            except CompileError:
                low = mid + 1
        return low
    
    def check_func(self, func, headers=None, include_dirs=None, libraries=None, library_dirs=None, decl=False, call=False, call_args=None):
        self._check_compiler()
        body = []
        if decl:
            if type(decl) == str:
                body.append(decl)
            else:
                body.append('int %s (void);' % func)
        body.append('#ifdef _MSC_VER')
        body.append('#pragma function(%s)' % func)
        body.append('#endif')
        body.append('int main (void) {')
        if call:
            if call_args is None:
                call_args = ''
            body.append('  %s(%s);' % (func, call_args))
        else:
            body.append('  %s;' % func)
        body.append('  return 0;')
        body.append('}')
        body = '\n'.join(body) + '\n'
        return self.try_link(body, headers, include_dirs, libraries, library_dirs)
    
    def check_funcs_once(self, funcs, headers=None, include_dirs=None, libraries=None, library_dirs=None, decl=False, call=False, call_args=None):
        """Check a list of functions at once.

        This is useful to speed up things, since all the functions in the funcs
        list will be put in one compilation unit.

        Arguments
        ---------
        funcs : seq
            list of functions to test
        include_dirs : seq
            list of header paths
        libraries : seq
            list of libraries to link the code snippet to
        library_dirs : seq
            list of library paths
        decl : dict
            for every (key, value), the declaration in the value will be
            used for function in key. If a function is not in the
            dictionary, no declaration will be used.
        call : dict
            for every item (f, value), if the value is True, a call will be
            done to the function f.
        """
        self._check_compiler()
        body = []
        if decl:
            for (f, v) in decl.items():
                if v:
                    body.append('int %s (void);' % f)
        body.append('#ifdef _MSC_VER')
        for func in funcs:
            body.append('#pragma function(%s)' % func)
        body.append('#endif')
        body.append('int main (void) {')
        if call:
            for f in funcs:
                if (f in call and call[f]):
                    if not ((call_args and f in call_args and call_args[f])):
                        args = ''
                    else:
                        args = call_args[f]
                    body.append('  %s(%s);' % (f, args))
                else:
                    body.append('  %s;' % f)
        else:
            for f in funcs:
                body.append('  %s;' % f)
        body.append('  return 0;')
        body.append('}')
        body = '\n'.join(body) + '\n'
        return self.try_link(body, headers, include_dirs, libraries, library_dirs)
    
    def check_inline(self):
        """Return the inline keyword recognized by the compiler, empty string
        otherwise."""
        return check_inline(self)
    
    def check_restrict(self):
        """Return the restrict keyword recognized by the compiler, empty string
        otherwise."""
        return check_restrict(self)
    
    def check_compiler_gcc(self):
        """Return True if the C compiler is gcc"""
        return check_compiler_gcc(self)
    
    def check_gcc_function_attribute(self, attribute, name):
        return check_gcc_function_attribute(self, attribute, name)
    
    def check_gcc_function_attribute_with_intrinsics(self, attribute, name, code, include):
        return check_gcc_function_attribute_with_intrinsics(self, attribute, name, code, include)
    
    def check_gcc_variable_attribute(self, attribute):
        return check_gcc_variable_attribute(self, attribute)
    
    def check_gcc_version_at_least(self, major, minor=0, patchlevel=0):
        """Return True if the GCC version is greater than or equal to the
        specified version."""
        return check_gcc_version_at_least(self, major, minor, patchlevel)
    
    def get_output(self, body, headers=None, include_dirs=None, libraries=None, library_dirs=None, lang='c', use_tee=None):
        """Try to compile, link to an executable, and run a program
        built from 'body' and 'headers'. Returns the exit status code
        of the program and its output.
        """
        warnings.warn('\n+++++++++++++++++++++++++++++++++++++++++++++++++\nUsage of get_output is deprecated: please do not \nuse it anymore, and avoid configuration checks \ninvolving running executable on the target machine.\n+++++++++++++++++++++++++++++++++++++++++++++++++\n', DeprecationWarning, stacklevel=2)
        self._check_compiler()
        (exitcode, output) = (255, '')
        try:
            grabber = GrabStdout()
            try:
                (src, obj, exe) = self._link(body, headers, include_dirs, libraries, library_dirs, lang)
                grabber.restore()
            except Exception:
                output = grabber.data
                grabber.restore()
                raise
            exe = os.path.join('.', exe)
            try:
                output = subprocess.check_output([exe], cwd='.')
            except subprocess.CalledProcessError as exc:
                exitstatus = exc.returncode
                output = ''
            except OSError:
                exitstatus = 127
                output = ''
            else:
                output = filepath_from_subprocess_output(output)
            if hasattr(os, 'WEXITSTATUS'):
                exitcode = os.WEXITSTATUS(exitstatus)
                if os.WIFSIGNALED(exitstatus):
                    sig = os.WTERMSIG(exitstatus)
                    log.error('subprocess exited with signal %d' % (sig, ))
                    if sig == signal.SIGINT:
                        raise KeyboardInterrupt
            else:
                exitcode = exitstatus
            log.info('success!')
        except (CompileError, LinkError):
            log.info('failure.')
        self._clean()
        return (exitcode, output)



class GrabStdout:
    
    def __init__(self):
        self.sys_stdout = sys.stdout
        self.data = ''
        sys.stdout = self
    
    def write(self, data):
        self.sys_stdout.write(data)
        self.data += data
    
    def flush(self):
        self.sys_stdout.flush()
    
    def restore(self):
        sys.stdout = self.sys_stdout


