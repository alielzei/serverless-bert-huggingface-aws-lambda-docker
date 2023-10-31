from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import glob
import imp
import os
import re
import setuptools
import subprocess
import sys
import sysconfig
import tempfile
import warnings
import collections
import torch
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from setuptools.command.build_ext import build_ext
IS_WINDOWS = sys.platform == 'win32'

def _find_cuda_home():
    """Finds the CUDA install path."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._find_cuda_home', '_find_cuda_home()', {'os': os, 'IS_WINDOWS': IS_WINDOWS, 'subprocess': subprocess, 'glob': glob, 'torch': torch}, 1)

def _find_rocm_home():
    """Finds the ROCm install path."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._find_rocm_home', '_find_rocm_home()', {'os': os, 'subprocess': subprocess, 'torch': torch}, 1)

def _join_rocm_home(*paths):
    """
    Joins paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._join_rocm_home', '_join_rocm_home(*paths)', {'ROCM_HOME': ROCM_HOME, 'EnvironmentError': EnvironmentError, 'IS_WINDOWS': IS_WINDOWS, 'os': os, 'paths': paths}, 1)
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)
ABI_INCOMPATIBILITY_WARNING = '\n\n                               !! WARNING !!\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nYour compiler ({}) may be ABI-incompatible with PyTorch!\nPlease use a compiler that is ABI-compatible with GCC 5.0 and above.\nSee https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.\n\nSee https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6\nfor instructions on how to install GCC 5 or higher.\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n                              !! WARNING !!\n'
WRONG_COMPILER_WARNING = '\n\n                               !! WARNING !!\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nYour compiler ({user_compiler}) is not compatible with the compiler Pytorch was\nbuilt with for this platform, which is {pytorch_compiler} on {platform}. Please\nuse {pytorch_compiler} to to compile your extension. Alternatively, you may\ncompile PyTorch from source using {user_compiler}, and then you can also use\n{user_compiler} to compile your extension.\n\nSee https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help\nwith compiling PyTorch from source.\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n                              !! WARNING !!\n'
ROCM_HOME = _find_rocm_home()
MIOPEN_HOME = (_join_rocm_home('miopen') if ROCM_HOME else None)
IS_HIP_EXTENSION = (True if (ROCM_HOME is not None and torch.version.hip is not None) else False)
CUDA_HOME = _find_cuda_home()
CUDNN_HOME = (os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH'))
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile('\\d+\\.\\d+\\.\\d+\\w+\\+\\w+')
COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/EHsc']
COMMON_NVCC_FLAGS = ['-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__', '--expt-relaxed-constexpr']
COMMON_HIPCC_FLAGS = ['-fPIC', '-D__HIP_PLATFORM_HCC__=1', '-DCUDA_HAS_FP16=1', '-D__HIP_NO_HALF_OPERATORS__=1', '-D__HIP_NO_HALF_CONVERSIONS__=1']
JIT_EXTENSION_VERSIONER = ExtensionVersioner()

def _is_binary_build():
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)

def _accepted_compilers_for_platform():
    return (['clang++', 'clang'] if sys.platform.startswith('darwin') else ['g++', 'gcc', 'gnu-c++', 'gnu-cc'])

def get_default_build_root():
    """
    Returns the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.
    """
    return os.path.realpath(os.path.join(tempfile.gettempdir(), 'torch_extensions'))

def check_compiler_ok_for_platform(compiler):
    """
    Verifies that the compiler is the expected one for the current platform.

    Arguments:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension.check_compiler_ok_for_platform', 'check_compiler_ok_for_platform(compiler)', {'IS_WINDOWS': IS_WINDOWS, 'subprocess': subprocess, 'os': os, '_accepted_compilers_for_platform': _accepted_compilers_for_platform, 'compiler': compiler}, 1)

def check_compiler_abi_compatibility(compiler):
    """
    Verifies that the given compiler is ABI-compatible with PyTorch.

    Arguments:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        False if the compiler is (likely) ABI-incompatible with PyTorch,
        else True.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension.check_compiler_abi_compatibility', 'check_compiler_abi_compatibility(compiler)', {'_is_binary_build': _is_binary_build, 'os': os, 'check_compiler_ok_for_platform': check_compiler_ok_for_platform, 'warnings': warnings, 'WRONG_COMPILER_WARNING': WRONG_COMPILER_WARNING, '_accepted_compilers_for_platform': _accepted_compilers_for_platform, 'sys': sys, 'MINIMUM_GCC_VERSION': MINIMUM_GCC_VERSION, 'subprocess': subprocess, 'MINIMUM_MSVC_VERSION': MINIMUM_MSVC_VERSION, 're': re, 'ABI_INCOMPATIBILITY_WARNING': ABI_INCOMPATIBILITY_WARNING, 'compiler': compiler}, 1)


class BuildExtension(build_ext, object):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++14``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``nvcc``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """
    
    @classmethod
    def with_options(cls, **options):
        """
        Returns an alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        """
        
        def init_with_options(*args, **kwargs):
            kwargs = kwargs.copy()
            kwargs.update(options)
            return cls(*args, **kwargs)
        return init_with_options
    
    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get('no_python_abi_suffix', False)
        self.use_ninja = kwargs.get('use_ninja', (False if IS_HIP_EXTENSION else True))
        if self.use_ninja:
            msg = 'Attempted to use ninja as the BuildExtension backend but {}. Falling back to using the slow distutils backend.'
            if IS_HIP_EXTENSION:
                warnings.warn(msg.format('HIP extensions is not supported yet for ninja.'))
                self.use_ninja = False
            elif not _is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False
    
    def build_extensions(self):
        self._check_abi()
        for extension in self.extensions:
            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile
        
        def append_std14_if_no_std_present(cflags):
            cpp_format_prefix = ('/{}:' if self.compiler.compiler_type == 'msvc' else '-{}=')
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++14'
            if not any((flag.startswith(cpp_flag_prefix) for flag in cflags)):
                cflags.append(cpp_flag)
        
        def unix_cuda_flags(cflags):
            return COMMON_NVCC_FLAGS + ['--compiler-options', "'-fPIC'"] + cflags + _get_cuda_arch_flags(cflags)
        
        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = (_join_rocm_home('bin', 'hipcc') if IS_HIP_EXTENSION else _join_cuda_home('bin', 'nvcc'))
                    if not isinstance(nvcc, list):
                        nvcc = [nvcc]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    if IS_HIP_EXTENSION:
                        cflags = cflags + _get_rocm_arch_flags(cflags)
                    else:
                        cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                if IS_HIP_EXTENSION:
                    cflags = cflags + COMMON_HIPCC_FLAGS
                append_std14_if_no_std_present(cflags)
                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                self.compiler.set_executable('compiler_so', original_compiler)
        
        def unix_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            """Compiles sources by outputting a ninja file and running it."""
            output_dir = os.path.abspath(output_dir)
            (_, objects, extra_postargs, pp_opts, _) = self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std14_if_no_std_present(post_cflags)
            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std14_if_no_std_present(cuda_post_cflags)
            _write_ninja_file_and_compile_objects(sources=sources, objects=objects, cflags=extra_cc_cflags + common_cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags, cuda_post_cflags=cuda_post_cflags, build_directory=output_dir, verbose=True, with_cuda=with_cuda)
            return objects
        
        def win_cuda_flags(cflags):
            return COMMON_NVCC_FLAGS + cflags + _get_cuda_arch_flags(cflags)
        
        def win_wrap_single_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None
            
            def spawn(cmd):
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [m.group(2) for m in (src_regex.match(elem) for elem in cmd) if m]
                obj_regex = re.compile('/Fo(.*)')
                obj_list = [m.group(1) for m in (obj_regex.match(elem) for elem in cmd) if m]
                include_regex = re.compile('((\\-|\\/)I.*)')
                include_list = [m.group(1) for m in (include_regex.match(elem) for elem in cmd) if m]
                if (len(src_list) >= 1 and len(obj_list) >= 1):
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []
                        cflags = win_cuda_flags(cflags)
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        cmd = [nvcc, '-c', src, '-o', obj] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        cmd += cflags
                return original_spawn(cmd)
            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros, include_dirs, debug, extra_preargs, extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn
        
        def win_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            if not self.compiler.initialized:
                self.compiler.initialize()
            output_dir = os.path.abspath(output_dir)
            (_, objects, extra_postargs, pp_opts, _) = self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            common_cflags = (extra_preargs or [])
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            common_cflags.extend(COMMON_MSVC_FLAGS)
            cflags = cflags + common_cflags + pp_opts
            with_cuda = any(map(_is_cuda_file, sources))
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std14_if_no_std_present(post_cflags)
            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = []
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = win_cuda_flags(cuda_post_cflags)
            from distutils.spawn import _nt_quote_args
            cflags = _nt_quote_args(cflags)
            post_cflags = _nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = _nt_quote_args(cuda_cflags)
                cuda_post_cflags = _nt_quote_args(cuda_post_cflags)
            _write_ninja_file_and_compile_objects(sources=sources, objects=objects, cflags=cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags, cuda_post_cflags=cuda_post_cflags, build_directory=output_dir, verbose=True, with_cuda=with_cuda)
            return objects
        if self.compiler.compiler_type == 'msvc':
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        elif self.use_ninja:
            self.compiler.compile = unix_wrap_ninja_compile
        else:
            self.compiler._compile = unix_wrap_single_compile
        build_ext.build_extensions(self)
    
    def get_ext_filename(self, ext_name):
        ext_filename = super(BuildExtension, self).get_ext_filename(ext_name)
        if (self.no_python_abi_suffix and sys.version_info >= (3, 0)):
            ext_filename_parts = ext_filename.split('.')
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename
    
    def _check_abi(self):
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        elif IS_WINDOWS:
            compiler = os.environ.get('CXX', 'cl')
        else:
            compiler = os.environ.get('CXX', 'c++')
        check_compiler_abi_compatibility(compiler)
    
    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)
    
    def _define_torch_extension_name(self, extension):
        names = extension.name.split('.')
        name = names[-1]
        define = '-DTORCH_EXTENSION_NAME={}'.format(name)
        self._add_compile_flag(extension, define)
    
    def _add_gnu_cpp_abi_flag(self, extension):
        self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))


def CppExtension(name, sources, *args, **kwargs):
    """
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
                name='extension',
                ext_modules=[
                    CppExtension(
                        name='extension',
                        sources=['extension.cpp'],
                        extra_compile_args=['-g']),
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension.CppExtension', 'CppExtension(name, sources, *args, **kwargs)', {'include_paths': include_paths, 'library_paths': library_paths, 'setuptools': setuptools, 'name': name, 'sources': sources, 'args': args, 'kwargs': kwargs}, 1)

def CUDAExtension(name, sources, *args, **kwargs):
    """
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        >>> setup(
                name='cuda_extension',
                ext_modules=[
                    CUDAExtension(
                            name='cuda_extension',
                            sources=['extension.cpp', 'extension_kernel.cu'],
                            extra_compile_args={'cxx': ['-g'],
                                                'nvcc': ['-O2']})
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension.CUDAExtension', 'CUDAExtension(name, sources, *args, **kwargs)', {'library_paths': library_paths, 'IS_HIP_EXTENSION': IS_HIP_EXTENSION, 'include_paths': include_paths, 'setuptools': setuptools, 'name': name, 'sources': sources, 'args': args, 'kwargs': kwargs}, 1)

def include_paths(cuda=False):
    """
    Get the include paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific include paths.

    Returns:
        A list of include path strings.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension.include_paths', 'include_paths(cuda=False)', {'os': os, '__file__': __file__, 'IS_HIP_EXTENSION': IS_HIP_EXTENSION, '_join_rocm_home': _join_rocm_home, 'MIOPEN_HOME': MIOPEN_HOME, '_join_cuda_home': _join_cuda_home, 'CUDNN_HOME': CUDNN_HOME, 'cuda': cuda}, 1)

def library_paths(cuda=False):
    """
    Get the library paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific library paths.

    Returns:
        A list of library path strings.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension.library_paths', 'library_paths(cuda=False)', {'os': os, '__file__': __file__, 'IS_HIP_EXTENSION': IS_HIP_EXTENSION, '_join_rocm_home': _join_rocm_home, 'IS_WINDOWS': IS_WINDOWS, '_join_cuda_home': _join_cuda_home, 'CUDNN_HOME': CUDNN_HOME, 'cuda': cuda}, 1)

def load(name, sources, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True):
    """
    Loads a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    CUDA support with mixed compilation is provided. Simply pass CUDA source
    files (``.cu`` or ``.cuh``) along with other sources. Such files will be
    detected and compiled with nvcc rather than the C++ compiler. This includes
    passing the CUDA lib64 directory as a library directory, and linking
    ``cudart``. You can pass additional flags to nvcc via
    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the CUDA install directory are used, which usually
    work fine. If not, setting the ``CUDA_HOME`` environment variable is the
    safest option.

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_cuda_cflags: optional list of compiler flags to forward to nvcc
            when building CUDA sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.cu`` or
            ``.cuh`` in ``sources``. Set it to `True`` to force CUDA headers
            and libraries to be included.
        is_python_module: If ``True`` (default), imports the produced shared
            library as a Python module. If ``False``, loads it into the process
            as a plain dynamic library.

    Returns:
        If ``is_python_module`` is ``True``, returns the loaded PyTorch
        extension as a Python module. If ``is_python_module`` is ``False``
        returns nothing (the shared library is loaded into the process as a side
        effect).

    Example:
        >>> from torch.utils.cpp_extension import load
        >>> module = load(
                name='extension',
                sources=['extension.cpp', 'extension_kernel.cu'],
                extra_cflags=['-O2'],
                verbose=True)
    """
    return _jit_compile(name, ([sources] if isinstance(sources, str) else sources), extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, (build_directory or _get_build_directory(name, verbose)), verbose, with_cuda, is_python_module)

def load_inline(name, cpp_sources, cuda_sources=None, functions=None, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True, with_pytorch_error_handling=True):
    """
    Loads a PyTorch C++ extension just-in-time (JIT) from string sources.

    This function behaves exactly like :func:`load`, but takes its sources as
    strings rather than filenames. These strings are stored to files in the
    build directory, after which the behavior of :func:`load_inline` is
    identical to :func:`load`.

    See `the
    tests <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions.py>`_
    for good examples of using this function.

    Sources may omit two required parts of a typical non-inline C++ extension:
    the necessary header includes, as well as the (pybind11) binding code. More
    precisely, strings passed to ``cpp_sources`` are first concatenated into a
    single ``.cpp`` file. This file is then prepended with ``#include
    <torch/extension.h>``.

    Furthermore, if the ``functions`` argument is supplied, bindings will be
    automatically generated for each function specified. ``functions`` can
    either be a list of function names, or a dictionary mapping from function
    names to docstrings. If a list is given, the name of each function is used
    as its docstring.

    The sources in ``cuda_sources`` are concatenated into a separate ``.cu``
    file and  prepended with ``torch/types.h``, ``cuda.h`` and
    ``cuda_runtime.h`` includes. The ``.cpp`` and ``.cu`` files are compiled
    separately, but ultimately linked into a single library. Note that no
    bindings are generated for functions in ``cuda_sources`` per  se. To bind
    to a CUDA kernel, you must create a C++ function that calls it, and either
    declare or define this C++ function in one of the ``cpp_sources`` (and
    include its name in ``functions``).

    See :func:`load` for a description of arguments omitted below.

    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        cuda_sources: A string, or list of strings, containing CUDA source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``cuda_sources`` is
            provided. Set it to ``True`` to force CUDA headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.

    Example:
        >>> from torch.utils.cpp_extension import load_inline
        >>> source = '''
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        '''
        >>> module = load_inline(name='inline_extension',
                                 cpp_sources=[source],
                                 functions=['sin_add'])

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension.load_inline', 'load_inline(name, cpp_sources, cuda_sources=None, functions=None, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True, with_pytorch_error_handling=True)', {'_get_build_directory': _get_build_directory, 'os': os, '_jit_compile': _jit_compile, 'name': name, 'cpp_sources': cpp_sources, 'cuda_sources': cuda_sources, 'functions': functions, 'extra_cflags': extra_cflags, 'extra_cuda_cflags': extra_cuda_cflags, 'extra_ldflags': extra_ldflags, 'extra_include_paths': extra_include_paths, 'build_directory': build_directory, 'verbose': verbose, 'with_cuda': with_cuda, 'is_python_module': is_python_module, 'with_pytorch_error_handling': with_pytorch_error_handling}, 1)

def _jit_compile(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory, verbose, with_cuda, is_python_module):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._jit_compile', '_jit_compile(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory, verbose, with_cuda, is_python_module)', {'JIT_EXTENSION_VERSIONER': JIT_EXTENSION_VERSIONER, 'FileBaton': FileBaton, 'os': os, '_write_ninja_file_and_build_library': _write_ninja_file_and_build_library, '_import_module_from_library': _import_module_from_library, 'name': name, 'sources': sources, 'extra_cflags': extra_cflags, 'extra_cuda_cflags': extra_cuda_cflags, 'extra_ldflags': extra_ldflags, 'extra_include_paths': extra_include_paths, 'build_directory': build_directory, 'verbose': verbose, 'with_cuda': with_cuda, 'is_python_module': is_python_module}, 1)

def _write_ninja_file_and_compile_objects(sources, objects, cflags, post_cflags, cuda_cflags, cuda_post_cflags, build_directory, verbose, with_cuda):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.cpp_extension._write_ninja_file_and_compile_objects', '_write_ninja_file_and_compile_objects(sources, objects, cflags, post_cflags, cuda_cflags, cuda_post_cflags, build_directory, verbose, with_cuda)', {'verify_ninja_availability': verify_ninja_availability, 'IS_WINDOWS': IS_WINDOWS, 'os': os, 'check_compiler_abi_compatibility': check_compiler_abi_compatibility, '_is_cuda_file': _is_cuda_file, '_write_ninja_file': _write_ninja_file, '_run_ninja_build': _run_ninja_build, 'sources': sources, 'objects': objects, 'cflags': cflags, 'post_cflags': post_cflags, 'cuda_cflags': cuda_cflags, 'cuda_post_cflags': cuda_post_cflags, 'build_directory': build_directory, 'verbose': verbose, 'with_cuda': with_cuda}, 0)

def _write_ninja_file_and_build_library(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory, verbose, with_cuda):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.cpp_extension._write_ninja_file_and_build_library', '_write_ninja_file_and_build_library(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory, verbose, with_cuda)', {'verify_ninja_availability': verify_ninja_availability, 'IS_WINDOWS': IS_WINDOWS, 'os': os, 'check_compiler_abi_compatibility': check_compiler_abi_compatibility, '_is_cuda_file': _is_cuda_file, '_prepare_ldflags': _prepare_ldflags, '_write_ninja_file_to_build_library': _write_ninja_file_to_build_library, '_run_ninja_build': _run_ninja_build, 'name': name, 'sources': sources, 'extra_cflags': extra_cflags, 'extra_cuda_cflags': extra_cuda_cflags, 'extra_ldflags': extra_ldflags, 'extra_include_paths': extra_include_paths, 'build_directory': build_directory, 'verbose': verbose, 'with_cuda': with_cuda}, 0)

def _is_ninja_available():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._is_ninja_available', '_is_ninja_available()', {'os': os, 'subprocess': subprocess}, 1)

def verify_ninja_availability():
    """
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system.
    """
    if not _is_ninja_available():
        raise RuntimeError('Ninja is required to load C++ extensions')

def _prepare_ldflags(extra_ldflags, with_cuda, verbose):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._prepare_ldflags', '_prepare_ldflags(extra_ldflags, with_cuda, verbose)', {'os': os, '__file__': __file__, 'IS_WINDOWS': IS_WINDOWS, 'sys': sys, '_join_cuda_home': _join_cuda_home, 'CUDNN_HOME': CUDNN_HOME, 'extra_ldflags': extra_ldflags, 'with_cuda': with_cuda, 'verbose': verbose}, 1)

def _get_cuda_arch_flags(cflags=None):
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._get_cuda_arch_flags', '_get_cuda_arch_flags(cflags=None)', {'collections': collections, 'os': os, 'torch': torch, 'cflags': cflags}, 1)

def _get_rocm_arch_flags(cflags=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._get_rocm_arch_flags', '_get_rocm_arch_flags(cflags=None)', {'cflags': cflags}, 1)

def _get_build_directory(name, verbose):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._get_build_directory', '_get_build_directory(name, verbose)', {'os': os, 'get_default_build_root': get_default_build_root, 'name': name, 'verbose': verbose}, 1)

def _get_num_workers(verbose):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._get_num_workers', '_get_num_workers(verbose)', {'os': os, 'verbose': verbose}, 1)

def _run_ninja_build(build_directory, verbose, error_prefix):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.cpp_extension._run_ninja_build', '_run_ninja_build(build_directory, verbose, error_prefix)', {'_get_num_workers': _get_num_workers, 'sys': sys, 'subprocess': subprocess, 'build_directory': build_directory, 'verbose': verbose, 'error_prefix': error_prefix}, 0)

def _import_module_from_library(module_name, path, is_python_module):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._import_module_from_library', '_import_module_from_library(module_name, path, is_python_module)', {'imp': imp, 'torch': torch, 'module_name': module_name, 'path': path, 'is_python_module': is_python_module}, 1)

def _write_ninja_file_to_build_library(path, name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, with_cuda):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._write_ninja_file_to_build_library', '_write_ninja_file_to_build_library(path, name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, with_cuda)', {'os': os, 'include_paths': include_paths, 'sysconfig': sysconfig, 'IS_WINDOWS': IS_WINDOWS, 'torch': torch, 'COMMON_MSVC_FLAGS': COMMON_MSVC_FLAGS, 'COMMON_NVCC_FLAGS': COMMON_NVCC_FLAGS, '_get_cuda_arch_flags': _get_cuda_arch_flags, '_is_cuda_file': _is_cuda_file, 'sys': sys, '_write_ninja_file': _write_ninja_file, 'path': path, 'name': name, 'sources': sources, 'extra_cflags': extra_cflags, 'extra_cuda_cflags': extra_cuda_cflags, 'extra_ldflags': extra_ldflags, 'extra_include_paths': extra_include_paths, 'with_cuda': with_cuda}, 1)

def _write_ninja_file(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags, sources, objects, ldflags, library_target, with_cuda):
    """Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._write_ninja_file', '_write_ninja_file(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags, sources, objects, ldflags, library_target, with_cuda)', {'IS_WINDOWS': IS_WINDOWS, 'os': os, '_join_cuda_home': _join_cuda_home, '_is_cuda_file': _is_cuda_file, 'subprocess': subprocess, 'path': path, 'cflags': cflags, 'post_cflags': post_cflags, 'cuda_cflags': cuda_cflags, 'cuda_post_cflags': cuda_post_cflags, 'sources': sources, 'objects': objects, 'ldflags': ldflags, 'library_target': library_target, 'with_cuda': with_cuda}, 1)

def _join_cuda_home(*paths):
    """
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._join_cuda_home', '_join_cuda_home(*paths)', {'CUDA_HOME': CUDA_HOME, 'EnvironmentError': EnvironmentError, 'os': os, 'paths': paths}, 1)

def _is_cuda_file(path):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.cpp_extension._is_cuda_file', '_is_cuda_file(path)', {'IS_HIP_EXTENSION': IS_HIP_EXTENSION, 'os': os, 'path': path}, 1)

