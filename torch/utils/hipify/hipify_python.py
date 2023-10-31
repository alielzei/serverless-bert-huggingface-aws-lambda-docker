""" The Python Hipify script.
##
# Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
#               2017-2018 Advanced Micro Devices, Inc. and
#                         Facebook Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

from __future__ import absolute_import, division, print_function
import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
'This dictionary provides the mapping from PyTorch kernel template types\nto their actual types.'
PYTORCH_TEMPLATE_MAP = {'Dtype': 'scalar_t', 'T': 'scalar_t'}
CAFFE2_TEMPLATE_MAP = {}


class InputError(Exception):
    
    def __init__(self, message):
        super(InputError, self).__init__(message)
        self.message = message
    
    def __str__(self):
        return '{}: {}'.format('Input error', self.message)


def openf(filename, mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.openf', 'openf(filename, mode)', {'sys': sys, 'filename': filename, 'mode': mode}, 1)


class bcolors:
    HEADER = '\x1b[95m'
    OKBLUE = '\x1b[94m'
    OKGREEN = '\x1b[92m'
    WARNING = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'
    BOLD = '\x1b[1m'
    UNDERLINE = '\x1b[4m'


def matched_files_iter(root_path, includes=('*', ), ignores=(), extensions=(), out_of_place_only=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.matched_files_iter', "matched_files_iter(root_path, includes=('*', ), ignores=(), extensions=(), out_of_place_only=False)", {'fnmatch': fnmatch, 'os': os, 'is_pytorch_file': is_pytorch_file, 'is_caffe2_gpu_file': is_caffe2_gpu_file, 'is_out_of_place': is_out_of_place, 'root_path': root_path, 'includes': includes, 'ignores': ignores, 'extensions': extensions, 'out_of_place_only': out_of_place_only}, 1)

def preprocess(output_directory, all_files, show_detailed=False, show_progress=True, hip_clang_launch=False, is_pytorch_extension=False):
    """
    Call preprocessor on selected files.

    Arguments)
        show_detailed - Show a detailed summary of the transpilation process.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.preprocess', 'preprocess(output_directory, all_files, show_detailed=False, show_progress=True, hip_clang_launch=False, is_pytorch_extension=False)', {'preprocessor': preprocessor, 'get_hip_file_path': get_hip_file_path, 'bcolors': bcolors, 'sys': sys, 'compute_stats': compute_stats, 'output_directory': output_directory, 'all_files': all_files, 'show_detailed': show_detailed, 'show_progress': show_progress, 'hip_clang_launch': hip_clang_launch, 'is_pytorch_extension': is_pytorch_extension}, 0)

def compute_stats(stats):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.compute_stats', 'compute_stats(stats)', {'stats': stats}, 0)

def add_dim3(kernel_string, cuda_kernel):
    """adds dim3() to the second and third arguments in the kernel launch"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.add_dim3', 'add_dim3(kernel_string, cuda_kernel)', {'kernel_string': kernel_string, 'cuda_kernel': cuda_kernel}, 1)
RE_KERNEL_LAUNCH = re.compile('([ ]+)(detail?)::[ ]+\\\\\\n[ ]+')

def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.processKernelLaunches', 'processKernelLaunches(string, stats)', {'RE_KERNEL_LAUNCH': RE_KERNEL_LAUNCH, 'InputError': InputError, 'add_dim3': add_dim3, 'extract_arguments': extract_arguments, 'string': string, 'stats': stats}, 1)

def find_closure_group(input_string, start, group):
    """Generalization for finding a balancing closure group

         if group = ["(", ")"], then finds the first balanced parantheses.
         if group = ["{", "}"], then finds the first balanced bracket.

    Given an input string, a starting position in the input string, and the group type,
    find_closure_group returns the positions of group[0] and group[1] as a tuple.

    Example:
        find_closure_group("(hi)", 0, ["(", ")"])

    Returns:
        0, 3
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.find_closure_group', 'find_closure_group(input_string, start, group)', {'input_string': input_string, 'start': start, 'group': group}, 2)

def find_bracket_group(input_string, start):
    """Finds the first balanced parantheses."""
    return find_closure_group(input_string, start, group=['{', '}'])

def find_parentheses_group(input_string, start):
    """Finds the first balanced bracket."""
    return find_closure_group(input_string, start, group=['(', ')'])
RE_ASSERT = re.compile('\\bassert[ ]*\\(')

def replace_math_functions(input_string):
    """FIXME: Temporarily replace std:: invocations of math functions
        with non-std:: versions to prevent linker errors NOTE: This
        can lead to correctness issues when running tests, since the
        correct version of the math function (exp/expf) might not get
        called.  Plan is to remove this function once HIP supports
        std:: math function calls inside device code

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.replace_math_functions', 'replace_math_functions(input_string)', {'MATH_TRANSPILATIONS': MATH_TRANSPILATIONS, 'input_string': input_string}, 1)
RE_SYNCTHREADS = re.compile('[:]?[:]?\\b(__syncthreads)\\b(\\w*\\()')

def hip_header_magic(input_string):
    """If the file makes kernel builtin calls and does not include the cuda_runtime.h header,
    then automatically add an #include to match the "magic" includes provided by NVCC.
    TODO:
        Update logic to ignore cases where the cuda_runtime.h is included by another file.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.hip_header_magic', 'hip_header_magic(input_string)', {'re': re, 'RE_SYNCTHREADS': RE_SYNCTHREADS, 'input_string': input_string}, 1)
RE_EXTERN_SHARED = re.compile('extern\\s+([\\w\\(\\)]+)?\\s*__shared__\\s+([\\w:<>\\s]+)\\s+(\\w+)\\s*\\[\\s*\\]\\s*;')

def replace_extern_shared(input_string):
    """Match extern __shared__ type foo[]; syntax and use HIP_DYNAMIC_SHARED() MACRO instead.
       https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md#__shared__
    Example:
        "extern __shared__ char smemChar[];" => "HIP_DYNAMIC_SHARED( char, smemChar)"
        "extern __shared__ unsigned char smem[];" => "HIP_DYNAMIC_SHARED( unsigned char, my_smem)"
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.replace_extern_shared', 'replace_extern_shared(input_string)', {'RE_EXTERN_SHARED': RE_EXTERN_SHARED, 'input_string': input_string}, 1)

def get_hip_file_path(filepath):
    """
    Returns the new name of the hipified file
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.get_hip_file_path', 'get_hip_file_path(filepath)', {'is_out_of_place': is_out_of_place, 'os': os, 'filepath': filepath}, 1)

def is_out_of_place(filepath):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.is_out_of_place', 'is_out_of_place(filepath)', {'filepath': filepath}, 1)

def is_pytorch_file(filepath):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.is_pytorch_file', 'is_pytorch_file(filepath)', {'filepath': filepath}, 1)

def is_caffe2_gpu_file(filepath):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.is_caffe2_gpu_file', 'is_caffe2_gpu_file(filepath)', {'os': os, 'filepath': filepath}, 1)


class Trie:
    """Regex::Trie in Python. Creates a Trie out of a list of words. The trie can be exported to a Regex pattern.
    The corresponding Regex should match much faster than a simple Regex union."""
    
    def __init__(self):
        self.data = {}
    
    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = ((char in ref and ref[char]) or {})
            ref = ref[char]
        ref[''] = 1
    
    def dump(self):
        return self.data
    
    def quote(self, char):
        return re.escape(char)
    
    def _pattern(self, pData):
        data = pData
        if ('' in data and len(data.keys()) == 1):
            return None
        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except Exception:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0
        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')
        if len(alt) == 1:
            result = alt[0]
        else:
            result = '(?:' + '|'.join(alt) + ')'
        if q:
            if cconly:
                result += '?'
            else:
                result = '(?:%s)?' % result
        return result
    
    def pattern(self):
        return self._pattern(self.dump())

CAFFE2_TRIE = Trie()
CAFFE2_MAP = {}
PYTORCH_TRIE = Trie()
PYTORCH_MAP = {}
for mapping in CUDA_TO_HIP_MAPPINGS:
    for (src, value) in mapping.items():
        dst = value[0]
        meta_data = value[1:]
        if constants.API_CAFFE2 not in meta_data:
            PYTORCH_TRIE.add(src)
            PYTORCH_MAP[src] = dst
        if constants.API_PYTORCH not in meta_data:
            CAFFE2_TRIE.add(src)
            CAFFE2_MAP[src] = dst
RE_CAFFE2_PREPROCESSOR = re.compile(CAFFE2_TRIE.pattern())
RE_PYTORCH_PREPROCESSOR = re.compile('(?<=\\W)({0})(?=\\W)'.format(PYTORCH_TRIE.pattern()))
RE_QUOTE_HEADER = re.compile('#include "([^"]+)"')
RE_ANGLE_HEADER = re.compile('#include <([^>]+)>')
RE_THC_GENERIC_FILE = re.compile('#define THC_GENERIC_FILE "([^"]+)"')
RE_CU_SUFFIX = re.compile('\\.cu\\b')

def preprocessor(output_directory, filepath, stats, hip_clang_launch, is_pytorch_extension):
    """ Executes the CUDA -> HIP conversion on the specified file. """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.preprocessor', 'preprocessor(output_directory, filepath, stats, hip_clang_launch, is_pytorch_extension)', {'os': os, 'get_hip_file_path': get_hip_file_path, 'PYTORCH_MAP': PYTORCH_MAP, 'RE_PYTORCH_PREPROCESSOR': RE_PYTORCH_PREPROCESSOR, 'is_pytorch_file': is_pytorch_file, 'CAFFE2_MAP': CAFFE2_MAP, 'RE_CAFFE2_PREPROCESSOR': RE_CAFFE2_PREPROCESSOR, 'RE_QUOTE_HEADER': RE_QUOTE_HEADER, 'RE_ANGLE_HEADER': RE_ANGLE_HEADER, 'RE_THC_GENERIC_FILE': RE_THC_GENERIC_FILE, 'RE_CU_SUFFIX': RE_CU_SUFFIX, 'processKernelLaunches': processKernelLaunches, 'replace_math_functions': replace_math_functions, 'hip_header_magic': hip_header_magic, 'replace_extern_shared': replace_extern_shared, 'output_directory': output_directory, 'filepath': filepath, 'stats': stats, 'hip_clang_launch': hip_clang_launch, 'is_pytorch_extension': is_pytorch_extension}, 1)

def file_specific_replacement(filepath, search_string, replace_string, strict=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.file_specific_replacement', 'file_specific_replacement(filepath, search_string, replace_string, strict=False)', {'openf': openf, 're': re, 'filepath': filepath, 'search_string': search_string, 'replace_string': replace_string, 'strict': strict}, 0)

def file_add_header(filepath, header):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.file_add_header', 'file_add_header(filepath, header)', {'openf': openf, 'filepath': filepath, 'header': header}, 0)

def fix_static_global_kernels(in_txt):
    """Static global kernels in HIP results in a compilation error."""
    in_txt = in_txt.replace(' __global__ static', '__global__')
    return in_txt
RE_INCLUDE = re.compile('#include .*\\n')

def extract_arguments(start, string):
    """ Return the list of arguments in the upcoming function parameter closure.
        Example:
        string (input): '(blocks, threads, 0, THCState_getCurrentStream(state))'
        arguments (output):
            '[{'start': 1, 'end': 7},
            {'start': 8, 'end': 16},
            {'start': 17, 'end': 19},
            {'start': 20, 'end': 53}]'
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.extract_arguments', 'extract_arguments(start, string)', {'start': start, 'string': string}, 1)

def str2bool(v):
    """ArgumentParser doesn't support type=bool. Thus, this helper method will convert
    from possible string types to True / False."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.str2bool', 'str2bool(v)', {'argparse': argparse, 'v': v}, 1)

def hipify(project_directory, show_detailed=False, extensions=('.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.in', '.hpp'), output_directory='', includes=(), out_of_place_only=False, ignores=(), show_progress=True, hip_clang_launch=False, is_pytorch_extension=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.hipify.hipify_python.hipify', "hipify(project_directory, show_detailed=False, extensions=('.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.in', '.hpp'), output_directory='', includes=(), out_of_place_only=False, ignores=(), show_progress=True, hip_clang_launch=False, is_pytorch_extension=False)", {'os': os, 'sys': sys, 'shutil': shutil, 'matched_files_iter': matched_files_iter, 'preprocess': preprocess, 'project_directory': project_directory, 'show_detailed': show_detailed, 'extensions': extensions, 'output_directory': output_directory, 'includes': includes, 'out_of_place_only': out_of_place_only, 'ignores': ignores, 'show_progress': show_progress, 'hip_clang_launch': hip_clang_launch, 'is_pytorch_extension': is_pytorch_extension}, 0)

