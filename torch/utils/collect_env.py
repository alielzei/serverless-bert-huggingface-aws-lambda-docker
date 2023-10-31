from __future__ import absolute_import, division, print_function, unicode_literals
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    TORCH_AVAILABLE = False
PY3 = sys.version_info >= (3, 0)
SystemEnv = namedtuple('SystemEnv', ['torch_version', 'is_debug_build', 'cuda_compiled_version', 'gcc_version', 'cmake_version', 'os', 'python_version', 'is_cuda_available', 'cuda_runtime_version', 'nvidia_driver_version', 'nvidia_gpu_models', 'cudnn_version', 'pip_version', 'pip_packages', 'conda_packages'])

def run(command):
    """Returns (return-code, stdout, stderr)"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.run', 'run(command)', {'subprocess': subprocess, 'PY3': PY3, 'locale': locale, 'command': command}, 3)

def run_and_read_all(run_lambda, command):
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.run_and_read_all', 'run_and_read_all(run_lambda, command)', {'run_lambda': run_lambda, 'command': command}, 1)

def run_and_parse_first_match(run_lambda, command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.run_and_parse_first_match', 'run_and_parse_first_match(run_lambda, command, regex)', {'re': re, 'run_lambda': run_lambda, 'command': command, 'regex': regex}, 1)

def get_conda_packages(run_lambda):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_conda_packages', 'get_conda_packages(run_lambda)', {'get_platform': get_platform, 'os': os, 'run_and_read_all': run_and_read_all, 're': re, 'run_lambda': run_lambda}, 1)

def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'gcc --version', 'gcc (.*)')

def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cmake --version', 'cmake (.*)')

def get_nvidia_driver_version(run_lambda):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_nvidia_driver_version', 'get_nvidia_driver_version(run_lambda)', {'get_platform': get_platform, 'run_and_parse_first_match': run_and_parse_first_match, 'get_nvidia_smi': get_nvidia_smi, 'run_lambda': run_lambda}, 1)

def get_gpu_info(run_lambda):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_gpu_info', 'get_gpu_info(run_lambda)', {'get_platform': get_platform, 'TORCH_AVAILABLE': TORCH_AVAILABLE, 'torch': torch, 'get_nvidia_smi': get_nvidia_smi, 're': re, 'run_lambda': run_lambda}, 1)

def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'nvcc --version', 'V(.*)$')

def get_cudnn_version(run_lambda):
    """This will return a list of libcudnn.so; it's hard to tell which one is being used"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_cudnn_version', 'get_cudnn_version(run_lambda)', {'get_platform': get_platform, 'os': os, 'run_lambda': run_lambda}, 1)

def get_nvidia_smi():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_nvidia_smi', 'get_nvidia_smi()', {'get_platform': get_platform}, 1)

def get_platform():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_platform', 'get_platform()', {'sys': sys}, 1)

def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'sw_vers -productVersion', '(.*)')

def get_windows_version(run_lambda):
    return run_and_read_all(run_lambda, 'wmic os get Caption | findstr /v Caption')

def get_lsb_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'lsb_release -a', 'Description:\\t(.*)')

def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cat /etc/*-release', 'PRETTY_NAME="(.*)"')

def get_os(run_lambda):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_os', 'get_os(run_lambda)', {'get_platform': get_platform, 'get_windows_version': get_windows_version, 'get_mac_version': get_mac_version, 'get_lsb_version': get_lsb_version, 'check_release_file': check_release_file, 'run_lambda': run_lambda}, 1)

def get_pip_packages(run_lambda):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_pip_packages', 'get_pip_packages(run_lambda)', {'get_platform': get_platform, 'run_and_read_all': run_and_read_all, 'PY3': PY3, 'run_lambda': run_lambda}, 1)

def get_env_info():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.get_env_info', 'get_env_info()', {'run': run, 'get_pip_packages': get_pip_packages, 'TORCH_AVAILABLE': TORCH_AVAILABLE, 'torch': torch, 'SystemEnv': SystemEnv, 'sys': sys, 'get_running_cuda_version': get_running_cuda_version, 'get_gpu_info': get_gpu_info, 'get_nvidia_driver_version': get_nvidia_driver_version, 'get_cudnn_version': get_cudnn_version, 'get_conda_packages': get_conda_packages, 'get_os': get_os, 'get_gcc_version': get_gcc_version, 'get_cmake_version': get_cmake_version}, 1)
env_info_fmt = '\nPyTorch version: {torch_version}\nIs debug build: {is_debug_build}\nCUDA used to build PyTorch: {cuda_compiled_version}\n\nOS: {os}\nGCC version: {gcc_version}\nCMake version: {cmake_version}\n\nPython version: {python_version}\nIs CUDA available: {is_cuda_available}\nCUDA runtime version: {cuda_runtime_version}\nGPU models and configuration: {nvidia_gpu_models}\nNvidia driver version: {nvidia_driver_version}\ncuDNN version: {cudnn_version}\n\nVersions of relevant libraries:\n{pip_packages}\n{conda_packages}\n'.strip()

def pretty_str(envinfo):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.collect_env.pretty_str', 'pretty_str(envinfo)', {'TORCH_AVAILABLE': TORCH_AVAILABLE, 'torch': torch, 'env_info_fmt': env_info_fmt, 'envinfo': envinfo}, 1)

def get_pretty_env_info():
    return pretty_str(get_env_info())

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.collect_env.main', 'main()', {'get_pretty_env_info': get_pretty_env_info}, 0)
if __name__ == '__main__':
    main()

