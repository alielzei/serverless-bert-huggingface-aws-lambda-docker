import argparse
import cProfile
import pstats
import sys
import os
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info

def redirect_argv(new_argv):
    sys.argv[:] = new_argv[:]

def compiled_with_cuda(sysinfo):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.compiled_with_cuda', 'compiled_with_cuda(sysinfo)', {'sysinfo': sysinfo}, 1)
env_summary = '\n--------------------------------------------------------------------------------\n  Environment Summary\n--------------------------------------------------------------------------------\nPyTorch {pytorch_version}{debug_str} {cuda_compiled}\nRunning with Python {py_version} and {cuda_runtime}\n\n`{pip_version} list` truncated output:\n{pip_list_output}\n'.strip()

def run_env_analysis():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.run_env_analysis', 'run_env_analysis()', {'get_env_info': get_env_info, 'compiled_with_cuda': compiled_with_cuda, 'sys': sys, 'env_summary': env_summary}, 1)

def run_cprofile(code, globs, launch_blocking=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.run_cprofile', 'run_cprofile(code, globs, launch_blocking=False)', {'cProfile': cProfile, 'code': code, 'globs': globs, 'launch_blocking': launch_blocking}, 1)
cprof_summary = '\n--------------------------------------------------------------------------------\n  cProfile output\n--------------------------------------------------------------------------------\n'.strip()

def print_cprofile_summary(prof, sortby='tottime', topk=15):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.print_cprofile_summary', "print_cprofile_summary(prof, sortby='tottime', topk=15)", {'cprof_summary': cprof_summary, 'pstats': pstats, 'prof': prof, 'sortby': sortby, 'topk': topk}, 0)

def run_autograd_prof(code, globs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.run_autograd_prof', 'run_autograd_prof(code, globs)', {'profiler': profiler, 'torch': torch, 'code': code, 'globs': globs}, 1)
autograd_prof_summary = '\n--------------------------------------------------------------------------------\n  autograd profiler output ({mode} mode)\n--------------------------------------------------------------------------------\n        {description}\n{cuda_warning}\n{output}\n'.strip()

def print_autograd_prof_summary(prof, mode, sortby='cpu_time', topk=15):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.print_autograd_prof_summary', "print_autograd_prof_summary(prof, mode, sortby='cpu_time', topk=15)", {'autograd_prof_sortby': autograd_prof_sortby, 'torch': torch, 'autograd_prof_summary': autograd_prof_summary, 'prof': prof, 'mode': mode, 'sortby': sortby, 'topk': topk}, 0)
descript = "\n`bottleneck` is a tool that can be used as an initial step for debugging\nbottlenecks in your program.\n\nIt summarizes runs of your script with the Python profiler and PyTorch's\nautograd profiler. Because your script will be profiled, please ensure that it\nexits in a finite amount of time.\n\nFor more complicated uses of the profilers, please see\nhttps://docs.python.org/3/library/profile.html and\nhttps://pytorch.org/docs/master/autograd.html#profiler for more information.\n".strip()

def parse_args():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.parse_args', 'parse_args()', {'argparse': argparse, 'descript': descript}, 1)

def cpu_time_total(autograd_prof):
    return sum([event.cpu_time_total for event in autograd_prof.function_events])

def main():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.bottleneck.__main__.main', 'main()', {'parse_args': parse_args, 'redirect_argv': redirect_argv, 'sys': sys, 'os': os, 'descript': descript, 'run_env_analysis': run_env_analysis, 'torch': torch, 'run_cprofile': run_cprofile, 'run_autograd_prof': run_autograd_prof, 'print_cprofile_summary': print_cprofile_summary, 'print_autograd_prof_summary': print_autograd_prof_summary, 'cpu_time_total': cpu_time_total}, 1)
if __name__ == '__main__':
    main()

