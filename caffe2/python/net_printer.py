from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto.caffe2_pb2 import OperatorDef, NetDef
from caffe2.python.checkpoint import Job
from caffe2.python.core import Net, ExecutionStep, Plan
from caffe2.python.task import Task, TaskGroup, WorkspaceType, TaskOutput
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from future.utils import viewkeys
from itertools import chain
from six import binary_type, text_type


class Visitor(object):
    
    @classmethod
    def register(cls, Type):
        if not hasattr(cls, 'visitors'):
            cls.visitors = {}
        else:
            assert Type not in cls.visitors, '{} already registered!'.format(Type)
        
        def _register(func):
            cls.visitors[Type] = func
            return func
        return _register
    
    def __call__(self, obj, *args, **kwargs):
        if obj is None:
            return
        Type = type(obj)
        if Type not in self.__class__.visitors:
            raise TypeError('%s: unsupported object type: %s' % (self.__class__.__name__, Type))
        func = self.__class__.visitors[Type]
        return func(self, obj, *args, **kwargs)



class Analyzer(Visitor):
    PREFIXES_TO_IGNORE = {'distributed_ctx_init'}
    
    def __init__(self):
        self.workspaces = defaultdict(lambda: defaultdict(lambda: 0))
        self.workspace_ctx = []
    
    @property
    def workspace(self):
        return self.workspace_ctx[-1]
    
    @contextmanager
    def set_workspace(self, node=None, ws=None, do_copy=False):
        if ws is not None:
            ws = ws
        elif node is not None:
            ws = self.workspaces[str(node)]
        else:
            ws = self.workspace
        if do_copy:
            ws = copy(ws)
        self.workspace_ctx.append(ws)
        yield ws
        del self.workspace_ctx[-1]
    
    def define_blob(self, blob):
        self.workspace[blob] += 1
    
    def need_blob(self, blob):
        if any((blob.startswith(p) for p in Analyzer.PREFIXES_TO_IGNORE)):
            return
        assert blob in self.workspace, 'Blob undefined: %s' % blob


@Analyzer.register(OperatorDef)
def analyze_op(analyzer, op):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.analyze_op', 'analyze_op(analyzer, op)', {'Analyzer': Analyzer, 'OperatorDef': OperatorDef, 'analyzer': analyzer, 'op': op}, 0)

@Analyzer.register(Net)
def analyze_net(analyzer, net):
    for x in net.Proto().op:
        analyzer(x)

@Analyzer.register(ExecutionStep)
def analyze_step(analyzer, step):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.analyze_step', 'analyze_step(analyzer, step)', {'viewkeys': viewkeys, 'Analyzer': Analyzer, 'ExecutionStep': ExecutionStep, 'analyzer': analyzer, 'step': step}, 0)

@Analyzer.register(Task)
def analyze_task(analyzer, task):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.analyze_task', 'analyze_task(analyzer, task)', {'Plan': Plan, 'WorkspaceType': WorkspaceType, 'Analyzer': Analyzer, 'Task': Task, 'analyzer': analyzer, 'task': task}, 0)

@Analyzer.register(TaskGroup)
def analyze_task_group(analyzer, tg):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.analyze_task_group', 'analyze_task_group(analyzer, tg)', {'Analyzer': Analyzer, 'TaskGroup': TaskGroup, 'analyzer': analyzer, 'tg': tg}, 0)

@Analyzer.register(Job)
def analyze_job(analyzer, job):
    analyzer(job.init_group)
    analyzer(job.epoch_group)

def analyze(obj):
    """
    Given a Job, visits all the execution steps making sure that:
      - no undefined blobs will be found during execution
      - no blob with same name is defined in concurrent steps
    """
    Analyzer()(obj)


class Text(object):
    
    def __init__(self):
        self._indent = 0
        self._lines_in_context = [0]
        self.lines = []
    
    @contextmanager
    def context(self, text):
        if text is not None:
            self.add('with %s:' % text)
            self._indent += 4
            self._lines_in_context.append(0)
        yield
        if text is not None:
            if self._lines_in_context[-1] == 0:
                self.add('pass')
            self._indent -= 4
            del self._lines_in_context[-1]
    
    def add(self, text):
        self._lines_in_context[-1] += 1
        self.lines.append(' ' * self._indent + text)
    
    def __str__(self):
        return '\n'.join(self.lines)



class Printer(Visitor, Text):
    
    def __init__(self, factor_prefixes=False, c2_syntax=True):
        super(Visitor, self).__init__()
        super(Text, self).__init__()
        self.factor_prefixes = factor_prefixes
        self.c2_syntax = c2_syntax
        self.c2_net_name = None


def _sanitize_str(s):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer._sanitize_str', '_sanitize_str(s)', {'text_type': text_type, 'binary_type': binary_type, 's': s}, 1)

def _arg_val(arg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer._arg_val', '_arg_val(arg)', {'_sanitize_str': _sanitize_str, 'arg': arg}, 1)

def commonprefix(m):
    """Given a list of strings, returns the longest common prefix"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer.commonprefix', 'commonprefix(m)', {'m': m}, 1)

def format_value(val):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer.format_value', 'format_value(val)', {'val': val}, 1)

def factor_prefix(vals, do_it):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer.factor_prefix', 'factor_prefix(vals, do_it)', {'format_value': format_value, 'commonprefix': commonprefix, 'vals': vals, 'do_it': do_it}, 1)

def call(op, inputs=None, outputs=None, factor_prefixes=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer.call', 'call(op, inputs=None, outputs=None, factor_prefixes=False)', {'chain': chain, 'factor_prefix': factor_prefix, 'op': op, 'inputs': inputs, 'outputs': outputs, 'factor_prefixes': factor_prefixes}, 1)

def format_device_option(dev_opt):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer.format_device_option', 'format_device_option(dev_opt)', {'call': call, 'dev_opt': dev_opt}, 1)

@Printer.register(OperatorDef)
def print_op(text, op):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.print_op', 'print_op(text, op)', {'_arg_val': _arg_val, 'format_device_option': format_device_option, 'call': call, 'Printer': Printer, 'OperatorDef': OperatorDef, 'text': text, 'op': op}, 0)

@Printer.register(NetDef)
def print_net_def(text, net_def):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.print_net_def', 'print_net_def(text, net_def)', {'call': call, 'Printer': Printer, 'NetDef': NetDef, 'text': text, 'net_def': net_def}, 0)

@Printer.register(Net)
def print_net(text, net):
    text(net.Proto())

def _get_step_context(step):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer._get_step_context', '_get_step_context(step)', {'call': call, 'step': step}, 2)

@Printer.register(ExecutionStep)
def print_step(text, step):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.print_step', 'print_step(text, step)', {'_get_step_context': _get_step_context, 'call': call, 'Printer': Printer, 'ExecutionStep': ExecutionStep, 'text': text, 'step': step}, 0)

def _print_task_output(x):
    assert isinstance(x, TaskOutput)
    return 'Output[' + ', '.join((str(x) for x in x.names)) + ']'

@Printer.register(Task)
def print_task(text, task):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.print_task', 'print_task(text, task)', {'_print_task_output': _print_task_output, 'call': call, 'Printer': Printer, 'Task': Task, 'text': text, 'task': task}, 0)

@Printer.register(TaskGroup)
def print_task_group(text, tg, header=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.print_task_group', 'print_task_group(text, tg, header=None)', {'call': call, 'Printer': Printer, 'TaskGroup': TaskGroup, 'text': text, 'tg': tg, 'header': header}, 0)

@Printer.register(Job)
def print_job(text, job):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer.print_job', 'print_job(text, job)', {'_print_task_output': _print_task_output, 'Printer': Printer, 'Job': Job, 'text': text, 'job': job}, 0)

def to_string(obj, **kwargs):
    """
    Given a Net, ExecutionStep, Task, TaskGroup or Job, produces a string
    with detailed description of the execution steps.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer.to_string', 'to_string(obj, **kwargs)', {'Printer': Printer, 'obj': obj, 'kwargs': kwargs}, 1)

def debug_net(net):
    """
    Given a Net, produce another net that logs info about the operator call
    before each operator execution. Use for debugging purposes.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer.debug_net', 'debug_net(net)', {'Net': Net, 'Text': Text, 'print_op': print_op, 'net': net}, 1)

