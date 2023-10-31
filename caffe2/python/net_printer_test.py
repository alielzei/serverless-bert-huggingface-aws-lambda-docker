from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import net_printer
from caffe2.python.checkpoint import Job
from caffe2.python.net_builder import ops
from caffe2.python.task import Task, final_output, WorkspaceType
import unittest

def example_loop():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_printer_test.example_loop', 'example_loop()', {'Task': Task, 'ops': ops}, 0)

def example_task():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer_test.example_task', 'example_task()', {'Task': Task, 'ops': ops, 'final_output': final_output}, 3)

def example_job():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_printer_test.example_job', 'example_job()', {'Job': Job, 'example_loop': example_loop, 'example_task': example_task}, 1)


class TestNetPrinter(unittest.TestCase):
    
    def test_print(self):
        self.assertTrue(len(net_printer.to_string(example_job())) > 0)
    
    def test_valid_job(self):
        job = example_job()
        with job:
            with Task():
                ops.Add(['distributed_ctx_init_a', 'distributed_ctx_init_b'])
        print(net_printer.to_string(example_job()))
    
    def test_undefined_blob(self):
        job = example_job()
        with job:
            with Task():
                ops.Add(['a', 'b'])
        with self.assertRaises(AssertionError) as e:
            net_printer.analyze(job)
        self.assertEqual('Blob undefined: a', str(e.exception))
    
    def test_multiple_definition(self):
        job = example_job()
        with job:
            with Task(workspace_type=WorkspaceType.GLOBAL):
                ops.Add([ops.Const(0), ops.Const(1)], 'out1')
            with Task(workspace_type=WorkspaceType.GLOBAL):
                ops.Add([ops.Const(2), ops.Const(3)], 'out1')
        with self.assertRaises(AssertionError):
            net_printer.analyze(job)


