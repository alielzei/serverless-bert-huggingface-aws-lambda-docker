from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace
from caffe2.python.core import Plan, to_execution_step, Net
from caffe2.python.task import Task, TaskGroup, final_output
from caffe2.python.net_builder import ops, NetBuilder
from caffe2.python.session import LocalSession
import unittest
import threading


class PythonOpStats(object):
    lock = threading.Lock()
    num_instances = 0
    num_calls = 0


def python_op_builder():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_builder_test.python_op_builder', 'python_op_builder()', {'PythonOpStats': PythonOpStats}, 1)

def _test_loop():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_builder_test._test_loop', '_test_loop()', {'ops': ops}, 1)

def _test_inner_stop(x):
    ops.stop_if(ops.LT([x, ops.Const(5)]))

def _test_outer():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_builder_test._test_outer', '_test_outer()', {'ops': ops, '_test_inner_stop': _test_inner_stop}, 4)

def _test_if(x):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_builder_test._test_if', '_test_if(x)', {'ops': ops, 'x': x}, 1)


class TestNetBuilder(unittest.TestCase):
    
    def test_ops(self):
        with NetBuilder() as nb:
            y = _test_loop()
            (z, w, a, b) = _test_outer()
            p = _test_if(ops.Const(75))
            q = _test_if(ops.Const(25))
        plan = Plan('name')
        plan.AddStep(to_execution_step(nb))
        ws = workspace.C.Workspace()
        ws.run(plan)
        expected = [(y, 5), (z, False), (w, True), (a, False), (b, False), (p, 2), (q, 3)]
        for (b, expected) in expected:
            actual = ws.blobs[str(b)].fetch()
            self.assertEquals(actual, expected)
    
    def _expected_loop(self):
        total = 0
        total_large = 0
        total_small = 0
        total_tiny = 0
        for loop_iter in range(10):
            outer = loop_iter * 10
            for inner_iter in range(loop_iter):
                val = outer + inner_iter
                if val >= 80:
                    total_large += val
                elif val >= 50:
                    total_small += val
                else:
                    total_tiny += val
                total += val
        return (total, total_large, total_small, total_tiny)
    
    def _actual_loop(self):
        total = ops.Const(0)
        total_large = ops.Const(0)
        total_small = ops.Const(0)
        total_tiny = ops.Const(0)
        with ops.loop(10) as loop:
            outer = ops.Mul([loop.iter(), ops.Const(10)])
            with ops.loop(loop.iter()) as inner:
                val = ops.Add([outer, inner.iter()])
                with ops.If(ops.GE([val, ops.Const(80)])) as c:
                    ops.Add([total_large, val], [total_large])
                with c.Elif(ops.GE([val, ops.Const(50)])) as c:
                    ops.Add([total_small, val], [total_small])
                with c.Else():
                    ops.Add([total_tiny, val], [total_tiny])
                ops.Add([total, val], total)
        return [final_output(x) for x in [total, total_large, total_small, total_tiny]]
    
    def test_net_multi_use(self):
        with Task() as task:
            total = ops.Const(0)
            net = Net('my_net')
            net.Add([total, net.Const(1)], [total])
            ops.net(net)
            ops.net(net)
            result = final_output(total)
        with LocalSession() as session:
            session.run(task)
            self.assertEquals(2, result.fetch())
    
    def test_loops(self):
        with Task() as task:
            out_actual = self._actual_loop()
        with LocalSession() as session:
            session.run(task)
            expected = self._expected_loop()
            actual = [o.fetch() for o in out_actual]
            for (e, a) in zip(expected, actual):
                self.assertEquals(e, a)
    
    def test_setup(self):
        with Task() as task:
            with ops.task_init():
                one = ops.Const(1)
            two = ops.Add([one, one])
            with ops.task_init():
                three = ops.Const(3)
            accum = ops.Add([two, three])
            with ops.task_exit():
                seven_1 = ops.Add([accum, one])
            six = ops.Add([accum, one])
            ops.Add([accum, one], [accum])
            seven_2 = ops.Add([accum, one])
            o6 = final_output(six)
            o7_1 = final_output(seven_1)
            o7_2 = final_output(seven_2)
        with LocalSession() as session:
            session.run(task)
            self.assertEquals(o6.fetch(), 6)
            self.assertEquals(o7_1.fetch(), 7)
            self.assertEquals(o7_2.fetch(), 7)
    
    def test_multi_instance_python_op(self):
        """
        When task instances are created at runtime, C++ concurrently creates
        multiple instances of operators in C++, and concurrently destroys them
        once the task is finished. This means that the destructor of PythonOp
        will be called concurrently, so the GIL must be acquired. This
        test exercises this condition.
        """
        with Task(num_instances=64) as task:
            with ops.loop(4):
                ops.Python((python_op_builder, [], {}))([], [])
        with LocalSession() as session:
            PythonOpStats.num_instances = 0
            PythonOpStats.num_calls = 0
            session.run(task)
            self.assertEquals(PythonOpStats.num_instances, 64)
            self.assertEquals(PythonOpStats.num_calls, 256)
    
    def test_multi_instance(self):
        NUM_INSTANCES = 10
        NUM_ITERS = 15
        with TaskGroup() as tg:
            with Task(num_instances=NUM_INSTANCES):
                with ops.task_init():
                    counter1 = ops.CreateCounter([], ['global_counter'])
                    counter2 = ops.CreateCounter([], ['global_counter2'])
                    counter3 = ops.CreateCounter([], ['global_counter3'])
                with ops.task_instance_init():
                    task_counter = ops.CreateCounter([], ['task_counter'])
                local_counter = ops.CreateCounter([], ['local_counter'])
                with ops.loop(NUM_ITERS):
                    ops.CountUp(counter1)
                    ops.CountUp(task_counter)
                    ops.CountUp(local_counter)
                with ops.task_instance_exit():
                    count2 = ops.RetrieveCount(task_counter)
                    with ops.loop(ops.Mul([count2, count2])):
                        ops.CountUp(counter2)
                count3 = ops.RetrieveCount(local_counter)
                with ops.loop(ops.Mul([count3, count3])):
                    ops.CountUp(counter3)
                with ops.task_exit():
                    total1 = final_output(ops.RetrieveCount(counter1))
                    total2 = final_output(ops.RetrieveCount(counter2))
                    total3 = final_output(ops.RetrieveCount(counter3))
        with LocalSession() as session:
            session.run(tg)
            self.assertEquals(total1.fetch(), NUM_INSTANCES * NUM_ITERS)
            self.assertEquals(total2.fetch(), NUM_INSTANCES * NUM_ITERS**2)
            self.assertEquals(total3.fetch(), NUM_INSTANCES * NUM_ITERS**2)
    
    def test_if_net(self):
        with NetBuilder() as nb:
            x0 = ops.Const(0)
            x1 = ops.Const(1)
            x2 = ops.Const(2)
            y0 = ops.Const(0)
            y1 = ops.Const(1)
            y2 = ops.Const(2)
            first_res = ops.Const(0)
            with ops.IfNet(ops.Const(True)):
                ops.Const(1, blob_out=first_res)
            with ops.Else():
                ops.Const(2, blob_out=first_res)
            second_res = ops.Const(0)
            with ops.IfNet(ops.Const(False)):
                ops.Const(1, blob_out=second_res)
            with ops.Else():
                ops.Const(2, blob_out=second_res)
            with ops.IfNet(ops.LT([x0, x1])):
                local_blob = ops.Const(900)
                ops.Add([ops.Const(100), local_blob], [y0])
                gt = ops.GT([x1, x2])
                with ops.IfNet(gt):
                    pass
                with ops.Else():
                    ops.Add([y1, local_blob], [local_blob])
                    ops.Add([ops.Const(100), y1], [y1])
                with ops.IfNet(ops.EQ([local_blob, ops.Const(901)])):
                    ops.Const(7, blob_out=y2)
                    ops.Add([y1, y2], [y2])
            with ops.Else():
                pass
        plan = Plan('if_net_test')
        plan.AddStep(to_execution_step(nb))
        ws = workspace.C.Workspace()
        ws.run(plan)
        first_res_value = ws.blobs[str(first_res)].fetch()
        second_res_value = ws.blobs[str(second_res)].fetch()
        y0_value = ws.blobs[str(y0)].fetch()
        y1_value = ws.blobs[str(y1)].fetch()
        y2_value = ws.blobs[str(y2)].fetch()
        self.assertEquals(first_res_value, 1)
        self.assertEquals(second_res_value, 2)
        self.assertEquals(y0_value, 1000)
        self.assertEquals(y1_value, 101)
        self.assertEquals(y2_value, 108)
        self.assertTrue(str(local_blob) not in ws.blobs)
    
    def test_while_net(self):
        with NetBuilder() as nb:
            x = ops.Const(0)
            y = ops.Const(0)
            with ops.WhileNet():
                with ops.Condition():
                    ops.Add([x, ops.Const(1)], [x])
                    ops.LT([x, ops.Const(7)])
                ops.Add([x, y], [y])
        plan = Plan('while_net_test')
        plan.AddStep(to_execution_step(nb))
        ws = workspace.C.Workspace()
        ws.run(plan)
        x_value = ws.blobs[str(x)].fetch()
        y_value = ws.blobs[str(y)].fetch()
        self.assertEqual(x_value, 7)
        self.assertEqual(y_value, 21)


