import sys
import threading
import time
import unittest
from enum import Enum
import torch
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.testing._internal.common_utils import IS_MACOS
import torch.testing._internal.dist_utils as dist_utils
from torch.testing._internal.dist_utils import dist_init, get_shutdown_error_regex, initialize_pg, wait_until_node_failure, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
rpc_done = [False, False, False, False]
ctx_ids = [-1, -1, -1, -1]
known_context_ids = set()
requires_grad_tensor = torch.ones(3, 3, requires_grad=True)

def _set_rpc_done(ctx_id, rank_distance):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.distributed.rpc.dist_autograd_test._set_rpc_done', '_set_rpc_done(ctx_id, rank_distance)', {'rpc_done': rpc_done, 'ctx_ids': ctx_ids, 'known_context_ids': known_context_ids, 'ctx_id': ctx_id, 'rank_distance': rank_distance}, 0)

def _check_rpc_done(rank_distance):
    while not rpc_done[rank_distance]:
        time.sleep(0.1)

def _torch_ones(sizes, requires_grad=False):
    return torch.ones(sizes, requires_grad=requires_grad)

def _compare_owner_value(context_id, rref, grad):
    grads = dist_autograd.get_gradients(context_id)
    return torch.equal(grads[rref.local_value()], grad)

def create_tensor():
    return torch.ones((3, 3), requires_grad=True)

@torch.jit.script
def create_torchscript_tensor():
    return torch.ones((3, 3)).requires_grad_()

def my_py_add(t1, t2):
    return torch.add(t1, t2)

def my_scalar_add(a, b):
    return a + b

def my_rref_add(rref_t1, t2):
    ret = torch.add(rref_t1.local_value(), t2)
    return ret

@torch.jit.script
def my_script_add(t1, t2):
    return torch.add(t1, t2)

@torch.jit.script
def my_script_ref_add(ref_t1, t2):
    t1 = ref_t1.to_here()
    return torch.add(t1, t2)

def my_nested_rref_add(dst, rref_t1, t2):
    return rpc.rpc_sync(dst, my_rref_add, args=(rref_t1, t2))

def ret_requires_grad():
    return requires_grad_tensor

def my_py_nested_call(t1, t2, dst, world_size, hops):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.distributed.rpc.dist_autograd_test.my_py_nested_call', 'my_py_nested_call(t1, t2, dst, world_size, hops)', {'rpc': rpc, 'worker_name': worker_name, 'my_py_nested_call': my_py_nested_call, 'my_py_add': my_py_add, 't1': t1, 't2': t2, 'dst': dst, 'world_size': world_size, 'hops': hops}, 1)

def _all_contexts_cleaned_up(timeout_seconds=10):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.distributed.rpc.dist_autograd_test._all_contexts_cleaned_up', '_all_contexts_cleaned_up(timeout_seconds=10)', {'time': time, 'known_context_ids': known_context_ids, 'dist_autograd': dist_autograd, 'timeout_seconds': timeout_seconds}, 1)

def _run_trainer(rref_t1, t2, ps, rank_diff):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.distributed.rpc.dist_autograd_test._run_trainer', '_run_trainer(rref_t1, t2, ps, rank_diff)', {'dist_autograd': dist_autograd, 'rpc': rpc, 'my_rref_add': my_rref_add, '_set_rpc_done': _set_rpc_done, '_check_rpc_done': _check_rpc_done, 'rref_t1': rref_t1, 't2': t2, 'ps': ps, 'rank_diff': rank_diff}, 0)

def _run_trainer_torchscript(rref_t1, t2, ps, rank_diff):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.distributed.rpc.dist_autograd_test._run_trainer_torchscript', '_run_trainer_torchscript(rref_t1, t2, ps, rank_diff)', {'dist_autograd': dist_autograd, 'rpc': rpc, 'my_script_ref_add': my_script_ref_add, '_set_rpc_done': _set_rpc_done, '_check_rpc_done': _check_rpc_done, 'rref_t1': rref_t1, 't2': t2, 'ps': ps, 'rank_diff': rank_diff}, 0)


class SimulateBackwardError(Function):
    _simulate_error = True
    
    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    @once_differentiable
    def backward(ctx, input):
        if SimulateBackwardError._simulate_error:
            raise Exception('Simulate error on backward pass')
        else:
            return input



class ExecMode(Enum):
    LOCAL = 1
    RPC_SYNC = 2
    REMOTE = 3
    RPC_ASYNC = 4



@unittest.skipIf(not torch._six.PY3, 'Pytorch distributed autograd package does not support python2')
class DistAutogradTest(RpcAgentTestFixture):
    
    def _exec_func_with_dst(self, dst, exec_mode, method, *args):
        if ExecMode.LOCAL == exec_mode:
            if (len(args) == 1 and isinstance(args[0], list)):
                return method(*args[0])
            return method(*args)
        elif ExecMode.RPC_SYNC == exec_mode:
            return rpc.rpc_sync(worker_name(dst), method, args=args)
        elif ExecMode.REMOTE == exec_mode:
            return rpc.remote(worker_name(dst), method, args=args).to_here()
        elif ExecMode.RPC_ASYNC == exec_mode:
            fut = rpc.rpc_async(worker_name(dst), method, args=args)
            return fut.wait()
        else:
            raise ValueError('Unrecognized ExecMode {}'.format(exec_mode))
    
    def _exec_func(self, exec_mode, method, *args):
        return self._exec_func_with_dst(self._next_rank(), exec_mode, method, *args)
    
    def _next_rank(self):
        if hasattr(self, 'dst_rank'):
            self.dst_rank = (self.dst_rank + 1) % self.world_size
            if self.dst_rank == self.rank:
                return self._next_rank()
        else:
            self.dst_rank = (self.rank + 1) % self.world_size
        return self.dst_rank
    
    def _check_rpc_done(self, rank_distance):
        _check_rpc_done(rank_distance)
    
    @dist_init
    def test_autograd_context(self):
        max_auto_increment = 281474976710655
        self.assertEqual(max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id())
        context_ids = []
        for i in range(1000):
            with dist_autograd.context() as context_id:
                self.assertEqual(context_id, dist_autograd._retrieve_context(context_id)._context_id())
                self.assertEqual(self.worker_id, context_id >> 48)
                context_ids.append(context_id)
        for context_id in context_ids:
            with self.assertRaisesRegex(RuntimeError, 'Could not find autograd context with id: {}'.format(context_id)):
                dist_autograd._retrieve_context(context_id)
    
    @dist_init
    def test_nested_context(self):
        with dist_autograd.context() as context_id:
            with self.assertRaisesRegex(RuntimeError, 'Already have an autograd context id for this thread'):
                with dist_autograd.context() as context_id:
                    pass
    
    def _verify_graph_for_first_rpc_call(self, send_function, recv_function, t1, t2, ret):
        next_funcs = send_function.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[0][0].name())
        self.assertEqual(t1, next_funcs[0][0].variable)
        self.assertEqual(0, next_funcs[0][1])
        self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[1][0].name())
        self.assertEqual(t2, next_funcs[1][0].variable)
        self.assertEqual(0, next_funcs[1][1])
        self.assertEqual(ret.grad_fn, recv_function)
    
    def _verify_graph_for_rpc_call_exec(self, send_function):
        next_funcs = send_function.next_functions
        self.assertEqual(1, len(next_funcs))
        add_backward_fn = next_funcs[0][0]
        self.assertEqual('AddBackward0', add_backward_fn.name())
        next_funcs = add_backward_fn.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[1][0].name())
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])
    
    def _verify_graph_for_nested_rpc_call(self, ctx):
        send_functions = ctx._send_functions()
        self.assertEqual(2, len(send_functions))
        next_funcs = list(send_functions.values())[0].next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[1][0].name())
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])
        next_funcs = list(send_functions.values())[1].next_functions
        self.assertEqual(1, len(next_funcs))
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())
    
    def _test_graph(self, fn, exec_mode):
        dst_rank = (self.rank + 1) % self.world_size
        initialize_pg(self.init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), fn, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), fn, args=(t1, t2)).to_here()
            else:
                raise ValueError('Unrecognized ExecMode {}'.format(exec_mode))
            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(list(send_functions.values())[0], list(recv_functions.values())[0], t1, t2, ret)
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[0])
            dist.barrier()
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()
    
    @dist_init
    def test_graph_for_builtin_call(self):
        self._test_graph(torch.add, ExecMode.RPC_SYNC)
    
    @dist_init
    def test_graph_for_python_call(self):
        self._test_graph(my_py_add, ExecMode.RPC_SYNC)
    
    @dist_init
    def test_graph_for_builtin_remote_call(self):
        self._test_graph(torch.add, ExecMode.REMOTE)
    
    @dist_init
    def test_graph_for_python_remote_call(self):
        self._test_graph(my_py_add, ExecMode.REMOTE)
    
    def _test_graph_for_py_nested_call(self, exec_mode):
        dst_rank = (self.rank + 1) % self.world_size
        initialize_pg(self.init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            nest_dst_rank = (dst_rank + 1) % self.world_size
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, dst_rank, self.world_size, 1))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, dst_rank, self.world_size, 1)).to_here()
            else:
                raise ValueError('Unrecognized ExecMode {}'.format(exec_mode))
            dist.barrier()
            for rd in [1, 2, 3]:
                rpc.rpc_sync(worker_name((self.rank + rd) % self.world_size), _set_rpc_done, args=(context_id, rd))
            dist.barrier()
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(list(send_functions.values())[0], list(recv_functions.values())[0], t1, t2, ret)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)
            ctx = dist_autograd._retrieve_context(ctx_ids[2])
            self._verify_graph_for_nested_rpc_call(ctx)
            ctx = dist_autograd._retrieve_context(ctx_ids[3])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[0])
            dist.barrier()
    
    @dist_init
    def test_graph_for_py_nested_call(self):
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC)
    
    @dist_init
    def test_graph_for_py_nested_remote_call(self):
        self._test_graph_for_py_nested_call(ExecMode.REMOTE)
    
    def _test_graph_for_py_nested_call_itself(self, exec_mode):
        dst_rank = (self.rank + 1) % self.world_size
        initialize_pg(self.init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, (self.rank - 1 + self.world_size) % self.world_size, self.world_size, 0))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, (self.rank - 1 + self.world_size) % self.world_size, self.world_size, 0)).to_here()
            else:
                raise ValueError('Unrecognized ExecMode {}'.format(exec_mode))
            rpc.rpc_sync(worker_name((self.rank + 1) % self.world_size), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(2, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(2, len(recv_functions))
            self._verify_graph_for_first_rpc_call(list(send_functions.values())[0], list(recv_functions.values())[1], t1, t2, ret)
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[1])
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)
            dist.barrier()
    
    @dist_init
    def test_graph_for_py_nested_call_itself(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC)
    
    @dist_init
    def test_graph_for_py_nested_remote_call_itself(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE)
    
    def _test_no_graph_with_tensors_not_require_grad(self, exec_mode):
        initialize_pg(self.init_method, self.rank, self.world_size)
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), torch.add, args=(t1, t2)).to_here()
            else:
                raise ValueError('Unrecognized ExecMode {}'.format(exec_mode))
            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            send_functions = ctx._send_functions()
            self.assertEqual(len(send_functions), 0)
            recv_functions = ctx._recv_functions()
            self.assertEqual(len(recv_functions), 0)
            self._check_rpc_done(1)
            self.assertNotEqual(-1, dist_autograd._retrieve_context(ctx_ids[1]))
            dist.barrier()
    
    @dist_init
    def test_no_graph_with_tensors_not_require_grad(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC)
    
    @dist_init
    def test_no_graph_with_tensors_not_require_grad_remote(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE)
    
    def _test_grad_only_on_return_value(self, exec_mode):
        initialize_pg(self.init_method, self.rank, self.world_size)
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), ret_requires_grad)
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), ret_requires_grad).to_here()
            else:
                raise ValueError('Unrecognized ExecMode {}'.format(exec_mode))
            dist_autograd.backward(context_id, [ret.sum()])
            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            self._check_rpc_done(1)
            grads = dist_autograd.get_gradients(ctx_ids[1])
            self.assertEqual(1, len(grads))
            self.assertIn(requires_grad_tensor, grads)
            self.assertEqual(torch.ones_like(ret), grads[requires_grad_tensor])
            dist.barrier()
    
    @dist_init
    def test_grad_only_on_return_value(self):
        self._test_grad_only_on_return_value(ExecMode.RPC_SYNC)
    
    @dist_init
    def test_grad_only_on_return_value_remote(self):
        self._test_grad_only_on_return_value(ExecMode.REMOTE)
    
    def _test_rpc_complex_args(self, exec_mode):
        with dist_autograd.context() as context_id:
            num_tensors = 10
            tensors = []
            for i in range(num_tensors):
                tensors.append(torch.ones(3, 3, requires_grad=i % 2 == 0))
            dst_rank = self._next_rank()
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.stack, args=(tensors, ))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), torch.stack, args=(tensors, )).to_here()
            else:
                raise ValueError('Unrecognized ExecMode {}'.format(exec_mode))
            self.assertEqual(torch.stack(tensors), ret)
            next_funcs = list(dist_autograd._current_context()._send_functions().values())[0].next_functions
            idx = 0
            for i in range(len(next_funcs)):
                self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[i][0].name())
                self.assertEqual(tensors[i], next_funcs[i][0].variable)
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), 1)
            self.assertEqual(worker_ids, {dst_rank})
    
    @dist_init
    def test_rpc_complex_args(self):
        self._test_rpc_complex_args(ExecMode.RPC_SYNC)
    
    @dist_init
    def test_remote_complex_args(self):
        self._test_rpc_complex_args(ExecMode.REMOTE)
    
    def context_cleanup_test_helper(self, rpc_args, func, nested=False):
        initialize_pg(self.init_method, self.rank, self.world_size)
        if nested:
            dst_rank = (self.rank + 1) % self.world_size
            nested_dst_rank = (dst_rank + 1) % self.world_size
            dst_ranks = {dst_rank}
        else:
            dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        with dist_autograd.context() as context_id:
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
                if nested:
                    rpc.rpc_sync(worker_name(nested_dst_rank), _set_rpc_done, args=(context_id, 2))
        with self.assertRaises(RuntimeError):
            dist_autograd._retrieve_context(context_id)
        dist.barrier()
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)
    
    @dist_init
    def test_context_cleanup_tensor_with_grad(self):
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)
    
    @dist_init
    def test_context_cleanup_tensor_no_grad(self):
        t1 = torch.ones(3, 3, requires_grad=False)
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)
    
    @dist_init
    def test_context_cleanup_no_tensors(self):
        self.context_cleanup_test_helper(rpc_args=(1, 1), func=my_scalar_add)
    
    @dist_init
    def test_context_cleanup_nested_rpc(self):
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        dst_rank = (self.rank + 1) % self.world_size
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(rpc_args=args, func=my_py_nested_call, nested=True)
    
    @dist_init
    def test_worker_ids_recorded(self):
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)
            t1.requires_grad = True
            t2.requires_grad = True
            for dst_rank in dst_ranks:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)
    
    @dist_init
    def test_error_in_context(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(6, 6, requires_grad=True)
            with self.assertRaises(RuntimeError):
                rpc.rpc_sync(worker_name(self._next_rank()), torch.matmul, args=(t1, t2))
    
    def _verify_backwards(self, exec_mode, tensors, context_id, local_grads, *args):
        if exec_mode == ExecMode.LOCAL:
            torch.autograd.backward(tensors)
            return [arg.grad for arg in args]
        else:
            self._verify_backwards_remote(tensors, context_id, local_grads, *args)
    
    def _verify_backwards_remote(self, tensors, context_id, local_grads, *args):
        dist_autograd.backward(context_id, tensors)
        grads = dist_autograd.get_gradients(context_id)
        nargs = len(args)
        ngrads = 0
        for i in range(0, nargs):
            if local_grads[i] is not None:
                self.assertIn(args[i], grads)
                self.assertEqual(local_grads[i], grads[args[i]])
                ngrads += 1
            else:
                self.assertNotIn(args[i], grads)
        self.assertEqual(ngrads, len(grads))
    
    @dist_init
    def test_backward_no_grad_on_tensor(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2)).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            self.assertIsNone(t1.grad)
            self.assertIsNone(t2.grad)
            loss_local = torch.add(t1, t2).sum()
            loss_local.backward()
            self.assertIsNotNone(t1.grad)
            self.assertIsNotNone(t2.grad)
            t1_grad_before = t1.grad
            t2_grad_before = t2.grad
            dist_autograd.backward(context_id, [loss])
            self.assertEqual(t1_grad_before, t1.grad)
            self.assertEqual(t2_grad_before, t2.grad)
    
    def _test_backward_simple(self, dst):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func_with_dst(dst, exec_mode, torch.add, t1, t2)
                loss = ret.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
                local_grads = (ret if ret else local_grads)
    
    @dist_init
    def test_backward_simple(self):
        self._test_backward_simple(self._next_rank())
    
    @dist_init
    def test_backward_simple_self(self):
        self._test_backward_simple(self.rank)
    
    def _test_backward_rref(self, callee, rref_owner):
        local_grads = None
        t1 = torch.ones((3, 3), requires_grad=True)
        t2 = torch.zeros((3, 3), requires_grad=True)
        local_ret = torch.add(t1, t2)
        local_ret.sum().backward()
        with dist_autograd.context() as context_id:
            rref_t1 = rpc.remote(rref_owner, _torch_ones, args=((3, 3), ), kwargs={'requires_grad': True})
            if callee == rref_owner:
                rref = rpc.remote(callee, my_rref_add, args=(rref_t1, t2))
            else:
                rref = rpc.remote(callee, my_nested_rref_add, args=(rref_owner, rref_t1, t2))
            ret = rref.to_here()
            dist_autograd.backward(context_id, [ret.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertIn(t2, grads)
            self.assertEqual(grads[t2], t2.grad)
            self.assertTrue(rpc.rpc_sync(rref_owner, _compare_owner_value, args=(context_id, rref_t1, t1.grad)))
    
    @dist_init
    def test_backward_rref(self):
        callee = worker_name(self._next_rank())
        rref_owner = callee
        self._test_backward_rref(callee, rref_owner)
    
    @dist_init
    def test_backward_rref_multi(self):
        if self.rank > 0:
            callee = 'worker0'
            rref_owner = callee
            self._test_backward_rref(callee, rref_owner)
    
    @dist_init
    def test_backward_rref_nested(self):
        callee = worker_name((self.rank + 1) % self.world_size)
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        self._test_backward_rref(callee, rref_owner)
    
    def _test_trainer_ps(self, create_ref_fn, trainer_fn):
        local_grads = None
        t1 = torch.ones((3, 3), requires_grad=True)
        t2 = torch.zeros((3, 3), requires_grad=True)
        local_ret = torch.add(t1, t2)
        local_ret.sum().backward()
        rref_t1 = rpc.remote(worker_name(self.rank), create_ref_fn, args=())
        rank_diffs = [1, 2, 3]
        futures = []
        for rank_diff in rank_diffs:
            futures.append(rpc.rpc_async(worker_name((self.rank + rank_diff) % self.world_size), trainer_fn, args=(rref_t1, t2, worker_name(self.rank), rank_diff)))
        for rank_diff in rank_diffs:
            self._check_rpc_done(rank_diff)
        accumulate_grad_func = None
        for rank_diff in rank_diffs:
            ctx_id = ctx_ids[rank_diff]
            grads = dist_autograd.get_gradients(ctx_id)
            local_t1 = rref_t1.to_here()
            self.assertIn(local_t1, grads)
            self.assertEqual(grads[local_t1], t1.grad)
        _set_rpc_done(None, 0)
        for fut in futures:
            fut.wait()
    
    @dist_init
    def test_trainer_ps(self):
        self._test_trainer_ps(create_tensor, _run_trainer)
    
    @dist_init
    def test_trainer_ps_torchscript_functions(self):
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = True
        self._test_trainer_ps(create_torchscript_tensor, _run_trainer_torchscript)
    
    @dist_init
    def test_backward_multiple_round_trips(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3))
        t3 = torch.rand((3, 3), requires_grad=True)
        t4 = torch.rand((3, 3))
        t5 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                val = self._exec_func(exec_mode, torch.add, t1, t2)
                val = self._exec_func(exec_mode, torch.mul, t3, val)
                s1 = self._exec_func(exec_mode, torch.stack, (t4, val))
                s2 = self._exec_func(exec_mode, torch.stack, (t5, val))
                val = self._exec_func(exec_mode, torch.bmm, s1, s2)
                val = self._exec_func(exec_mode, torch.matmul, val, val)
                loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t3, t4, t5)
                local_grads = (ret if ret else local_grads)
    
    @dist_init
    def test_backward_different_tensor_dims(self):
        local_grads = None
        t1 = torch.rand((4, 6), requires_grad=True)
        t2 = torch.rand((6, 5))
        t3 = torch.rand((5, 7), requires_grad=True)
        t4 = torch.rand((7, 9))
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                val = self._exec_func(exec_mode, torch.matmul, t1, t2)
                val = self._exec_func(exec_mode, torch.chain_matmul, [val, t3, t4])
                loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t2, t3, t4)
                local_grads = (ret if ret else local_grads)
    
    @dist_init
    def test_backward_unused_tensors(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        t3 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                s = self._exec_func(exec_mode, torch.stack, (t1, t2, t3))
                val = self._exec_func(exec_mode, torch.matmul, torch.narrow(s, 0, 0, 1), torch.narrow(s, 0, 2, 1))
                loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t3)
                local_grads = (ret if ret else local_grads)
    
    @dist_init
    def test_backward_multiple_output_tensors(self):
        local_grads = None
        t = torch.rand((10, 2), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                tensor_list = self._exec_func(exec_mode, torch.split, t, 2)
                t1 = tensor_list[0]
                t2 = tensor_list[2]
                t3 = tensor_list[4]
                val = self._exec_func(exec_mode, torch.chain_matmul, [t1, t2, t3])
                loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t)
                local_grads = (ret if ret else local_grads)
    
    def _run_test_backward_unused_send_function_in_thread(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            val = torch.mul(t1, t2)
            dist_autograd.backward(context_id, [val.sum()])
    
    @dist_init
    def test_backward_unused_send_function(self):
        t = threading.Thread(target=self._run_test_backward_unused_send_function_in_thread)
        t.daemon = True
        t.start()
        t.join(10)
        self.assertTrue(t.is_alive())
    
    @dist_init
    def test_backward_autograd_engine_error(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            tmp = (t1 + t2) * (t1 + t2)
            t3 = SimulateBackwardError.apply(tmp)
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t2, t3))
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.mul, args=(val, t2))
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.matmul, args=(val, t2))
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.div, args=(val, t2))
            with self.assertRaisesRegex(RuntimeError, 'Error on Node [0-9]+: Simulate error on backward pass'):
                dist_autograd.backward(context_id, [val.sum()])
    
    @dist_init(clean_shutdown=False)
    @unittest.skipIf(IS_MACOS, 'Test is flaky on MacOS since libuv error handling is not as robust as TCP')
    def test_backward_node_failure(self):
        rpc._set_rpc_timeout(timedelta(milliseconds=5000))
        initialize_pg(self.init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            dist.barrier()
            if self.rank % 2 == 0:
                shutdown_error_regex = get_shutdown_error_regex(dist_utils.TEST_CONFIG.rpc_backend_name)
                for rank in range(self.world_size):
                    if rank % 2 != 0:
                        wait_until_node_failure(rank, shutdown_error_regex)
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    dist_autograd.backward(context_id, [res.sum()])
            else:
                pass
    
    @dist_init
    def test_backward_without_context(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        context_id = 100
        with self.assertRaisesRegex(RuntimeError, 'Could not find autograd context with id: {}'.format(context_id)):
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            dist_autograd.backward(context_id, [res.sum()])
    
    @dist_init
    def test_backward_without_rpc(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = torch.add(t1, t2)
            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])
    
    @dist_init
    def test_backward_invalid_args(self):
        with dist_autograd.context() as context_id:
            with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
                dist_autograd.backward(context_id, None)
            with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
                dist_autograd.backward(None, None)
            with self.assertRaisesRegex(RuntimeError, 'No tensors provided for gradient computation'):
                dist_autograd.backward(context_id, [])
            with self.assertRaisesRegex(RuntimeError, 'requires_grad not set on'):
                t = torch.rand(3, 3)
                dist_autograd.backward(context_id, [t])
            with self.assertRaisesRegex(RuntimeError, 'is not a scalar, all roots need to be scalar'):
                t = torch.rand(3, 3, requires_grad=True)
                dist_autograd.backward(context_id, [t])
            with self.assertRaisesRegex(RuntimeError, 'does not have a valid gradient function'):
                t = torch.rand(1, requires_grad=True)
                dist_autograd.backward(context_id, [t])
    
    @dist_init
    def test_backward_multiple_roots(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
            with dist_autograd.context() as context_id:
                r1 = self._exec_func(exec_mode, torch.add, t1, t2).sum()
                r2 = self._exec_func(exec_mode, torch.mul, t1, t2).sum()
                r3 = self._exec_func(exec_mode, torch.cos, t1).sum()
                r4 = self._exec_func(exec_mode, torch.div, t1, t2).sum()
                local_grads = self._verify_backwards(exec_mode, [r1, r2, r3, r4], context_id, local_grads, t1, t2)
    
    @dist_init
    def test_backward_different_dtypes(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True, dtype=torch.float32)
        t2 = torch.rand((3, 3), requires_grad=True, dtype=torch.float64)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                loss = self._exec_func(exec_mode, torch.add, t1, t2).sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
    
    @dist_init
    def test_backward_simple_python_udf(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, my_py_add, t1, t2)
                loss = ret.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
    
    @dist_init
    def test_backward_simple_script_call(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.RPC_ASYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                forward_ret = self._exec_func(exec_mode, my_script_add, t1, t2)
                loss = forward_ret.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
                local_grads = (ret if ret else local_grads)
    
    @staticmethod
    def _complex_python_udf(t1, t2):
        t3 = torch.nn.functional.linear(t1, t2)
        t4 = torch.nn.functional.linear(t2, t3)
        t5 = torch.nn.functional.linear(t3, t4)
        return torch.chain_matmul(t1, t2, t3, t4, t5)
    
    @dist_init
    def test_backward_complex_python_udf(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, DistAutogradTest._complex_python_udf, t1, t2)
                loss = ret.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
    
    @staticmethod
    def _python_udf_with_backward_error(t1, t2):
        t3 = t1 + t2
        t4 = SimulateBackwardError.apply(t3)
        return torch.chain_matmul(t1, t2, t3, t4)
    
    @staticmethod
    def _nested_rpc_call_backward_error(t1, t2, dst):
        t1 = t1 * t2
        t2 = t1 + t2
        res = rpc.rpc_sync(worker_name(dst), DistAutogradTest._python_udf_with_backward_error, args=(t1, t2))
        return torch.chain_matmul(t1, t2, res)
    
    @dist_init
    def test_backward_python_udf_error(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._nested_rpc_call_backward_error, args=(t1, t2, self._next_rank()))
            with self.assertRaisesRegex(RuntimeError, 'Simulate error on backward pass'):
                dist_autograd.backward(context_id, [loss.sum()])
    _backward_done = False
    
    @staticmethod
    def _set_backward_done():
        DistAutogradTest._backward_done = True
    
    @staticmethod
    def _wait_backward_done():
        while not DistAutogradTest._backward_done:
            time.sleep(0.1)
    
    @dist_init(clean_shutdown=False)
    @unittest.skip('Test is flaky, see https://github.com/pytorch/pytorch/issues/35099')
    def test_backward_node_failure_python_udf(self):
        rpc._set_rpc_timeout(timedelta(milliseconds=5000))
        initialize_pg(self.init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            dst = self._next_rank()
            res = rpc.rpc_sync(worker_name(dst), my_py_nested_call, args=(t1, t2, dst, self.world_size, 1))
            dist.barrier()
            if self.rank == 2:
                return
            if self.rank == 0:
                shutdown_error_regex = get_shutdown_error_regex(dist_utils.TEST_CONFIG.rpc_backend_name)
                wait_until_node_failure(2, shutdown_error_regex)
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    dist_autograd.backward(context_id, [res.sum()])
                for i in range(self.world_size):
                    if (i != self.rank and i != 2):
                        try:
                            rpc.rpc_sync('worker{}'.format(i), DistAutogradTest._set_backward_done, args=())
                        except Exception as e:
                            pass
            else:
                DistAutogradTest._wait_backward_done()
    
    @staticmethod
    def _nested_python_udf(t1, t2, dst):
        t3 = t1 * t2
        t4 = t1 + t2
        res = rpc.rpc_sync(worker_name(dst), my_py_add, args=(t3, t4))
        return torch.chain_matmul(t1, t2, t3, t4, res)
    
    @dist_init
    def test_backwards_nested_python_udf(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        t3 = t1 * t2
        t4 = t1 + t2
        res = t3 + t4
        loss = torch.chain_matmul(t1, t2, t3, t4, res).sum()
        torch.autograd.backward([loss])
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._nested_python_udf, args=(t1, t2, self._next_rank()))
            dist_autograd.backward(context_id, [loss.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])
    _test_clean_context_backward_context_id = None
    
    
    class MyBackwardFunc(Function):
        
        @staticmethod
        def forward(ctx, input):
            return input
        
        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            assert DistAutogradTest._test_clean_context_backward_context_id is not None
            dist.barrier()
            dist_autograd._release_context(DistAutogradTest._test_clean_context_backward_context_id)
            assert _all_contexts_cleaned_up()
            return input
    
    
    @dist_init
    def test_clean_context_during_backward(self):
        """
        This test simulates the situation where the 'backward' call might throw
        an exception locally which would lead to the autograd context being
        cleaned up if we're using the context manager. As a result, the autograd
        context might be cleaned up while some threads are still using the
        autograd context.

        It is fine for the 'backward' call to throw an exception in this test,
        but the process should not crash.
        """
        initialize_pg(self.init_method, self.rank, self.world_size)
        context = dist_autograd._new_context()
        context_id = context._context_id()
        DistAutogradTest._test_clean_context_backward_context_id = context_id
        for i in range(0, self.world_size):
            if i != self.rank:
                rank_distance = (i - self.rank + self.world_size) % self.world_size
                rpc.rpc_sync(worker_name(i), _set_rpc_done, args=(context_id, rank_distance))
        dist.barrier()
        self.assertEqual(self.world_size - 1, len(known_context_ids))
        t1 = torch.rand((3, 3), requires_grad=True)
        for i in range(0, 100):
            dst = self._next_rank()
            t1 = rpc.rpc_sync(worker_name(dst), torch.add, args=(t1, t1))
        t1 = DistAutogradTest.MyBackwardFunc.apply(t1)
        self.assertEqual(100, len(context._send_functions()))
        context_id = 100
        with self.assertRaisesRegex(RuntimeError, 'Could not find autograd context with id: {}'.format(context_id)):
            dist_autograd.backward(context_id, [t1.sum()])
        dist.barrier()
        rpc.shutdown(graceful=False)
        sys.exit(0)
    
    @classmethod
    def _call_remote_embedding(cls, embedding_rref, input, offsets, per_sample_weights):
        embedding = embedding_rref.local_value()
        return embedding(input, offsets, per_sample_weights)
    
    @classmethod
    def _get_grad(cls, embedding_rref, context_id):
        embedding = embedding_rref.local_value()
        grad_map = dist_autograd.get_gradients(context_id)
        return grad_map[embedding.weight].to_dense()
    
    @dist_init
    def test_embedding_bag_with_no_grad_tensors(self):
        dst = self._next_rank()
        remote_embedding = rpc.remote(worker_name(dst), torch.nn.EmbeddingBag, args=(16, 16), kwargs={'mode': 'sum', 'sparse': True})
        local_embedding = torch.nn.EmbeddingBag(16, 16, mode='sum', sparse=True)
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        per_sample_weights = torch.rand(8, requires_grad=True)
        offsets = torch.LongTensor([0, 4])
        local_res = local_embedding(input, offsets, per_sample_weights)
        torch.autograd.backward([local_res.sum()], retain_graph=True)
        torch.autograd.backward([local_res.sum()])
        local_grad = local_embedding.weight.grad
        with dist_autograd.context() as context_id:
            res = rpc.rpc_sync(worker_name(dst), DistAutogradTest._call_remote_embedding, args=(remote_embedding, input, offsets, per_sample_weights))
            dist_autograd.backward(context_id, [res.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [res.sum()])
            remote_grad = rpc.rpc_sync(worker_name(dst), DistAutogradTest._get_grad, args=(remote_embedding, context_id))
            self.assertEqual(local_grad.to_dense(), remote_grad)
    
    @classmethod
    def _mixed_requires_grad(cls, t1, t2):
        if t2.requires_grad:
            return t1 - t2
        else:
            return t1 * t2
    
    @dist_init
    def test_mixed_requires_grad(self):
        for exec_mode in [ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=False)
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, DistAutogradTest._mixed_requires_grad, t1, t2)
                self.assertEqual(t1 * t2, ret)
                dist_autograd.backward(context_id, [ret.sum()])
                self.assertTrue(t1.requires_grad)
                self.assertFalse(t2.requires_grad)
                grads = dist_autograd.get_gradients(context_id)
                self.assertIn(t1, grads)
                self.assertNotIn(t2, grads)
                self.assertEqual(t2, grads[t1])
    
    
    class TestDebugInfoFunc(Function):
        
        @staticmethod
        def forward(ctx, input):
            return input
        
        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            debug_info = dist_autograd._get_debug_info()
            assert debug_info is not None
            backward_passes = int(debug_info['num_current_backward_passes'])
            assert (backward_passes >= 1 and backward_passes <= 4)
            return input
    
    
    @dist_init
    def test_debug_info(self):
        initialize_pg(self.init_method, self.rank, self.world_size)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            i = 0
            res = {}
            res[i] = t1
            for rank in range(self.world_size):
                if rank != self.rank:
                    res[i + 1] = rpc.rpc_sync(worker_name(rank), torch.add, args=(res[i], t2))
                    i += 1
            res[i + 1] = DistAutogradTest.TestDebugInfoFunc.apply(res[i])
            i += 1
            for rank in range(self.world_size):
                if rank != self.rank:
                    res[i + 1] = rpc.rpc_sync(worker_name(rank), torch.add, args=(res[i], t2))
                    i += 1
            dist_autograd.backward(context_id, [res[i].sum()])
            debug_info = dist_autograd._get_debug_info()
            num_autograd_context = int(debug_info['num_autograd_contexts'])
            self.assertTrue((num_autograd_context >= 1 and num_autograd_context <= 4))
        for rd in range(self.world_size - 1):
            rpc.rpc_sync(worker_name((self.rank + rd + 1) % self.world_size), _set_rpc_done, args=(context_id, rd + 1))
        dist.barrier()
        debug_info = dist_autograd._get_debug_info()
        assert debug_info is not None
        self.assertEqual(0, int(debug_info['num_current_backward_passes']))
        self.assertEqual(0, int(debug_info['local_autograd_engine_cpu_queue_size']))
        self.assertTrue(_all_contexts_cleaned_up())
        debug_info = dist_autograd._get_debug_info()
        self.assertEqual(0, int(debug_info['num_autograd_contexts']))
    
    @staticmethod
    def _workload_thread():
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            t3 = rpc.rpc_sync('worker0', torch.add, args=(t1, t2))
            t4 = rpc.rpc_sync('worker0', torch.mul, args=(t2, t3))
            t5 = rpc.rpc_sync('worker0', torch.matmul, args=(t3, t4))
            t6 = rpc.rpc_sync('worker0', torch.add, args=(t4, t5))
            dist_autograd.backward(context_id, [t6.sum()])
    
    @dist_init
    def test_async_dist_autograd(self):
        """
        This test ensures async processing for distributed autograd works
        appropriately. This is achieved by spawning multiple threads and
        hammering a single node with a lot of backward() calls.
        """
        initialize_pg(self.init_method, self.rank, self.world_size)
        if self.rank != 0:
            threads = []
            for i in range(20):
                t = threading.Thread(target=DistAutogradTest._workload_thread)
                t.start()
                threads.append(t)
            for thread in threads:
                thread.join()
        dist.barrier()
    
    @dist_init
    def test_backward_accumulate_grads(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            t3 = torch.matmul(t1, t2)
            torch.autograd.backward([t3.sum()], retain_graph=True)
            torch.autograd.backward([t3.sum()])
            t3 = rpc.rpc_sync(worker_name(self._next_rank()), torch.matmul, args=(t1, t2))
            dist_autograd.backward(context_id, [t3.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])
    
    @staticmethod
    def _test_nested_backward_accumulate_grads(t1, t2, dst_rank):
        return rpc.rpc_sync(worker_name(dst_rank), torch.matmul, args=(t1, t2))
    
    @dist_init
    def test_nested_backward_accumulate_grads(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._test_nested_backward_accumulate_grads, args=(t1, t2, self._next_rank())).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            dist_autograd.backward(context_id, [loss])
    
    @dist_init
    def test_multiple_backward(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2)).sum()
            for i in range(1000):
                dist_autograd.backward(context_id, [loss], retain_graph=True)
    
    @dist_init(clean_shutdown=False)
    def test_multiple_backward_with_errors(self):
        initialize_pg(self.init_method, self.rank, self.world_size)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync('worker{}'.format(self._next_rank()), DistAutogradTest._python_udf_with_backward_error, args=(t1, t2)).sum()
            try:
                for i in range(100):
                    if i < 50:
                        with self.assertRaisesRegex(RuntimeError, 'Simulate error on backward pass'):
                            dist_autograd.backward(context_id, [loss], retain_graph=True)
                    elif i > 50:
                        dist_autograd.backward(context_id, [loss], retain_graph=True)
                    else:
                        dist.barrier()
                        SimulateBackwardError._simulate_error = False
                        dist.barrier()
            finally:
                dist.barrier()
                SimulateBackwardError._simulate_error = True
    
    @dist_init
    def test_backward_verify_hooks(self):
        t1 = torch.ones((3, 3), requires_grad=True)
        t1.register_hook(lambda grad: grad * 2)
        t2 = torch.ones((3, 3), requires_grad=True)
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, torch.matmul, t1, t2)
                loss = ret.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
                local_grads = (ret if ret else local_grads)
    
    @dist_init
    def test_no_grad_copy(self):
        """
        Similar to test in test_autograd.py.
        """
        
        
        class MyFunc(Function):
            static_grad_ptr = None
            
            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2
            
            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return (grad, grad)
        
        
        
        class MyFuncSingleGrad(Function):
            static_grad_ptr = None
            
            @staticmethod
            def forward(ctx, inp):
                return inp
            
            @staticmethod
            def backward(ctx, grad):
                MyFuncSingleGrad.static_grad_ptr = grad.data_ptr()
                return grad
        
        
        
        class NonContGradFunc(Function):
            
            @staticmethod
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.0])
            
            @staticmethod
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)
        
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [NonContGradFunc.apply(MyFunc.apply(a, b))])
            grads = dist_autograd.get_gradients(context_id)
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFuncSingleGrad.apply(a)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFuncSingleGrad.static_grad_ptr
            p_a = grads[a].data_ptr()
            self.assertTrue(p_a == p_g)
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFunc.apply(a, b)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a].data_ptr()
            p_b = grads[b].data_ptr()
            self.assertFalse(p_a == p_b)
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)


