import unittest
from typing import Dict, Tuple
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.testing._internal.dist_utils import dist_init, worker_name, initialize_pg
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
from torch.testing._internal.common_utils import TemporaryFileName

def rpc_return_rref(dst):
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))


class MyScriptModuleWithRRefs(torch.jit.ScriptModule):
    
    def __init__(self, dst_worker):
        super().__init__()
        self.rrefs = []
        for _ in range(4):
            self.rrefs.append(rpc_return_rref(dst_worker))
    
    @torch.jit.script_method
    def forward(self):
        res_tensor = torch.ones(2, 2)
        for rref in self.rrefs:
            res_tensor += rref.to_here()
        return res_tensor



@torch.jit.script
class MyScriptClass:
    
    def __init__(self, a):
        self.a = a
    
    def get_value(self):
        return self.a



@torch.jit.interface
class MyModuleInterface(torch.nn.Module):
    
    def forward(self):
        pass



class MyScriptModule(torch.jit.ScriptModule):
    
    def __init__(self, rank):
        super().__init__()
        self.a = torch.ones(rank)
    
    @torch.jit.script_method
    def forward(self):
        return self.a


def owner_create_rref_my_script_class(a):
    return rpc.RRef(MyScriptClass(a))

def owner_create_rref_my_script_module(a):
    return rpc.RRef(MyScriptModule(a), MyModuleInterface)

@torch.jit.script
def script_run_get_value_rref_my_script_class(rref):
    return rref.to_here().get_value()

@torch.jit.script
def script_run_forward_rref_my_script_module(rref):
    return rref.to_here().forward()


class LocalRRefTest(RpcAgentTestFixture):
    
    @dist_init
    def test_create_local_script_class_rref_in_py(self):
        if self.rank != 0:
            return
        rref_script_class = rpc.RRef(MyScriptClass(self.rank))
        ret = rref_script_class.to_here().get_value()
        self.assertEqual(ret, self.rank)
    
    @dist_init
    def test_create_local_script_module_rref_in_py(self):
        if self.rank != 0:
            return
        rref_script_module = rpc.RRef(MyScriptModule(self.rank), MyModuleInterface)
        ret = rref_script_module.to_here().forward()
        self.assertEqual(ret, torch.ones(self.rank))
        with self.assertRaisesRegex(RuntimeError, 'The RRef being created contains a ScriptModule, must provide its ModuleInterface type hint.'):
            rref_script_module = rpc.RRef(MyScriptModule(self.rank))
    
    @dist_init
    def test_return_local_script_class_rref_in_py_and_use_in_script(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        rref = rpc.rpc_sync(dst_worker_name, owner_create_rref_my_script_class, args=(self.rank, ))
        
        def use_rref_on_owner(rref):
            args = (rref, )
            kwargs: Dict[(str, Any)] = {}
            fut = rpc.rpc_async(rref.owner(), script_run_get_value_rref_my_script_class, args, kwargs)
            ret = fut.wait()
            return ret
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, self.rank)
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, self.rank)
    
    @dist_init
    def test_return_local_script_module_rref_in_py_and_use_in_script(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        rref = rpc.rpc_sync(dst_worker_name, owner_create_rref_my_script_module, args=(self.rank, ))
        
        def use_rref_on_owner(rref):
            args = (rref, )
            kwargs: Dict[(str, Any)] = {}
            fut = rpc.rpc_async(rref.owner_name(), script_run_forward_rref_my_script_module, args, kwargs)
            ret = fut.wait()
            return ret
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, torch.ones(self.rank))
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, torch.ones(self.rank))


def python_function():
    return 0

@torch.jit.script
def no_arg():
    return 0

@torch.jit.script
def two_args_two_kwargs(first_arg, second_arg, first_kwarg=torch.tensor([3, 3]), second_kwarg=torch.tensor([4, 4])):
    return first_arg + second_arg + first_kwarg + second_kwarg

@torch.jit.script
def assorted_types_args_kwargs(tensor_arg: Tensor, str_arg: str, int_arg: int, tensor_kwarg: Tensor = torch.tensor([2, 2]), str_kwarg: str = 'str_kwarg', int_kwarg: int = 2):
    return (tensor_arg + tensor_kwarg, str_arg + str_kwarg, int_arg + int_kwarg)

@torch.jit.script
def raise_script():
    raise RuntimeError('Expected error')
    return 0

@torch.jit.script
def rpc_async_call_remote_torchscript_in_torchscript(dst_worker_name: str, args: Tuple[(Tensor, Tensor)], kwargs: Dict[(str, Tensor)]):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.distributed.rpc.jit.rpc_test.rpc_async_call_remote_torchscript_in_torchscript', 'rpc_async_call_remote_torchscript_in_torchscript(dst_worker_name, args, kwargs)', {'rpc': rpc, 'two_args_two_kwargs': two_args_two_kwargs, 'torch': torch, 'dst_worker_name': dst_worker_name, 'args': args, 'kwargs': kwargs, 'Tuple': Tuple, 'Dict': Dict}, 1)


class JitRpcAsyncOpTest:
    
    @dist_init
    def test_all_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {}
        ret = rpc_async_call_remote_torchscript_in_torchscript(dst_worker_name, args, kwargs)
        self.assertEqual(ret, torch.tensor([10, 10]))
    
    @dist_init
    def test_some_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {'first_kwarg': torch.tensor([2, 2])}
        ret = rpc_async_call_remote_torchscript_in_torchscript(dst_worker_name, args, kwargs)
        self.assertEqual(ret, torch.tensor([9, 9]))
    
    @dist_init
    def test_no_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {'first_kwarg': torch.tensor([2, 2]), 'second_kwarg': torch.tensor([3, 3])}
        ret = rpc_async_call_remote_torchscript_in_torchscript(dst_worker_name, args, kwargs)
        self.assertEqual(ret, torch.tensor([8, 8]))
    
    @dist_init
    def test_kwargs_in_the_front_can_be_specified_by_extra_args(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_with_extra_arg(dst_worker_name: str):
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]), torch.tensor([2, 2]))
            kwargs = {'second_kwarg': torch.tensor([3, 3])}
            fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            ret = fut.wait()
            return ret
        ret = rpc_async_call_remote_torchscript_in_torchscript_with_extra_arg(dst_worker_name)
        self.assertEqual(ret, torch.tensor([8, 8]))
    
    @dist_init
    def test_args_and_kwargs_contain_different_types(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_with_assorted_types(dst_worker_name: str):
            args = (torch.tensor([1, 1]), 'str_arg', 1)
            kwargs: Dict[(str, Any)] = {'tensor_kwarg': torch.tensor([3, 3]), 'str_kwarg': '_str_kwarg', 'int_kwarg': 3}
            fut = rpc.rpc_async(dst_worker_name, assorted_types_args_kwargs, args, kwargs)
            ret = fut.wait()
            return ret
        ret = rpc_async_call_remote_torchscript_in_torchscript_with_assorted_types(dst_worker_name)
        self.assertEqual(ret, (torch.tensor([4, 4]), 'str_arg_str_kwarg', 4))
    
    @dist_init
    def test_kwargs_not_passed(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_without_kwargs_passed(dst_worker_name: str):
            args = ()
            fut = rpc.rpc_async(dst_worker_name, no_arg, args)
            ret = fut.wait()
            return ret
        ret = rpc_async_call_remote_torchscript_in_torchscript_without_kwargs_passed(dst_worker_name)
        self.assertEqual(ret, 0)
    
    @dist_init
    def test_args_kwargs_are_neither_passed(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_without_args_kwargs_passed(dst_worker_name: str):
            fut = rpc.rpc_async(dst_worker_name, no_arg)
            ret = fut.wait()
            return ret
        ret = rpc_async_call_remote_torchscript_in_torchscript_without_args_kwargs_passed(dst_worker_name)
        self.assertEqual(ret, 0)
    
    @dist_init
    def test_less_than_needed_args_are_specified(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'Argument second_arg not provided'):
            
            @torch.jit.script
            def rpc_async_call_remote_torchscript_in_torchscript_with_less_args(dst_worker_name: str):
                args = (torch.tensor([1, 1]), )
                kwargs = {}
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                ret = fut.wait()
                return ret
    
    @dist_init
    def test_more_than_needed_args_are_specified(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'Expected at most 4 arguments but found 5 positional arguments'):
            
            @torch.jit.script
            def rpc_async_call_remote_torchscript_in_torchscript_with_more_args(dst_worker_name: str):
                args = (torch.tensor([1, 1]), torch.tensor([2, 2]), torch.tensor([3, 3]), torch.tensor([4, 4]), torch.tensor([5, 5]))
                kwargs = {}
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                ret = fut.wait()
                return ret
    
    @dist_init
    def test_unexepected_kwarg_is_specified(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_with_unexpected_kwarg(dst_worker_name: str):
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {'third_kwarg': torch.tensor([1, 1])}
            fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, "Unknown keyword argument 'third_kwarg'"):
            ret = rpc_async_call_remote_torchscript_in_torchscript_with_unexpected_kwarg(dst_worker_name)
            self.assertEqual(ret, 0)
    
    @dist_init
    def test_call_python_function_remotely_from_script_not_supported(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def rpc_async_call_remote_py_function_in_torchscript(dst_worker_name: str):
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, python_function, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, 'attempted to get undefined function'):
            ret = rpc_async_call_remote_py_function_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)
    
    @dist_init
    def test_call_script_function_that_raises_remotely_from_script(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def rpc_async_call_remote_raising_torchscript_in_torchscript(dst_worker_name: str):
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, raise_script, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, 'Exception'):
            ret = rpc_async_call_remote_raising_torchscript_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)
    
    @dist_init
    def test_call_script_function_that_not_exists_remotely_from_script(self):
        if self.rank != 0:
            return
        dst_worker_name = 'worker{}'.format((self.rank + 1) % self.world_size)
        
        @torch.jit.script
        def nonexisting_script():
            return 0
        
        @torch.jit.script
        def rpc_async_call_remote_nonexisting_torchscript_in_torchscript(dst_worker_name: str):
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, nonexisting_script, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, 'attempted to get undefined function nonexisting_script'):
            ret = rpc_async_call_remote_nonexisting_torchscript_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)


@torch.jit.script
def one_arg(value):
    return value + 1

@torch.jit.script
def rref_to_here(rref_var):
    return rref_var.to_here()

@torch.jit.script
def return_rref(rref_var):
    return rref_var

@torch.jit.ignore
def my_script_module_init(rank):
    return MyScriptModule(rank)

@torch.jit.script
def construct_my_script_module(rank):
    return my_script_module_init(rank)

@torch.jit.script
def run_ref_script_module(ref_script_module, t):
    module = ref_script_module.to_here()
    return module.forward() + t

@torch.jit.ignore
def rref_python_annotation(rref_var):
    return rref_var

@torch.jit.script
def rref_script_annotation(rref_var):
    return rref_python_annotation(rref_var).to_here()

@torch.jit.script
def script_check_rref_confirmed(rref):
    return rref.confirmed_by_owner()

@torch.jit.script
def save_rref(rref_var, fname):
    torch.save(rref_var, fname)


@unittest.skipIf(not torch._six.PY3, 'Pytorch distributed rpc package does not support python2')
class JitRpcTest(LocalRRefTest, JitRpcAsyncOpTest, RpcAgentTestFixture):
    
    @dist_init
    def test_torchscript_function(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        local_ret = one_arg(torch.ones(2, 2))
        ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(torch.ones(2, 2), ))
        self.assertEqual(ret, local_ret)
        rref = rpc.remote(dst_worker_name, one_arg, args=(torch.ones(2, 2), ))
        self.assertEqual(rref.to_here(), local_ret)
        local_rref = rpc.remote(worker_name(self.rank), one_arg, args=(torch.ones(2, 2), ))
        self.assertEqual(local_rref.to_here(), local_ret)
    
    @dist_init
    def test_torchscript_function_exception(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'one_arg\\(\\) expected at most'):
            ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(10, 20))
        with self.assertRaisesRegex(RuntimeError, 'one_arg\\(\\) expected at most'):
            rref = rpc.remote(dst_worker_name, one_arg, args=(10, 20))
    
    @dist_init
    def test_torchscript_functions_not_supported(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        my_local_script_module = MyScriptModule(self.rank)
        initialize_pg(self.init_method, self.rank, self.world_size)
        dist.barrier()
        ret = rpc.rpc_sync(dst_worker_name, MyScriptClass, args=(self.rank, ))
        with self.assertRaisesRegex(RuntimeError, 'ScriptModules cannot be deepcopied'):
            ret = rpc.rpc_sync(dst_worker_name, MyScriptModule, args=(self.rank, ))
        with self.assertRaisesRegex(TypeError, 'pickle'):
            ret = rpc.rpc_async(dst_worker_name, my_local_script_module.forward, args=())
    
    @dist_init
    def test_rref_as_arg_and_return(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        local_ret = one_arg(torch.ones(2, 2))
        rref = rpc.remote(worker_name(self.rank), one_arg, args=(torch.ones(2, 2), ))
        ret = rpc.rpc_sync(worker_name(dst_rank), rref_to_here, args=(rref, ))
        self.assertEqual(ret, local_ret)
        rref1 = rpc.rpc_sync(worker_name(dst_rank), return_rref, args=(rref, ))
        self.assertEqual(rref1.to_here(), local_ret)
        rref2 = rpc.remote(worker_name(dst_rank), rref_to_here, args=(rref, ))
        self.assertEqual(rref2.to_here(), local_ret)
        rref3 = rpc.remote(worker_name(dst_rank), return_rref, args=(rref, ))
        self.assertEqual(rref3.to_here().to_here(), local_ret)
    
    @dist_init
    def test_remote_script_module(self):
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = True
        local_ret = torch.ones(self.rank) + torch.ones(self.rank)
        n = self.rank + 1
        dst_rank = n % self.world_size
        remote_ref = rpc.remote(worker_name(dst_rank), construct_my_script_module, args=(self.rank, ))
        ret = rpc.rpc_sync(worker_name(dst_rank), run_ref_script_module, args=(remote_ref, torch.ones(self.rank)))
        self.assertEqual(ret, local_ret)
    
    @dist_init
    def test_rref_is_owner(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_var = rpc_return_rref(worker_name(dst_rank))
        
        @torch.jit.script
        def rref_tensor_is_owner(rref_var):
            return rref_var.is_owner()
        res = rref_tensor_is_owner(rref_var)
        self.assertEqual(res, False)
    
    @dist_init
    def test_my_script_module_with_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        module_with_rrefs = MyScriptModuleWithRRefs(worker_name(dst_rank))
        res = module_with_rrefs()
        self.assertEqual(res, torch.ones(2, 2) * 9)
    
    @dist_init
    def test_rref_python_annotation(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_var = rpc_return_rref(worker_name(dst_rank))
        res = rref_script_annotation(rref_var)
        self.assertEqual(res, torch.ones(2, 2) + 1)
    
    def _create_rref(self):
        owner_rank = (self.rank + 2) % self.world_size
        return rpc.remote('worker{}'.format(owner_rank), torch.add, args=(torch.zeros(2, 2), 1))
    
    @dist_init
    def test_user_rrefs_confirmed(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret = rpc.rpc_sync('worker{}'.format(dst_rank), script_check_rref_confirmed, args=(rref, ))
        self.assertEqual(ret, True)
    
    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret_rref = rpc.remote('worker{}'.format(dst_rank), script_check_rref_confirmed, args=(rref, ))
        self.assertEqual(ret_rref.to_here(), True)
    
    @dist_init
    def test_rref_jit_pickle_not_supported(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_var = rpc_return_rref(worker_name(dst_rank))
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, 'RRef jit pickling is only allowed inside RPC calls'):
                save_rref(rref_var, fname)


