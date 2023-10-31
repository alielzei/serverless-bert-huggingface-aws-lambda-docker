from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace, net_drawer
from caffe2.proto import caffe2_pb2

def getGradientForOp(op):
    return core.GradientRegistry.GetGradientForOp(op, [s + '_grad' for s in op.output])

def _get_grad_blob(grad_map, input_to_check):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.gradient_checker._get_grad_blob', '_get_grad_blob(grad_map, input_to_check)', {'core': core, 'workspace': workspace, 'grad_map': grad_map, 'input_to_check': input_to_check}, 1)

def _get_grad(net, outputs, outputs_with_grad, input_values, inputs_with_grads):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.gradient_checker._get_grad', '_get_grad(net, outputs, outputs_with_grad, input_values, inputs_with_grads)', {'workspace': workspace, '_get_grad_blob': _get_grad_blob, 'net': net, 'outputs': outputs, 'outputs_with_grad': outputs_with_grad, 'input_values': input_values, 'inputs_with_grads': inputs_with_grads}, 3)

def _assert_close(value1, value2, threshold, err_msg=''):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.gradient_checker._assert_close', "_assert_close(value1, value2, threshold, err_msg='')", {'np': np, 'value1': value1, 'value2': value2, 'threshold': threshold, 'err_msg': err_msg}, 2)


class NetGradientChecker(object):
    
    @staticmethod
    def CompareNets(nets, outputs, outputs_with_grad_ids, inputs_with_grads, input_values=None, threshold=1e-07, print_net_images=False):
        
        def _get_output_with_grad_names(net_outputs):
            return [net_outputs[i] for i in outputs_with_grad_ids]
        if print_net_images:
            for (i, net) in enumerate(nets):
                png = net_drawer.GetPydotGraph(net).create_png()
                with open('caffe2_net_forward_' + str(i) + net.Name() + '.png', 'wb') as f:
                    f.write(png)
        results = [_get_grad(net, net_outputs, _get_output_with_grad_names(net_outputs), input_values, inputs_with_grads) for (net, net_outputs) in zip(nets, outputs)]
        if print_net_images:
            (_, _, backward_nets) = zip(*results)
            for (i, net) in enumerate(backward_nets):
                png = net_drawer.GetPydotGraph(net).create_png()
                with open('caffe2_net_' + str(i) + net.Name() + '.png', 'wb') as f:
                    f.write(png)
        (first_net_results, first_net_grads, _) = results[0]
        for (net_results, net_grads, _) in results[1:]:
            assert len(net_results) == len(first_net_results)
            for (idx, ((blob1, blob_value1), (blob2, blob_value2))) in enumerate(zip(first_net_results, net_results)):
                _assert_close(blob_value1, blob_value2, threshold, err_msg='Different forward pass results for output id {}. Corresponding output blobs: {} and {}'.format(idx, blob1, blob2))
            assert net_grads.keys() == first_net_grads.keys()
            for (blob, blob_grad_value) in net_grads.items():
                _assert_close(first_net_grads[blob], blob_grad_value, threshold, err_msg='Different gradients for input {}'.format(blob))
    
    @staticmethod
    def Check(net, outputs_with_grad, input_values, input_to_check, step_size=0.0001, threshold=0.05, print_net=True):
        (net_results, net_grads, full_net) = _get_grad(net, [], outputs_with_grad, input_values, [input_to_check])
        analytic_grad = net_grads[input_to_check]
        
        def GetLoss(new_value):
            workspace.blobs[input_to_check] = new_value
            workspace.RunNetOnce(full_net)
            return sum([workspace.blobs[output] for output in outputs_with_grad]).sum()
        
        def GetValue(dim, delta):
            input_value = input_values[input_to_check].copy()
            input_value.flat[dim] += delta
            return input_value
        grad_estimate = np.zeros_like(input_values[input_to_check])
        for dim in range(input_values[input_to_check].size):
            pos_loss = GetLoss(GetValue(dim, step_size))
            neg_loss = GetLoss(GetValue(dim, -step_size))
            grad_estimate.flat[dim] = (pos_loss - neg_loss) / step_size / 2
        err_msg = 'Error in gradient check for net_copy {}'.format(net.Name())
        if print_net:
            err_msg += ': {}'.format(net.Proto())
        return _assert_close(analytic_grad, grad_estimate, threshold, err_msg)



class GradientChecker:
    """A gradient checker in Python.

    This is not the most efficient way to check gradients, as the Python
    interface will involve a lot of copies back and forth operations. Use at your
    own risk.
    """
    
    def __init__(self, stepsize, threshold, device_option=None, workspace_name='gradient_check', input_device_options=None):
        self._stepsize = stepsize
        self._threshold = threshold
        self._device_option = (device_option or caffe2_pb2.DeviceOption())
        self._workspace_name = workspace_name
        if input_device_options is None:
            self._input_device_options = {}
        else:
            self._input_device_options = input_device_options
    
    def GetLossAndGrad(self, op, grad_ops, inputs, input_names, input_to_check, grad_name, outputs_with_grads):
        for i in range(len(inputs)):
            workspace.FeedBlob(input_names[i], inputs[i], self._input_device_options.get(input_names[i], self._device_option))
        x = inputs[input_to_check]
        workspace.RunOperatorOnce(op)
        loss = 0.0
        for idx in outputs_with_grads:
            name = op.output[idx]
            arr = workspace.FetchBlob(name)
            loss += (arr**2).sum()
            workspace.FeedBlob(name + '_grad', arr, self._device_option)
        loss /= 2.0
        workspace.RunOperatorsOnce(grad_ops)
        if isinstance(grad_name, core.GradientSlice):
            workspace.FeedBlob('zeros', np.zeros_like(x, dtype=np.float32))
            workspace.FeedBlob('ones', np.ones(1, dtype=np.float32))
            gv_cpu_op = core.CreateOperator('EnsureCPUOutput', grad_name.values, grad_name.values + '_cpu', device_option=self._device_option)
            gi_cpu_op = core.CreateOperator('EnsureCPUOutput', grad_name.indices, grad_name.indices + '_cpu', device_option=self._device_option)
            sparse_to_dense_op = core.CreateOperator('ScatterWeightedSum', ['zeros', 'ones', grad_name.indices + '_cpu', grad_name.values + '_cpu', 'ones'], 'zeros')
            workspace.RunOperatorOnce(gv_cpu_op)
            workspace.RunOperatorOnce(gi_cpu_op)
            workspace.RunOperatorOnce(sparse_to_dense_op)
            grad = workspace.FetchBlob('zeros')
        else:
            grad = workspace.FetchBlob(grad_name)
        return (loss, grad)
    
    def CheckSimple(self, op, inputs, input_to_check, outputs_with_grads, grad_ops=None, input_device_options=None):
        """Checks the operator in a very simple fashion by stacking a sum of
        squares on the top.

        Inputs:
          op: the operator to be checked.
          inputs: the input data in numpy arrays.
          input_to_check: an index specifying which input blob we should
              check.
          outputs_with_grads: indices specifying which output blobs will we
              need to check gradients with. For these outputs, we will collect a
              squared sum and also feed in their gradients.
          grad_operator: the gradient operator. If not given, we will get the
              gradient operator from the gradient registry.
          input_device_options: an optional mapping from input names to
              DeviceOptions (to override the default DeviceOption)
        Outputs:
          boolean: True if it passes, False if it does not pass.
        """
        old_ws_name = workspace.CurrentWorkspace()
        if self._workspace_name != old_ws_name:
            workspace.SwitchWorkspace(self._workspace_name, True)
        op.device_option.CopyFrom(self._device_option)
        if grad_ops is None:
            (grad_ops, g_input) = getGradientForOp(op)
        _input_device_options = (input_device_options or core.InferOpBlobDevicesAsDict(op)[0])
        for (i, arr) in enumerate(inputs):
            workspace.FeedBlob(op.input[i], arr, _input_device_options.get(op.input[i], self._device_option))
        grad_name = g_input[input_to_check]
        (loss, grad) = self.GetLossAndGrad(op, grad_ops, inputs, op.input, input_to_check, grad_name, outputs_with_grads)
        grad_estimate = np.zeros_like(inputs[input_to_check])
        if grad_estimate.shape != grad.shape:
            raise Exception('Mismatched gradient shapes: estimated ({}), grad ({})'.format(grad_estimate.shape, grad.shape))
        dims_to_check = inputs[input_to_check].size
        for current_dim in range(dims_to_check):
            inputs[input_to_check].flat[current_dim] += self._stepsize
            (pos_loss, _) = self.GetLossAndGrad(op, grad_ops, inputs, op.input, input_to_check, grad_name, outputs_with_grads)
            inputs[input_to_check].flat[current_dim] -= self._stepsize * 2
            (neg_loss, _) = self.GetLossAndGrad(op, grad_ops, inputs, op.input, input_to_check, grad_name, outputs_with_grads)
            inputs[input_to_check].flat[current_dim] += self._stepsize
            grad_estimate.flat[current_dim] = (pos_loss - neg_loss) / self._stepsize / 2
        fail_mat = ~np.isclose(grad, grad_estimate, atol=self._threshold, rtol=self._threshold)
        if np.any(fail_mat):
            idx = np.flatnonzero(fail_mat)
            print('Failed. [idx, grad, grad_estimate] are:')
            print(np.vstack([idx, grad.flat[idx], grad_estimate.flat[idx]]).T)
            ret = False
        else:
            ret = True
        if self._workspace_name != old_ws_name:
            workspace.ResetWorkspace()
            workspace.SwitchWorkspace(old_ws_name)
        return (ret, grad, grad_estimate)


