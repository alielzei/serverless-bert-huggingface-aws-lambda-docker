from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Manager
import numpy as np
import unittest
import tempfile
import shutil
import logging
from hypothesis import given
import hypothesis.strategies as st
log = logging.getLogger('parallelize_bmuf_distributed_test')
log.setLevel(logging.INFO)

def bmuf_process(filestore_dir, process_id, shared_results, cpu_device=False, nesterov=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.parallelize_bmuf_distributed_test.bmuf_process', 'bmuf_process(filestore_dir, process_id, shared_results, cpu_device=False, nesterov=False)', {'log': log, 'np': np, 'filestore_dir': filestore_dir, 'process_id': process_id, 'shared_results': shared_results, 'cpu_device': cpu_device, 'nesterov': nesterov}, 1)


class DistributedTest(unittest.TestCase):
    
    @given(cpu_device=st.booleans(), nesterov=st.booleans())
    def test_bmuf_distributed(self, cpu_device, nesterov):
        self._test_bmuf_distributed(cpu_device=cpu_device, nesterov=nesterov)
    
    def _test_bmuf_distributed(self, cpu_device=False, nesterov=False):
        processes = []
        filestore_dir = tempfile.mkdtemp()
        results = Manager().dict()
        for idx in range(0, 2):
            process = Process(target=bmuf_process, args=(filestore_dir, idx, results, cpu_device, nesterov))
            processes.append(process)
            process.start()
        while len(processes) > 0:
            process = processes.pop()
            process.join()
        shutil.rmtree(filestore_dir)
        if len(results) == 0:
            return
        w_0 = results[0]['w_0']
        w_1 = results[0]['w_1']
        b_0 = results[0]['b_0']
        b_1 = results[0]['b_1']
        np.testing.assert_equal(w_0, w_1)
        np.testing.assert_equal(w_0, results[1]['w_0'])
        np.testing.assert_equal(w_0, results[1]['w_1'])
        np.testing.assert_equal(b_0, b_1)
        np.testing.assert_equal(b_0, results[1]['b_0'])
        np.testing.assert_equal(b_0, results[1]['b_1'])
        w_g_ = results[0]['w_g_']
        b_g_ = results[0]['b_g_']
        g_b = (results[0]['b_0_'] + results[1]['b_0_'] + results[0]['b_1_'] + results[1]['b_1_']) / 4 - b_g_
        g_w = (results[0]['w_0_'] + results[1]['w_0_'] + results[0]['w_1_'] + results[1]['w_1_']) / 4 - w_g_
        v_b_ = results[0]['v_b_']
        v_b = results[0]['v_b']
        v_w_ = results[0]['v_w_']
        v_w = results[0]['v_w']
        for pid in results.keys():
            for k in results[pid].keys():
                if k.startswith('sync_num'):
                    self.assertEqual(2603, results[pid][k])
        np.testing.assert_almost_equal(v_b, 0.75 * v_b_ + g_b)
        np.testing.assert_almost_equal(v_w, 0.75 * v_w_ + g_w)
        if nesterov:
            np.testing.assert_equal(w_0, w_g_ + v_w - 0.75 * (v_w - v_w_))
            np.testing.assert_equal(b_0, b_g_ + v_b - 0.75 * (v_b - v_b_))
        else:
            np.testing.assert_equal(w_0, w_g_ + v_w)
            np.testing.assert_equal(b_0, b_g_ + v_b)


