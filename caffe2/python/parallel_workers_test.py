from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest
from caffe2.python import workspace, core
import caffe2.python.parallel_workers as parallel_workers

def create_queue():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.parallel_workers_test.create_queue', 'create_queue()', {'workspace': workspace, 'core': core}, 1)

def create_worker(queue, get_blob_data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.parallel_workers_test.create_worker', 'create_worker(queue, get_blob_data)', {'workspace': workspace, 'core': core, 'queue': queue, 'get_blob_data': get_blob_data}, 1)

def dequeue_value(queue):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.parallel_workers_test.dequeue_value', 'dequeue_value(queue)', {'workspace': workspace, 'core': core, 'queue': queue}, 1)


class ParallelWorkersTest(unittest.TestCase):
    
    def testParallelWorkers(self):
        workspace.ResetWorkspace()
        queue = create_queue()
        dummy_worker = create_worker(queue, lambda worker_id: str(worker_id))
        worker_coordinator = parallel_workers.init_workers(dummy_worker)
        worker_coordinator.start()
        for _ in range(10):
            value = dequeue_value(queue)
            self.assertTrue(value in [b'0', b'1'], 'Got unexpected value ' + str(value))
        self.assertTrue(worker_coordinator.stop())
    
    def testParallelWorkersInitFun(self):
        workspace.ResetWorkspace()
        queue = create_queue()
        dummy_worker = create_worker(queue, lambda worker_id: workspace.FetchBlob('data'))
        workspace.FeedBlob('data', 'not initialized')
        
        def init_fun(worker_coordinator, global_coordinator):
            workspace.FeedBlob('data', 'initialized')
        worker_coordinator = parallel_workers.init_workers(dummy_worker, init_fun=init_fun)
        worker_coordinator.start()
        for _ in range(10):
            value = dequeue_value(queue)
            self.assertEqual(value, b'initialized', 'Got unexpected value ' + str(value))
        worker_coordinator.stop()
    
    def testParallelWorkersShutdownFun(self):
        workspace.ResetWorkspace()
        queue = create_queue()
        dummy_worker = create_worker(queue, lambda worker_id: str(worker_id))
        workspace.FeedBlob('data', 'not shutdown')
        
        def shutdown_fun():
            workspace.FeedBlob('data', 'shutdown')
        worker_coordinator = parallel_workers.init_workers(dummy_worker, shutdown_fun=shutdown_fun)
        worker_coordinator.start()
        self.assertTrue(worker_coordinator.stop())
        data = workspace.FetchBlob('data')
        self.assertEqual(data, b'shutdown', 'Got unexpected value ' + str(data))


