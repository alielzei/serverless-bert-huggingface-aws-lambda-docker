from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, dataio
from caffe2.python.task import TaskGroup
import logging
logger = logging.getLogger(__name__)


class _QueueReader(dataio.Reader):
    
    def __init__(self, wrapper, num_dequeue_records=1):
        assert wrapper.schema is not None, 'Queue needs a schema in order to be read from.'
        dataio.Reader.__init__(self, wrapper.schema())
        self._wrapper = wrapper
        self._num_dequeue_records = num_dequeue_records
    
    def setup_ex(self, init_net, exit_net):
        exit_net.CloseBlobsQueue([self._wrapper.queue()], 0)
    
    def read_ex(self, local_init_net, local_finish_net):
        self._wrapper._new_reader(local_init_net)
        dequeue_net = core.Net('dequeue')
        (fields, status_blob) = dequeue(dequeue_net, self._wrapper.queue(), len(self.schema().field_names()), field_names=self.schema().field_names(), num_records=self._num_dequeue_records)
        return ([dequeue_net], status_blob, fields)
    
    def read(self, net):
        (net, _, fields) = self.read_ex(net, None)
        return (net, fields)



class _QueueWriter(dataio.Writer):
    
    def __init__(self, wrapper):
        self._wrapper = wrapper
    
    def setup_ex(self, init_net, exit_net):
        exit_net.CloseBlobsQueue([self._wrapper.queue()], 0)
    
    def write_ex(self, fields, local_init_net, local_finish_net, status):
        self._wrapper._new_writer(self.schema(), local_init_net)
        enqueue_net = core.Net('enqueue')
        enqueue(enqueue_net, self._wrapper.queue(), fields, status)
        return [enqueue_net]



class QueueWrapper(dataio.Pipe):
    
    def __init__(self, handler, schema=None, num_dequeue_records=1):
        dataio.Pipe.__init__(self, schema, TaskGroup.LOCAL_SETUP)
        self._queue = handler
        self._num_dequeue_records = num_dequeue_records
    
    def reader(self):
        return _QueueReader(self, num_dequeue_records=self._num_dequeue_records)
    
    def writer(self):
        return _QueueWriter(self)
    
    def queue(self):
        return self._queue



class Queue(QueueWrapper):
    
    def __init__(self, capacity, schema=None, name='queue', num_dequeue_records=1):
        net = core.Net(name)
        queue_blob = net.AddExternalInput(net.NextName('handler'))
        QueueWrapper.__init__(self, queue_blob, schema, num_dequeue_records=num_dequeue_records)
        self.capacity = capacity
        self._setup_done = False
    
    def setup(self, global_init_net):
        assert self._schema, 'This queue does not have a schema.'
        self._setup_done = True
        global_init_net.CreateBlobsQueue([], [self._queue], capacity=self.capacity, num_blobs=len(self._schema.field_names()), field_names=self._schema.field_names())


def enqueue(net, queue, data_blobs, status=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.queue_util.enqueue', 'enqueue(net, queue, data_blobs, status=None)', {'logger': logger, 'net': net, 'queue': queue, 'data_blobs': data_blobs, 'status': status}, 1)

def dequeue(net, queue, num_blobs, status=None, field_names=None, num_records=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.queue_util.dequeue', 'dequeue(net, queue, num_blobs, status=None, field_names=None, num_records=1)', {'net': net, 'queue': queue, 'num_blobs': num_blobs, 'status': status, 'field_names': field_names, 'num_records': num_records}, 2)

def close_queue(step, *queues):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.queue_util.close_queue', 'close_queue(step, *queues)', {'core': core, 'step': step, 'queues': queues}, 1)

