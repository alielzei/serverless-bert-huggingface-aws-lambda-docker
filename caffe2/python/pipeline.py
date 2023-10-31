from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, queue_util
from caffe2.python.dataio import Reader, Writer
from caffe2.python.net_builder import NetBuilder, ops
from caffe2.python.schema import as_record, Field
from caffe2.python.task import Node, Task, TaskGroup


class Output(object):
    """
    Represents the result of a processor function. A processor can either
    return an Output, or it can return a record, in which case an Output will be
    created for it afterwards.
    """
    
    def __init__(self, nets=None, record=None, should_stop=None):
        builder_children = NetBuilder.current().get()
        assert (nets is None or len(builder_children) == 0), 'Cannot both use `ops` syntax and return a list of nets.'
        if nets is None:
            nets = builder_children
        if isinstance(nets, core.Net):
            nets = [nets]
        self.nets = ([] if nets is None else list(nets))
        self.record = (None if record is None else as_record(record))
        self.should_stop = should_stop

DEFAULT_QUEUE_CAPACITY = 100

def _init_output(output, capacity, global_init_net, global_exit_net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline._init_output', '_init_output(output, capacity, global_init_net, global_exit_net)', {'queue_util': queue_util, 'DEFAULT_QUEUE_CAPACITY': DEFAULT_QUEUE_CAPACITY, 'Writer': Writer, 'output': output, 'capacity': capacity, 'global_init_net': global_init_net, 'global_exit_net': global_exit_net}, 2)

def make_processor(processor, reader=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline.make_processor', 'make_processor(processor, reader=None)', {'core': core, 'NetProcessor': NetProcessor, 'processor': processor, 'reader': reader}, 1)

def normalize_processor_output(output):
    """
    Allow for processors to return results in several formats.
    TODO(azzolini): simplify once all processors use NetBuilder API.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline.normalize_processor_output', 'normalize_processor_output(output)', {'Output': Output, 'Field': Field, 'core': core, 'output': output}, 1)

def pipe(input, output=None, num_threads=1, processor=None, name=None, capacity=None, group=None, num_runtime_threads=1):
    """
    Given a Reader, Queue or DataStream in `input`, and optionally, a Writer,
    Queue or DataStream in `output`, creates a Task that, when run, will
    pipe the input into the output, using multiple parallel threads.
    Additionally, if a processor is given, it will be called between reading
    and writing steps, allowing it to transform the record.

    Args:
        input:       either a Reader, Queue or DataStream that will be read
                     until a stop is signaled either by the reader or the
                     writer.
        output:      either a Writer, a Queue or a DataStream that will be
                     written to as long as neither reader nor writer signal
                     a stop condition. If output is not provided or is None,
                     a Queue is created with given `capacity` and written to.
        num_threads: number of concurrent threads used for processing and
                     piping. If set to 0, no Task is created, and a
                     reader is returned instead -- the reader returned will
                     read from the reader passed in and process it.
                     ** DEPRECATED **. Use `num_runtime_threads` instead.
                     This option will be removed once all readers/processors
                     support `num_runtime_threads`.
        processor:   (optional) function that takes an input record and
                     optionally returns a record; this will be called
                     between read and write steps. If the processor does
                     not return a record, a writer will not be instantiated.
                     Processor can also be a core.Net with input and output
                     records properly set. In that case, a NetProcessor is
                     instantiated, cloning the net for each of the threads.
        name:        (optional) name of the task to be created.
        capacity:    when output is not passed, a queue of given `capacity`
                     is created and written to.
        group:       (optional) explicitly add the created Task to this
                     TaskGroup, instead of using the currently active one.
        num_runtime_threads: Similar to `num_threads`, but instead of expanding
                     the tasks with a `for` loop in python, does that at
                     runtime. This is preferable to `num_threads`, but some
                     processors/readers still require to be called multiple
                     times in python.

    Returns:
        Output Queue, DataStream, Reader, or None, depending on the parameters
        passed.
    """
    (result, _) = _pipe_step(input, output, num_threads, processor, name, capacity, group, num_runtime_threads)
    return result

def pipe_and_output(input, output=None, num_threads=1, processor=None, name=None, capacity=None, group=None, num_runtime_threads=1, final_outputs=None):
    """
    Similar to `pipe`, with the additional ability for the pipe Task to
    return output values to the `Session` once done.

    Returns:
        Tuple (out_queue, *task_outputs)
            out_queue:    same as return value of `pipe`.
            task_outputs: TaskOutput object, fetchable from the client after
                          session.run() returns.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline.pipe_and_output', 'pipe_and_output(input, output=None, num_threads=1, processor=None, name=None, capacity=None, group=None, num_runtime_threads=1, final_outputs=None)', {'_pipe_step': _pipe_step, 'input': input, 'output': output, 'num_threads': num_threads, 'processor': processor, 'name': name, 'capacity': capacity, 'group': group, 'num_runtime_threads': num_runtime_threads, 'final_outputs': final_outputs}, 2)

def processor_name(processor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline.processor_name', 'processor_name(processor)', {'processor': processor}, 1)

def _runtime_threads_task(name, group, final_outputs, reader, num_threads, output, capacity):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline._runtime_threads_task', '_runtime_threads_task(name, group, final_outputs, reader, num_threads, output, capacity)', {'Node': Node, 'processor_name': processor_name, 'Task': Task, 'core': core, '_init_output': _init_output, 'ops': ops, 'name': name, 'group': group, 'final_outputs': final_outputs, 'reader': reader, 'num_threads': num_threads, 'output': output, 'capacity': capacity}, 2)

def _static_threads_task(name, group, final_outputs, reader, num_threads, output, capacity):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline._static_threads_task', '_static_threads_task(name, group, final_outputs, reader, num_threads, output, capacity)', {'Node': Node, 'processor_name': processor_name, 'Task': Task, 'core': core, 'NetBuilder': NetBuilder, '_init_output': _init_output, 'ops': ops, 'name': name, 'group': group, 'final_outputs': final_outputs, 'reader': reader, 'num_threads': num_threads, 'output': output, 'capacity': capacity}, 2)

def _pipe_step(input, output=None, num_threads=1, processor=None, name=None, capacity=None, group=None, num_runtime_threads=None, final_outputs=None):
    """
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.pipeline._pipe_step', '_pipe_step(input, output=None, num_threads=1, processor=None, name=None, capacity=None, group=None, num_runtime_threads=None, final_outputs=None)', {'Reader': Reader, 'ProcessingReader': ProcessingReader, 'processor_name': processor_name, '_static_threads_task': _static_threads_task, '_runtime_threads_task': _runtime_threads_task, 'input': input, 'output': output, 'num_threads': num_threads, 'processor': processor, 'name': name, 'capacity': capacity, 'group': group, 'num_runtime_threads': num_runtime_threads, 'final_outputs': final_outputs}, 2)


class ProcessingReader(Reader):
    """
    Reader that reads from an upstream reader, calls the processor, and returns
    the processed record.
    """
    
    def __init__(self, reader, processor):
        Reader.__init__(self)
        self.reader = reader
        self.processor = make_processor(processor, reader)
    
    def schema(self):
        return self.processor.schema()
    
    def setup_ex(self, init_net, finish_net):
        self.reader.setup_ex(init_net, finish_net)
    
    def read_ex(self, init_net, exit_net):
        (read_nets, status, rec) = self.reader.read_record_ex(init_net, exit_net)
        with NetBuilder() as nb:
            result = normalize_processor_output(self.processor(rec))
        read_nets += result.nets
        if (result.should_stop or nb._stop_blob):
            stop_net = core.Net('stop_net')
            if result.should_stop:
                stop_net.Or([status, result.should_stop], [status])
            if nb._stop_blob:
                stop_net.Or([status, nb._stop_blob], [status])
            read_nets.append(stop_net)
        if hasattr(self.processor, 'setup'):
            init_net.add_attribute(TaskGroup.LOCAL_SETUP, self.processor)
        self._set_schema(result.record)
        fields = (result.record.field_blobs() if result.record else None)
        return (read_nets, status, fields)



class NetProcessor(object):
    """
    Processor that clones a core.Net each time it's called, executing
    the cloned net as the processor. It requires the Net to have input
    and (optionally) output records set, with net.set_input_record() and
    net.set_output_record().
    """
    
    def __init__(self, net, stop_signal=None, thread_init_nets=None, name=None):
        assert isinstance(net, core.Net)
        assert (stop_signal is None or isinstance(stop_signal, core.BlobReference))
        self.name = (name or str(net))
        self.thread_init_nets = (thread_init_nets or [])
        self.net = net
        self._stop_signal = stop_signal
        self._blob_maps = []
        self._frozen = False
        self._cloned_init_nets = []
    
    def schema(self):
        return self.net.output_record()
    
    def setup(self, init_net):
        self._frozen = True
        cloned_init_nets = self._cloned_init_nets
        self._cloned_init_nets = []
        return cloned_init_nets
    
    def __call__(self, rec):
        assert not self._frozen
        prefix = NetBuilder.current().name + '/'
        blob_remap = {}
        for net in self.thread_init_nets:
            (new_net, _) = core.clone_and_bind_net(net, str(net) + prefix, prefix, blob_remap)
            self._cloned_init_nets.append(new_net)
        (new_net, remappings) = core.clone_and_bind_net(self.net, str(self.net) + prefix, prefix, blob_remap, rec)
        if self._stop_signal is None:
            stop_signal = None
        elif str(self._stop_signal) in remappings:
            stop_signal = core.BlobReference(remappings[str(self._stop_signal)], net=new_net)
        else:
            stop_signal = self._stop_signal
        self._blob_maps.append(remappings)
        return Output([new_net], new_net.output_record(), stop_signal)
    
    def blob_maps(self):
        self._frozen = True
        return self._blob_maps


