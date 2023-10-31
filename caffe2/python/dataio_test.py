from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.dataio import CompositeReader, CompositeReaderBuilder, Reader, ReaderBuilder, ReaderWithDelay, ReaderWithLimit, ReaderWithTimeLimit
from caffe2.python.dataset import Dataset
from caffe2.python.db_file_reader import DBFileReader
from caffe2.python.pipeline import pipe
from caffe2.python.schema import Struct, NewRecord, FeedRecord
from caffe2.python.session import LocalSession
from caffe2.python.task import TaskGroup, final_output, WorkspaceType
from caffe2.python.test_util import TestCase
from caffe2.python.cached_reader import CachedReader
from caffe2.python import core, workspace, schema
from caffe2.python.net_builder import ops
import numpy as np
import numpy.testing as npt
import os
import shutil
import unittest
import tempfile
import time

def make_source_dataset(ws, size=100, offset=0, name=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.dataio_test.make_source_dataset', 'make_source_dataset(ws, size=100, offset=0, name=None)', {'core': core, 'Struct': Struct, 'np': np, 'NewRecord': NewRecord, 'Dataset': Dataset, 'FeedRecord': FeedRecord, 'ws': ws, 'size': size, 'offset': offset, 'name': name}, 1)

def make_destination_dataset(ws, schema, name=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.dataio_test.make_destination_dataset', 'make_destination_dataset(ws, schema, name=None)', {'core': core, 'Dataset': Dataset, 'ws': ws, 'schema': schema, 'name': name}, 1)


class TestReaderBuilder(ReaderBuilder):
    
    def __init__(self, name, size, offset):
        self._schema = schema.Struct(('label', schema.Scalar()))
        self._name = name
        self._size = size
        self._offset = offset
        self._src_ds = None
    
    def schema(self):
        return self._schema
    
    def setup(self, ws):
        self._src_ds = make_source_dataset(ws, offset=self._offset, size=self._size, name=self._name)
        return {}
    
    def new_reader(self, **kwargs):
        return self._src_ds



class TestCompositeReader(TestCase):
    
    @unittest.skipIf(os.environ.get('JENKINS_URL'), 'Flaky test on Jenkins')
    def test_composite_reader(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)
        num_srcs = 3
        names = ['src_{}'.format(i) for i in range(num_srcs)]
        size = 100
        offsets = [i * size for i in range(num_srcs)]
        src_dses = [make_source_dataset(ws, offset=offset, size=size, name=name) for (name, offset) in zip(names, offsets)]
        data = [ws.fetch_blob(str(src.field_blobs[0])) for src in src_dses]
        for (d, offset) in zip(data, offsets):
            npt.assert_array_equal(d, range(offset, offset + size))
        dst_ds_schema = schema.Struct(*[(name, src_ds.content().clone_schema()) for (name, src_ds) in zip(names, src_dses)])
        dst_ds = make_destination_dataset(ws, dst_ds_schema)
        with TaskGroup() as tg:
            reader = CompositeReader(names, [src_ds.reader() for src_ds in src_dses])
            pipe(reader, dst_ds.writer(), num_runtime_threads=3)
        session.run(tg)
        for i in range(num_srcs):
            written_data = sorted(ws.fetch_blob(str(dst_ds.content()[names[i]].label())))
            npt.assert_array_equal(data[i], written_data, 'i: {}'.format(i))
    
    @unittest.skipIf(os.environ.get('JENKINS_URL'), 'Flaky test on Jenkins')
    def test_composite_reader_builder(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)
        num_srcs = 3
        names = ['src_{}'.format(i) for i in range(num_srcs)]
        size = 100
        offsets = [i * size for i in range(num_srcs)]
        src_ds_builders = [TestReaderBuilder(offset=offset, size=size, name=name) for (name, offset) in zip(names, offsets)]
        dst_ds_schema = schema.Struct(*[(name, src_ds_builder.schema()) for (name, src_ds_builder) in zip(names, src_ds_builders)])
        dst_ds = make_destination_dataset(ws, dst_ds_schema)
        with TaskGroup() as tg:
            reader_builder = CompositeReaderBuilder(names, src_ds_builders)
            reader_builder.setup(ws=ws)
            pipe(reader_builder.new_reader(), dst_ds.writer(), num_runtime_threads=3)
        session.run(tg)
        for (name, offset) in zip(names, offsets):
            written_data = sorted(ws.fetch_blob(str(dst_ds.content()[name].label())))
            npt.assert_array_equal(range(offset, offset + size), written_data, 'name: {}'.format(name))



class TestReaderWithLimit(TestCase):
    
    def test_runtime_threads(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)
        src_ds = make_source_dataset(ws)
        totals = [None] * 3
        
        def proc(rec):
            with ops.task_init():
                counter1 = ops.CreateCounter([], ['global_counter'])
                counter2 = ops.CreateCounter([], ['global_counter2'])
                counter3 = ops.CreateCounter([], ['global_counter3'])
            with ops.task_instance_init():
                task_counter = ops.CreateCounter([], ['task_counter'])
            ops.CountUp(counter1)
            ops.CountUp(task_counter)
            with ops.task_instance_exit():
                with ops.loop(ops.RetrieveCount(task_counter)):
                    ops.CountUp(counter2)
                ops.CountUp(counter3)
            with ops.task_exit():
                totals[0] = final_output(ops.RetrieveCount(counter1))
                totals[1] = final_output(ops.RetrieveCount(counter2))
                totals[2] = final_output(ops.RetrieveCount(counter3))
            return rec
        with TaskGroup() as tg:
            pipe(src_ds.reader(), num_runtime_threads=8, processor=proc)
        session.run(tg)
        self.assertEqual(totals[0].fetch(), 100)
        self.assertEqual(totals[1].fetch(), 100)
        self.assertEqual(totals[2].fetch(), 8)
        with TaskGroup() as tg:
            q1 = pipe(src_ds.reader(), num_runtime_threads=2)
            q2 = pipe(ReaderWithLimit(q1.reader(), num_iter=25), num_runtime_threads=3)
            pipe(q2, processor=proc, num_runtime_threads=6)
        session.run(tg)
        self.assertEqual(totals[0].fetch(), 25)
        self.assertEqual(totals[1].fetch(), 25)
        self.assertEqual(totals[2].fetch(), 6)
    
    def _test_limit_reader_init_shared(self, size):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)
        src_ds = make_source_dataset(ws, size=size)
        dst_ds = make_destination_dataset(ws, src_ds.content().clone_schema())
        return (ws, session, src_ds, dst_ds)
    
    def _test_limit_reader_shared(self, reader_class, size, expected_read_len, expected_read_len_threshold, expected_finish, num_threads, read_delay, **limiter_args):
        (ws, session, src_ds, dst_ds) = self._test_limit_reader_init_shared(size)
        with TaskGroup(workspace_type=WorkspaceType.GLOBAL) as tg:
            if read_delay > 0:
                reader = reader_class(ReaderWithDelay(src_ds.reader(), read_delay), **limiter_args)
            else:
                reader = reader_class(src_ds.reader(), **limiter_args)
            pipe(reader, dst_ds.writer(), num_runtime_threads=num_threads)
        session.run(tg)
        read_len = len(sorted(ws.blobs[str(dst_ds.content().label())].fetch()))
        self.assertGreaterEqual(read_len, expected_read_len - expected_read_len_threshold)
        self.assertLessEqual(read_len, expected_read_len + expected_read_len_threshold)
        self.assertEqual(sorted(ws.blobs[str(dst_ds.content().label())].fetch()), list(range(read_len)))
        self.assertEqual(ws.blobs[str(reader.data_finished())].fetch(), expected_finish)
    
    def test_count_limit_reader_without_limit(self):
        self._test_limit_reader_shared(ReaderWithLimit, size=100, expected_read_len=100, expected_read_len_threshold=0, expected_finish=True, num_threads=8, read_delay=0, num_iter=None)
    
    def test_count_limit_reader_with_zero_limit(self):
        self._test_limit_reader_shared(ReaderWithLimit, size=100, expected_read_len=0, expected_read_len_threshold=0, expected_finish=False, num_threads=8, read_delay=0, num_iter=0)
    
    def test_count_limit_reader_with_low_limit(self):
        self._test_limit_reader_shared(ReaderWithLimit, size=100, expected_read_len=10, expected_read_len_threshold=0, expected_finish=False, num_threads=8, read_delay=0, num_iter=10)
    
    def test_count_limit_reader_with_high_limit(self):
        self._test_limit_reader_shared(ReaderWithLimit, size=100, expected_read_len=100, expected_read_len_threshold=0, expected_finish=True, num_threads=8, read_delay=0, num_iter=110)
    
    def test_time_limit_reader_without_limit(self):
        self._test_limit_reader_shared(ReaderWithTimeLimit, size=100, expected_read_len=100, expected_read_len_threshold=0, expected_finish=True, num_threads=8, read_delay=0.1, duration=0)
    
    def test_time_limit_reader_with_short_limit(self):
        size = 50
        num_threads = 4
        sleep_duration = 0.25
        duration = 1
        expected_read_len = int(round(num_threads * duration / sleep_duration))
        duration = duration - 0.25 * sleep_duration
        self._test_limit_reader_shared(ReaderWithTimeLimit, size=size, expected_read_len=expected_read_len / 2, expected_read_len_threshold=expected_read_len / 2, expected_finish=False, num_threads=num_threads, read_delay=sleep_duration, duration=duration)
    
    def test_time_limit_reader_with_long_limit(self):
        self._test_limit_reader_shared(ReaderWithTimeLimit, size=50, expected_read_len=50, expected_read_len_threshold=0, expected_finish=True, num_threads=4, read_delay=0.2, duration=10)



class TestDBFileReader(TestCase):
    
    def setUp(self):
        self.temp_paths = []
    
    def tearDown(self):
        for path in self.temp_paths:
            self._delete_path(path)
    
    @staticmethod
    def _delete_path(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    
    def _make_temp_path(self):
        with tempfile.NamedTemporaryFile() as f:
            temp_path = f.name
        self.temp_paths.append(temp_path)
        return temp_path
    
    @staticmethod
    def _build_source_reader(ws, size):
        src_ds = make_source_dataset(ws, size)
        return src_ds.reader()
    
    @staticmethod
    def _read_all_data(ws, reader, session):
        dst_ds = make_destination_dataset(ws, reader.schema().clone_schema())
        with TaskGroup() as tg:
            pipe(reader, dst_ds.writer(), num_runtime_threads=8)
        session.run(tg)
        return ws.blobs[str(dst_ds.content().label())].fetch()
    
    def test_cached_reader(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)
        db_path = self._make_temp_path()
        cached_reader1 = CachedReader(self._build_source_reader(ws, 100), db_path, loop_over=False)
        build_cache_step = cached_reader1.build_cache_step()
        session.run(build_cache_step)
        data = self._read_all_data(ws, cached_reader1, session)
        self.assertEqual(sorted(data), list(range(100)))
        cached_reader2 = CachedReader(self._build_source_reader(ws, 200), db_path)
        build_cache_step = cached_reader2.build_cache_step()
        session.run(build_cache_step)
        data = self._read_all_data(ws, cached_reader2, session)
        self.assertEqual(sorted(data), list(range(100)))
        self._delete_path(db_path)
        cached_reader3 = CachedReader(self._build_source_reader(ws, 300), db_path)
        build_cache_step = cached_reader3.build_cache_step()
        session.run(build_cache_step)
        data = self._read_all_data(ws, cached_reader3, session)
        self.assertEqual(sorted(data), list(range(300)))
        self._delete_path(db_path)
    
    def test_db_file_reader(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)
        db_path = self._make_temp_path()
        cached_reader = CachedReader(self._build_source_reader(ws, 100), db_path=db_path, db_type='LevelDB')
        build_cache_step = cached_reader.build_cache_step()
        session.run(build_cache_step)
        db_file_reader = DBFileReader(db_path=db_path, db_type='LevelDB')
        data = self._read_all_data(ws, db_file_reader, session)
        self.assertEqual(sorted(data), list(range(100)))
        self._delete_path(db_path)


