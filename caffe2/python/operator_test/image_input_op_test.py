from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest
try:
    import cv2
    import lmdb
except ImportError:
    pass
from PIL import Image
import numpy as np
import shutil
import six
import sys
import tempfile
from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st
from caffe2.proto import caffe2_pb2
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import workspace, core

def verify_apply_bounding_box(img, box):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.image_input_op_test.verify_apply_bounding_box', 'verify_apply_bounding_box(img, box)', {'np': np, 'skimage': skimage, 'img': img, 'box': box}, 1)

def verify_rescale(img, minsize):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.image_input_op_test.verify_rescale', 'verify_rescale(img, minsize)', {'cv2': cv2, 'np': np, 'img': img, 'minsize': minsize}, 1)

def verify_crop(img, crop):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.image_input_op_test.verify_crop', 'verify_crop(img, crop)', {'skimage': skimage, 'img': img, 'crop': crop}, 1)

def verify_color_normalize(img, means, stds):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.image_input_op_test.verify_color_normalize', 'verify_color_normalize(img, means, stds)', {'img': img, 'means': means, 'stds': stds}, 1)

def caffe2_img(img):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.image_input_op_test.caffe2_img', 'caffe2_img(img)', {'np': np, 'img': img}, 1)

def create_test(output_dir, width, height, default_bound, minsize, crop, means, stds, count, label_type, num_labels, output1=None, output2_size=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.image_input_op_test.create_test', 'create_test(output_dir, width, height, default_bound, minsize, crop, means, stds, count, label_type, num_labels, output1=None, output2_size=None)', {'lmdb': lmdb, 'np': np, 'Image': Image, 'six': six, 'verify_apply_bounding_box': verify_apply_bounding_box, 'verify_rescale': verify_rescale, 'verify_crop': verify_crop, 'verify_color_normalize': verify_color_normalize, 'caffe2_pb2': caffe2_pb2, 'caffe2_img': caffe2_img, 'output_dir': output_dir, 'width': width, 'height': height, 'default_bound': default_bound, 'minsize': minsize, 'crop': crop, 'means': means, 'stds': stds, 'count': count, 'label_type': label_type, 'num_labels': num_labels, 'output1': output1, 'output2_size': output2_size}, 1)

def run_test(size_tuple, means, stds, label_type, num_labels, is_test, scale_jitter_type, color_jitter, color_lighting, dc, validator, output1=None, output2_size=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.operator_test.image_input_op_test.run_test', 'run_test(size_tuple, means, stds, label_type, num_labels, is_test, scale_jitter_type, color_jitter, color_lighting, dc, validator, output1=None, output2_size=None)', {'tempfile': tempfile, 'create_test': create_test, 'hu': hu, 'core': core, 'workspace': workspace, 'shutil': shutil, 'size_tuple': size_tuple, 'means': means, 'stds': stds, 'label_type': label_type, 'num_labels': num_labels, 'is_test': is_test, 'scale_jitter_type': scale_jitter_type, 'color_jitter': color_jitter, 'color_lighting': color_lighting, 'dc': dc, 'validator': validator, 'output1': output1, 'output2_size': output2_size}, 0)


@unittest.skipIf('cv2' not in sys.modules, 'python-opencv is not installed')
@unittest.skipIf('lmdb' not in sys.modules, 'python-lmdb is not installed')
class TestImport(hu.HypothesisTestCase):
    
    def validate_image_and_label(self, expected_images, device_option, count_images, label_type, is_test, scale_jitter_type, color_jitter, color_lighting):
        l = workspace.FetchBlob('label')
        result = workspace.FetchBlob('data').astype(np.int32)
        if device_option.device_type != 1:
            expected = [img.swapaxes(0, 1).swapaxes(1, 2) for (img, _, _, _) in expected_images]
        else:
            expected = [img for (img, _, _, _) in expected_images]
        for i in range(count_images):
            if label_type == 0:
                self.assertEqual(l[i], expected_images[i][1])
            else:
                self.assertEqual((l[i] - expected_images[i][1] > 0).sum(), 0)
            if is_test == 0:
                for (s1, s2) in zip(expected[i].shape, result[i].shape):
                    self.assertEqual(s1, s2)
            else:
                self.assertEqual((expected[i] - result[i] > 1).sum(), 0)
    
    @given(size_tuple=st.tuples(st.integers(min_value=8, max_value=4096), st.integers(min_value=8, max_value=4096)).flatmap(lambda t: st.tuples(st.just(t[0]), st.just(t[1]), st.just(min(t[0] - 6, t[1] - 4)), st.integers(min_value=1, max_value=min(t[0] - 6, t[1] - 4)))), means=st.tuples(st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255)), stds=st.tuples(st.floats(min_value=1, max_value=10), st.floats(min_value=1, max_value=10), st.floats(min_value=1, max_value=10)), label_type=st.integers(0, 3), num_labels=st.integers(min_value=8, max_value=4096), is_test=st.integers(min_value=0, max_value=1), scale_jitter_type=st.integers(min_value=0, max_value=1), color_jitter=st.integers(min_value=0, max_value=1), color_lighting=st.integers(min_value=0, max_value=1), **hu.gcs)
    @settings(verbosity=Verbosity.verbose)
    def test_imageinput(self, size_tuple, means, stds, label_type, num_labels, is_test, scale_jitter_type, color_jitter, color_lighting, gc, dc):
        
        def validator(expected_images, device_option, count_images):
            self.validate_image_and_label(expected_images, device_option, count_images, label_type, is_test, scale_jitter_type, color_jitter, color_lighting)
        run_test(size_tuple, means, stds, label_type, num_labels, is_test, scale_jitter_type, color_jitter, color_lighting, dc, validator)
    
    @given(size_tuple=st.tuples(st.integers(min_value=8, max_value=4096), st.integers(min_value=8, max_value=4096)).flatmap(lambda t: st.tuples(st.just(t[0]), st.just(t[1]), st.just(min(t[0] - 6, t[1] - 4)), st.integers(min_value=1, max_value=min(t[0] - 6, t[1] - 4)))), means=st.tuples(st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255)), stds=st.tuples(st.floats(min_value=1, max_value=10), st.floats(min_value=1, max_value=10), st.floats(min_value=1, max_value=10)), label_type=st.integers(0, 3), num_labels=st.integers(min_value=8, max_value=4096), is_test=st.integers(min_value=0, max_value=1), scale_jitter_type=st.integers(min_value=0, max_value=1), color_jitter=st.integers(min_value=0, max_value=1), color_lighting=st.integers(min_value=0, max_value=1), output1=st.floats(min_value=1, max_value=10), output2_size=st.integers(min_value=2, max_value=10), **hu.gcs)
    @settings(verbosity=Verbosity.verbose)
    def test_imageinput_with_additional_outputs(self, size_tuple, means, stds, label_type, num_labels, is_test, scale_jitter_type, color_jitter, color_lighting, output1, output2_size, gc, dc):
        
        def validator(expected_images, device_option, count_images):
            self.validate_image_and_label(expected_images, device_option, count_images, label_type, is_test, scale_jitter_type, color_jitter, color_lighting)
            output1_result = workspace.FetchBlob('output1')
            output2_result = workspace.FetchBlob('output2')
            for i in range(count_images):
                self.assertEqual(output1_result[i], expected_images[i][2])
                self.assertEqual((output2_result[i] - expected_images[i][3] > 0).sum(), 0)
        run_test(size_tuple, means, stds, label_type, num_labels, is_test, scale_jitter_type, color_jitter, color_lighting, dc, validator, output1, output2_size)

if __name__ == '__main__':
    import unittest
    unittest.main()

