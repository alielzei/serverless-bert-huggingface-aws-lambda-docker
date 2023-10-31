from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st
import unittest
import numpy as np

def get_op(input_len, output_len, args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.box_with_nms_limit_op_test.get_op', 'get_op(input_len, output_len, args)', {'core': core, 'input_len': input_len, 'output_len': output_len, 'args': args}, 1)
HU_CONFIG = {'gc': hu.gcs_cpu_only['gc']}

def gen_boxes(count, center):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.box_with_nms_limit_op_test.gen_boxes', 'gen_boxes(count, center)', {'np': np, 'count': count, 'center': center}, 1)

def gen_multiple_boxes(centers, scores, count, num_classes):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.box_with_nms_limit_op_test.gen_multiple_boxes', 'gen_multiple_boxes(centers, scores, count, num_classes)', {'gen_boxes': gen_boxes, 'np': np, 'centers': centers, 'scores': scores, 'count': count, 'num_classes': num_classes}, 2)


class TestBoxWithNMSLimitOp(serial.SerializedTestCase):
    
    @serial.given(**HU_CONFIG)
    def test_simple(self, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.9, 0.8, 0.6]
        (boxes, scores) = gen_multiple_boxes(in_centers, in_scores, 10, 2)
        (gt_boxes, gt_scores) = gen_multiple_boxes(in_centers, in_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)
        op = get_op(2, 3, {'score_thresh': 0.5, 'nms': 0.9})
        
        def ref(*args, **kwargs):
            return (gt_scores.flatten(), gt_boxes, gt_classes)
        self.assertReferenceChecks(gc, op, [scores, boxes], ref)
    
    @given(**HU_CONFIG)
    def test_score_thresh(self, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.85, 0.6]
        (boxes, scores) = gen_multiple_boxes(in_centers, in_scores, 10, 2)
        gt_centers = [(20, 20)]
        gt_scores = [0.85]
        (gt_boxes, gt_scores) = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)
        op = get_op(2, 3, {'score_thresh': 0.8, 'nms': 0.9})
        
        def ref(*args, **kwargs):
            return (gt_scores.flatten(), gt_boxes, gt_classes)
        self.assertReferenceChecks(gc, op, [scores, boxes], ref)
    
    @given(det_per_im=st.integers(1, 3), **HU_CONFIG)
    def test_detections_per_im(self, det_per_im, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.85, 0.6]
        (boxes, scores) = gen_multiple_boxes(in_centers, in_scores, 10, 2)
        gt_centers = [(20, 20), (0, 0), (50, 50)][:det_per_im]
        gt_scores = [0.85, 0.7, 0.6][:det_per_im]
        (gt_boxes, gt_scores) = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)
        op = get_op(2, 3, {'score_thresh': 0.5, 'nms': 0.9, 'detections_per_im': det_per_im})
        
        def ref(*args, **kwargs):
            return (gt_scores.flatten(), gt_boxes, gt_classes)
        self.assertReferenceChecks(gc, op, [scores, boxes], ref)
    
    @given(num_classes=st.integers(2, 10), det_per_im=st.integers(1, 4), cls_agnostic_bbox_reg=st.booleans(), input_boxes_include_bg_cls=st.booleans(), output_classes_include_bg_cls=st.booleans(), **HU_CONFIG)
    def test_multiclass(self, num_classes, det_per_im, cls_agnostic_bbox_reg, input_boxes_include_bg_cls, output_classes_include_bg_cls, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.85, 0.6]
        (boxes, scores) = gen_multiple_boxes(in_centers, in_scores, 10, num_classes)
        if not input_boxes_include_bg_cls:
            boxes = boxes[:, 4:]
        if cls_agnostic_bbox_reg:
            boxes = boxes[:, :4]
        scores_bg_class_id = (0 if input_boxes_include_bg_cls else -1)
        scores[:, scores_bg_class_id] = np.random.rand(scores.shape[0]).astype(np.float32)
        gt_centers = [(20, 20), (0, 0), (50, 50)][:det_per_im]
        gt_scores = [0.85, 0.7, 0.6][:det_per_im]
        (gt_boxes, gt_scores) = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        gt_classes = np.tile(np.array(range(1, num_classes), dtype=np.float32), (gt_boxes.shape[0], 1)).T.flatten()
        if not output_classes_include_bg_cls:
            gt_classes -= 1
        gt_boxes = np.tile(gt_boxes, (num_classes - 1, 1))
        gt_scores = np.tile(gt_scores, (num_classes - 1, 1)).flatten()
        op = get_op(2, 3, {'score_thresh': 0.5, 'nms': 0.9, 'detections_per_im': (num_classes - 1) * det_per_im, 'cls_agnostic_bbox_reg': cls_agnostic_bbox_reg, 'input_boxes_include_bg_cls': input_boxes_include_bg_cls, 'output_classes_include_bg_cls': output_classes_include_bg_cls})
        
        def ref(*args, **kwargs):
            return (gt_scores, gt_boxes, gt_classes)
        self.assertReferenceChecks(gc, op, [scores, boxes], ref)
    
    @given(det_per_im=st.integers(1, 3), **HU_CONFIG)
    def test_detections_per_im_same_thresh(self, det_per_im, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.7, 0.7, 0.7]
        (boxes, scores) = gen_multiple_boxes(in_centers, in_scores, 10, 2)
        gt_centers = [(20, 20), (0, 0), (50, 50)][:det_per_im]
        gt_scores = [0.7, 0.7, 0.7][:det_per_im]
        (gt_boxes, gt_scores) = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        gt_classes = np.ones(gt_boxes.shape[0], dtype=np.float32)
        op = get_op(2, 3, {'score_thresh': 0.5, 'nms': 0.9, 'detections_per_im': det_per_im})
        
        def verify(inputs, outputs):
            np.testing.assert_allclose(outputs[0], gt_scores.flatten(), atol=0.0001, rtol=0.0001)
            np.testing.assert_allclose(outputs[2], gt_classes, atol=0.0001, rtol=0.0001)
            self.assertEqual(outputs[1].shape, gt_boxes.shape)
        self.assertValidationChecks(gc, op, [scores, boxes], verify, as_kwargs=False)
    
    @given(num_classes=st.integers(2, 10), **HU_CONFIG)
    def test_detections_per_im_same_thresh_multiclass(self, num_classes, gc):
        in_centers = [(0, 0), (20, 20), (50, 50)]
        in_scores = [0.6, 0.7, 0.7]
        (boxes, scores) = gen_multiple_boxes(in_centers, in_scores, 10, num_classes)
        det_per_im = 1
        gt_centers = [(20, 20), (50, 50)]
        gt_scores = [0.7, 0.7]
        (gt_boxes, gt_scores) = gen_multiple_boxes(gt_centers, gt_scores, 1, 1)
        op = get_op(2, 3, {'score_thresh': 0.5, 'nms': 0.9, 'detections_per_im': det_per_im})
        
        def verify(inputs, outputs):
            self.assertEqual(outputs[0].shape, (1, ))
            self.assertEqual(outputs[0][0], gt_scores[0])
            self.assertTrue((np.allclose(outputs[1], gt_boxes[0, :], atol=0.0001, rtol=0.0001) or np.allclose(outputs[1], gt_boxes[1, :], atol=0.0001, rtol=0.0001)))
            self.assertNotEqual(outputs[2][0], 0)
        self.assertValidationChecks(gc, op, [scores, boxes], verify, as_kwargs=False)

if __name__ == '__main__':
    unittest.main()

