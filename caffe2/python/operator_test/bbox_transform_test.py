from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.bbox_transform_test.bbox_transform', 'bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0))', {'np': np, 'boxes': boxes, 'deltas': deltas, 'weights': weights}, 1)

def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.bbox_transform_test.clip_tiled_boxes', 'clip_tiled_boxes(boxes, im_shape)', {'np': np, 'boxes': boxes, 'im_shape': im_shape}, 1)

def generate_rois(roi_counts, im_dims):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.bbox_transform_test.generate_rois', 'generate_rois(roi_counts, im_dims)', {'np': np, 'roi_counts': roi_counts, 'im_dims': im_dims}, 1)

def bbox_transform_rotated(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0), angle_bound_on=True, angle_bound_lo=-90, angle_bound_hi=90):
    """
    Similar to bbox_transform but for rotated boxes with angle info.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.bbox_transform_test.bbox_transform_rotated', 'bbox_transform_rotated(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0), angle_bound_on=True, angle_bound_lo=-90, angle_bound_hi=90)', {'np': np, 'boxes': boxes, 'deltas': deltas, 'weights': weights, 'angle_bound_on': angle_bound_on, 'angle_bound_lo': angle_bound_lo, 'angle_bound_hi': angle_bound_hi}, 1)

def clip_tiled_boxes_rotated(boxes, im_shape, angle_thresh=1.0):
    """
    Similar to clip_tiled_boxes but for rotated boxes with angle info.
    Only clips almost horizontal boxes within angle_thresh. The rest are
    left unchanged.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.bbox_transform_test.clip_tiled_boxes_rotated', 'clip_tiled_boxes_rotated(boxes, im_shape, angle_thresh=1.0)', {'np': np, 'boxes': boxes, 'im_shape': im_shape, 'angle_thresh': angle_thresh}, 1)

def generate_rois_rotated(roi_counts, im_dims):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.bbox_transform_test.generate_rois_rotated', 'generate_rois_rotated(roi_counts, im_dims)', {'generate_rois': generate_rois, 'np': np, 'roi_counts': roi_counts, 'im_dims': im_dims}, 1)


class TestBBoxTransformOp(serial.SerializedTestCase):
    
    @serial.given(num_rois=st.integers(1, 10), num_classes=st.integers(1, 10), im_dim=st.integers(100, 600), skip_batch_id=st.booleans(), rotated=st.booleans(), angle_bound_on=st.booleans(), clip_angle_thresh=st.sampled_from([-1.0, 1.0]), **hu.gcs_cpu_only)
    def test_bbox_transform(self, num_rois, num_classes, im_dim, skip_batch_id, rotated, angle_bound_on, clip_angle_thresh, gc, dc):
        """
        Test with all rois belonging to a single image per run.
        """
        rois = (generate_rois_rotated([num_rois], [im_dim]) if rotated else generate_rois([num_rois], [im_dim]))
        box_dim = (5 if rotated else 4)
        if skip_batch_id:
            rois = rois[:, 1:]
        deltas = np.random.randn(num_rois, box_dim * num_classes).astype(np.float32)
        im_info = np.array([im_dim, im_dim, 1.0]).astype(np.float32).reshape(1, 3)
        
        def bbox_transform_ref(rois, deltas, im_info):
            boxes = (rois if rois.shape[1] == box_dim else rois[:, 1:])
            im_shape = im_info[0, 0:2]
            if rotated:
                box_out = bbox_transform_rotated(boxes, deltas, angle_bound_on=angle_bound_on)
                box_out = clip_tiled_boxes_rotated(box_out, im_shape, angle_thresh=clip_angle_thresh)
            else:
                box_out = bbox_transform(boxes, deltas)
                box_out = clip_tiled_boxes(box_out, im_shape)
            return [box_out]
        op = core.CreateOperator('BBoxTransform', ['rois', 'deltas', 'im_info'], ['box_out'], apply_scale=False, correct_transform_coords=True, rotated=rotated, angle_bound_on=angle_bound_on, clip_angle_thresh=clip_angle_thresh)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[rois, deltas, im_info], reference=bbox_transform_ref)
    
    @given(roi_counts=st.lists(st.integers(0, 5), min_size=1, max_size=10), num_classes=st.integers(1, 10), rotated=st.booleans(), angle_bound_on=st.booleans(), clip_angle_thresh=st.sampled_from([-1.0, 1.0]), **hu.gcs_cpu_only)
    def test_bbox_transform_batch(self, roi_counts, num_classes, rotated, angle_bound_on, clip_angle_thresh, gc, dc):
        """
        Test with rois for multiple images in a batch
        """
        batch_size = len(roi_counts)
        total_rois = sum(roi_counts)
        im_dims = np.random.randint(100, 600, batch_size)
        rois = (generate_rois_rotated(roi_counts, im_dims) if rotated else generate_rois(roi_counts, im_dims))
        box_dim = (5 if rotated else 4)
        deltas = np.random.randn(total_rois, box_dim * num_classes).astype(np.float32)
        im_info = np.zeros((batch_size, 3)).astype(np.float32)
        im_info[:, 0] = im_dims
        im_info[:, 1] = im_dims
        im_info[:, 2] = 1.0
        
        def bbox_transform_ref(rois, deltas, im_info):
            box_out = []
            offset = 0
            for (i, num_rois) in enumerate(roi_counts):
                if num_rois == 0:
                    continue
                cur_boxes = rois[offset:offset + num_rois, 1:]
                cur_deltas = deltas[offset:offset + num_rois]
                im_shape = im_info[i, 0:2]
                if rotated:
                    cur_box_out = bbox_transform_rotated(cur_boxes, cur_deltas, angle_bound_on=angle_bound_on)
                    cur_box_out = clip_tiled_boxes_rotated(cur_box_out, im_shape, angle_thresh=clip_angle_thresh)
                else:
                    cur_box_out = bbox_transform(cur_boxes, cur_deltas)
                    cur_box_out = clip_tiled_boxes(cur_box_out, im_shape)
                box_out.append(cur_box_out)
                offset += num_rois
            if len(box_out) > 0:
                box_out = np.vstack(box_out)
            else:
                box_out = np.empty(deltas.shape).astype(np.float32)
            return [box_out, roi_counts]
        op = core.CreateOperator('BBoxTransform', ['rois', 'deltas', 'im_info'], ['box_out', 'roi_batch_splits'], apply_scale=False, correct_transform_coords=True, rotated=rotated, angle_bound_on=angle_bound_on, clip_angle_thresh=clip_angle_thresh)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[rois, deltas, im_info], reference=bbox_transform_ref)


