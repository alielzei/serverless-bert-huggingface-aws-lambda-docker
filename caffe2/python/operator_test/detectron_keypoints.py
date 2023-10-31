from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    import cv2
except ImportError:
    pass
import numpy as np
_NUM_KEYPOINTS = -1
_INFERENCE_MIN_SIZE = 0

def heatmaps_to_keypoints(maps, rois):
    """Extracts predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.detectron_keypoints.heatmaps_to_keypoints', 'heatmaps_to_keypoints(maps, rois)', {'np': np, '_NUM_KEYPOINTS': _NUM_KEYPOINTS, '_INFERENCE_MIN_SIZE': _INFERENCE_MIN_SIZE, 'cv2': cv2, 'scores_to_probs': scores_to_probs, 'maps': maps, 'rois': rois}, 1)

def scores_to_probs(scores):
    """Transforms CxHxW of scores to probabilities spatially."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.detectron_keypoints.scores_to_probs', 'scores_to_probs(scores)', {'np': np, 'scores': scores}, 1)

def approx_heatmap_keypoint(heatmaps_in, bboxes_in):
    """
Mask R-CNN uses bicubic upscaling before taking the maximum of the heat map
for keypoints. We are using bilinear upscaling, which means we can approximate
the maximum coordinate with the low dimension maximum coordinates. We would like
to avoid bicubic upscaling, because it is computationally expensive. Brown and
Lowe  (Invariant Features from Interest Point Groups, 2002) uses a method  for
fitting a 3D quadratic function to the local sample points to determine the
interpolated location of the maximum of scale space, and his experiments showed
that this provides a substantial improvement to matching and stability for
keypoint extraction. This approach uses the Taylor expansion (up to the
quadratic terms) of the scale-space function. It is equivalent with the Newton
method. This efficient method were used in many keypoint estimation algorithms
like SIFT, SURF etc...

The implementation of Newton methods with numerical analysis is straight forward
and super simple, though we need a linear solver.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.detectron_keypoints.approx_heatmap_keypoint', 'approx_heatmap_keypoint(heatmaps_in, bboxes_in)', {'np': np, 'scores_to_probs': scores_to_probs, 'heatmaps_in': heatmaps_in, 'bboxes_in': bboxes_in}, 1)

