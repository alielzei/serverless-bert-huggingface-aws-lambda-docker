from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import unittest
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python import core, utils
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

def boxes_area(boxes):
    """Compute the area of an array of boxes."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.collect_and_distribute_fpn_rpn_proposals_op_test.boxes_area', 'boxes_area(boxes)', {'np': np, 'boxes': boxes}, 1)

def map_rois_to_fpn_levels(rois, k_min, k_max, roi_canonical_scale, roi_canonical_level):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.collect_and_distribute_fpn_rpn_proposals_op_test.map_rois_to_fpn_levels', 'map_rois_to_fpn_levels(rois, k_min, k_max, roi_canonical_scale, roi_canonical_level)', {'np': np, 'boxes_area': boxes_area, 'rois': rois, 'k_min': k_min, 'k_max': k_max, 'roi_canonical_scale': roi_canonical_scale, 'roi_canonical_level': roi_canonical_level}, 1)

def collect(inputs, **args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.collect_and_distribute_fpn_rpn_proposals_op_test.collect', 'collect(inputs, **args)', {'np': np, 'inputs': inputs, 'args': args}, 1)

def distribute(rois, _, outputs, **args):
    """To understand the output blob order see return value of
    roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.operator_test.collect_and_distribute_fpn_rpn_proposals_op_test.distribute', 'distribute(rois, _, outputs, **args)', {'map_rois_to_fpn_levels': map_rois_to_fpn_levels, 'np': np, 'rois': rois, '_': _, 'outputs': outputs, 'args': args}, 0)

def collect_and_distribute_fpn_rpn_ref(*inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.collect_and_distribute_fpn_rpn_proposals_op_test.collect_and_distribute_fpn_rpn_ref', 'collect_and_distribute_fpn_rpn_ref(*inputs)', {'collect': collect, 'distribute': distribute, 'inputs': inputs}, 1)

def collect_rpn_ref(*inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.collect_and_distribute_fpn_rpn_proposals_op_test.collect_rpn_ref', 'collect_rpn_ref(*inputs)', {'collect': collect, 'inputs': inputs}, 1)

def distribute_fpn_ref(*inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.collect_and_distribute_fpn_rpn_proposals_op_test.distribute_fpn_ref', 'distribute_fpn_ref(*inputs)', {'distribute': distribute, 'inputs': inputs}, 1)


class TestCollectAndDistributeFpnRpnProposals(serial.SerializedTestCase):
    
    @staticmethod
    def _create_input(proposal_count, rpn_min_level, rpn_num_levels, roi_canonical_scale):
        np.random.seed(0)
        input_names = []
        inputs = []
        for lvl in range(rpn_num_levels):
            rpn_roi = roi_canonical_scale * np.random.rand(proposal_count, 5).astype(np.float32)
            for i in range(proposal_count):
                rpn_roi[i][3] += rpn_roi[i][1]
                rpn_roi[i][4] += rpn_roi[i][2]
            input_names.append('rpn_rois_fpn{}'.format(lvl + rpn_min_level))
            inputs.append(rpn_roi)
        for lvl in range(rpn_num_levels):
            rpn_roi_score = np.random.rand(proposal_count).astype(np.float32)
            input_names.append('rpn_roi_probs_fpn{}'.format(lvl + rpn_min_level))
            inputs.append(rpn_roi_score)
        return (input_names, inputs)
    
    @serial.given(proposal_count=st.integers(min_value=1000, max_value=8000), rpn_min_level=st.integers(min_value=1, max_value=4), rpn_num_levels=st.integers(min_value=1, max_value=6), roi_min_level=st.integers(min_value=1, max_value=4), roi_num_levels=st.integers(min_value=1, max_value=6), rpn_post_nms_topN=st.integers(min_value=1000, max_value=4000), roi_canonical_scale=st.integers(min_value=100, max_value=300), roi_canonical_level=st.integers(min_value=1, max_value=8), **hu.gcs_cpu_only)
    def test_collect_and_dist(self, proposal_count, rpn_min_level, rpn_num_levels, roi_min_level, roi_num_levels, rpn_post_nms_topN, roi_canonical_scale, roi_canonical_level, gc, dc):
        (input_names, inputs) = self._create_input(proposal_count, rpn_min_level, rpn_num_levels, roi_canonical_scale)
        output_names = ['rois']
        for lvl in range(roi_num_levels):
            output_names.append('rois_fpn{}'.format(lvl + roi_min_level))
        output_names.append('rois_idx_restore')
        op = core.CreateOperator('CollectAndDistributeFpnRpnProposals', input_names, output_names, arg=[utils.MakeArgument('roi_canonical_scale', roi_canonical_scale), utils.MakeArgument('roi_canonical_level', roi_canonical_level), utils.MakeArgument('roi_max_level', roi_min_level + roi_num_levels - 1), utils.MakeArgument('roi_min_level', roi_min_level), utils.MakeArgument('rpn_max_level', rpn_min_level + rpn_num_levels - 1), utils.MakeArgument('rpn_min_level', rpn_min_level), utils.MakeArgument('rpn_post_nms_topN', rpn_post_nms_topN)], device_option=gc)
        args = {'rpn_min_level': rpn_min_level, 'rpn_num_levels': rpn_num_levels, 'roi_min_level': roi_min_level, 'roi_num_levels': roi_num_levels, 'rpn_post_nms_topN': rpn_post_nms_topN, 'roi_canonical_scale': roi_canonical_scale, 'roi_canonical_level': roi_canonical_level}
        self.assertReferenceChecks(device_option=gc, op=op, inputs=inputs + [args], reference=collect_and_distribute_fpn_rpn_ref)
    
    @given(proposal_count=st.integers(min_value=1000, max_value=8000), rpn_min_level=st.integers(min_value=1, max_value=4), rpn_num_levels=st.integers(min_value=1, max_value=6), roi_min_level=st.integers(min_value=1, max_value=4), roi_num_levels=st.integers(min_value=1, max_value=6), rpn_post_nms_topN=st.integers(min_value=1000, max_value=4000), roi_canonical_scale=st.integers(min_value=100, max_value=300), roi_canonical_level=st.integers(min_value=1, max_value=8), **hu.gcs_cpu_only)
    def test_collect_and_dist_separately(self, proposal_count, rpn_min_level, rpn_num_levels, roi_min_level, roi_num_levels, rpn_post_nms_topN, roi_canonical_scale, roi_canonical_level, gc, dc):
        (input_names, inputs) = self._create_input(proposal_count, rpn_min_level, rpn_num_levels, roi_canonical_scale)
        collect_op = core.CreateOperator('CollectRpnProposals', input_names, ['rois'], arg=[utils.MakeArgument('rpn_max_level', rpn_min_level + rpn_num_levels - 1), utils.MakeArgument('rpn_min_level', rpn_min_level), utils.MakeArgument('rpn_post_nms_topN', rpn_post_nms_topN)], device_option=gc)
        collect_args = {'rpn_min_level': rpn_min_level, 'rpn_num_levels': rpn_num_levels, 'rpn_post_nms_topN': rpn_post_nms_topN}
        self.assertReferenceChecks(device_option=gc, op=collect_op, inputs=inputs + [collect_args], reference=collect_rpn_ref)
        rois = collect(inputs, **collect_args)
        output_names = []
        for lvl in range(roi_num_levels):
            output_names.append('rois_fpn{}'.format(lvl + roi_min_level))
        output_names.append('rois_idx_restore')
        distribute_op = core.CreateOperator('DistributeFpnProposals', ['rois'], output_names, arg=[utils.MakeArgument('roi_canonical_scale', roi_canonical_scale), utils.MakeArgument('roi_canonical_level', roi_canonical_level), utils.MakeArgument('roi_max_level', roi_min_level + roi_num_levels - 1), utils.MakeArgument('roi_min_level', roi_min_level)], device_option=gc)
        distribute_args = {'roi_min_level': roi_min_level, 'roi_num_levels': roi_num_levels, 'roi_canonical_scale': roi_canonical_scale, 'roi_canonical_level': roi_canonical_level}
        self.assertReferenceChecks(device_option=gc, op=distribute_op, inputs=[rois, distribute_args], reference=distribute_fpn_ref)

if __name__ == '__main__':
    unittest.main()

