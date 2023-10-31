from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import numpy as np
import os
from six.moves import range
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.summary_pb2 import HistogramProto
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.plugins.custom_scalar import layout_pb2
from ._convert_np import make_np
from ._utils import _prepare_video, convert_to_HWC

def _calc_scale_factor(tensor):
    converted = (tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor)
    return (1 if converted.dtype == np.uint8 else 255)

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, color='black', color_text='black', thickness=2):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary._draw_single_box', "_draw_single_box(image, xmin, ymin, xmax, ymax, display_str, color='black', color_text='black', thickness=2)", {'np': np, 'image': image, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'display_str': display_str, 'color': color, 'color_text': color_text, 'thickness': thickness}, 1)

def hparams(hparam_dict=None, metric_dict=None):
    """Outputs three `Summary` protocol buffers needed by hparams plugin.
    `Experiment` keeps the metadata of an experiment, such as the name of the
      hyperparameters and the name of the metrics.
    `SessionStartInfo` keeps key-value pairs of the hyperparameters
    `SessionEndInfo` describes status of the experiment e.g. STATUS_SUCCESS

    Args:
      hparam_dict: A dictionary that contains names of the hyperparameters
        and their values.
      metric_dict: A dictionary that contains names of the metrics
        and their values.

    Returns:
      The `Summary` protobufs for Experiment, SessionStartInfo and
        SessionEndInfo
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.hparams', 'hparams(hparam_dict=None, metric_dict=None)', {'logging': logging, 'SummaryMetadata': SummaryMetadata, 'Summary': Summary, 'make_np': make_np, 'hparam_dict': hparam_dict, 'metric_dict': metric_dict}, 3)

def scalar(name, scalar, collections=None):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    Raises:
      ValueError: If tensor has the wrong shape or type.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.scalar', 'scalar(name, scalar, collections=None)', {'make_np': make_np, 'Summary': Summary, 'name': name, 'scalar': scalar, 'collections': collections}, 1)

def histogram_raw(name, min, max, num, sum, sum_squares, bucket_limits, bucket_counts):
    """Outputs a `Summary` protocol buffer with a histogram.
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      min: A float or int min value
      max: A float or int max value
      num: Int number of values
      sum: Float or int sum of all values
      sum_squares: Float or int sum of squares for all values
      bucket_limits: A numeric `Tensor` with upper value per bucket
      bucket_counts: A numeric `Tensor` with number of values per bucket
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    hist = HistogramProto(min=min, max=max, num=num, sum=sum, sum_squares=sum_squares, bucket_limit=bucket_limits, bucket=bucket_counts)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])

def histogram(name, values, bins, max_bins=None):
    """Outputs a `Summary` protocol buffer with a histogram.
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    This op reports an `InvalidArgument` error if any value is not finite.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.histogram', 'histogram(name, values, bins, max_bins=None)', {'make_np': make_np, 'make_histogram': make_histogram, 'Summary': Summary, 'name': name, 'values': values, 'bins': bins, 'max_bins': max_bins}, 1)

def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.make_histogram', 'make_histogram(values, bins, max_bins=None)', {'np': np, 'HistogramProto': HistogramProto, 'values': values, 'bins': bins, 'max_bins': max_bins}, 1)

def image(tag, tensor, rescale=1, dataformats='NCHW'):
    """Outputs a `Summary` protocol buffer with images.
    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    The `name` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
        'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32).
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.image', "image(tag, tensor, rescale=1, dataformats='NCHW')", {'make_np': make_np, 'convert_to_HWC': convert_to_HWC, '_calc_scale_factor': _calc_scale_factor, 'np': np, 'make_image': make_image, 'Summary': Summary, 'tag': tag, 'tensor': tensor, 'rescale': rescale, 'dataformats': dataformats}, 1)

def image_boxes(tag, tensor_image, tensor_boxes, rescale=1, dataformats='CHW'):
    """Outputs a `Summary` protocol buffer with images."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.image_boxes', "image_boxes(tag, tensor_image, tensor_boxes, rescale=1, dataformats='CHW')", {'make_np': make_np, 'convert_to_HWC': convert_to_HWC, 'np': np, '_calc_scale_factor': _calc_scale_factor, 'make_image': make_image, 'Summary': Summary, 'tag': tag, 'tensor_image': tensor_image, 'tensor_boxes': tensor_boxes, 'rescale': rescale, 'dataformats': dataformats}, 1)

def draw_boxes(disp_image, boxes):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.draw_boxes', 'draw_boxes(disp_image, boxes)', {'_draw_single_box': _draw_single_box, 'disp_image': disp_image, 'boxes': boxes}, 1)

def make_image(tensor, rescale=1, rois=None):
    """Convert a numpy representation of an image to Image protobuf"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.make_image', 'make_image(tensor, rescale=1, rois=None)', {'draw_boxes': draw_boxes, 'Summary': Summary, 'tensor': tensor, 'rescale': rescale, 'rois': rois}, 1)

def video(tag, tensor, fps=4):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.video', 'video(tag, tensor, fps=4)', {'make_np': make_np, '_prepare_video': _prepare_video, '_calc_scale_factor': _calc_scale_factor, 'np': np, 'make_video': make_video, 'Summary': Summary, 'tag': tag, 'tensor': tensor, 'fps': fps}, 1)

def make_video(tensor, fps):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.make_video', 'make_video(tensor, fps)', {'os': os, 'logging': logging, 'Summary': Summary, 'tensor': tensor, 'fps': fps}, 1)

def audio(tag, tensor, sample_rate=44100):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.audio', 'audio(tag, tensor, sample_rate=44100)', {'make_np': make_np, 'Summary': Summary, 'tag': tag, 'tensor': tensor, 'sample_rate': sample_rate}, 1)

def custom_scalars(layout):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.custom_scalars', 'custom_scalars(layout)', {'layout_pb2': layout_pb2, 'SummaryMetadata': SummaryMetadata, 'TensorProto': TensorProto, 'TensorShapeProto': TensorShapeProto, 'Summary': Summary, 'layout': layout}, 1)

def text(tag, text):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.text', 'text(tag, text)', {'SummaryMetadata': SummaryMetadata, 'TextPluginData': TextPluginData, 'TensorProto': TensorProto, 'TensorShapeProto': TensorShapeProto, 'Summary': Summary, 'tag': tag, 'text': text}, 1)

def pr_curve_raw(tag, tp, fp, tn, fn, precision, recall, num_thresholds=127, weights=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.pr_curve_raw', 'pr_curve_raw(tag, tp, fp, tn, fn, precision, recall, num_thresholds=127, weights=None)', {'np': np, 'PrCurvePluginData': PrCurvePluginData, 'SummaryMetadata': SummaryMetadata, 'TensorProto': TensorProto, 'TensorShapeProto': TensorShapeProto, 'Summary': Summary, 'tag': tag, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'precision': precision, 'recall': recall, 'num_thresholds': num_thresholds, 'weights': weights}, 1)

def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.pr_curve', 'pr_curve(tag, labels, predictions, num_thresholds=127, weights=None)', {'compute_curve': compute_curve, 'PrCurvePluginData': PrCurvePluginData, 'SummaryMetadata': SummaryMetadata, 'TensorProto': TensorProto, 'TensorShapeProto': TensorShapeProto, 'Summary': Summary, 'tag': tag, 'labels': labels, 'predictions': predictions, 'num_thresholds': num_thresholds, 'weights': weights}, 1)

def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.compute_curve', 'compute_curve(labels, predictions, num_thresholds=None, weights=None)', {'np': np, 'labels': labels, 'predictions': predictions, 'num_thresholds': num_thresholds, 'weights': weights}, 1)

def _get_tensor_summary(name, display_name, description, tensor, content_type, components, json_config):
    """Creates a tensor summary with summary metadata.

    Args:
      name: Uniquely identifiable name of the summary op. Could be replaced by
        combination of name and type to make it unique even outside of this
        summary.
      display_name: Will be used as the display name in TensorBoard.
        Defaults to `name`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      tensor: Tensor to display in summary.
      content_type: Type of content inside the Tensor.
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.

    Returns:
      Tensor summary with metadata.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary._get_tensor_summary', '_get_tensor_summary(name, display_name, description, tensor, content_type, components, json_config)', {'TensorProto': TensorProto, 'TensorShapeProto': TensorShapeProto, 'Summary': Summary, 'name': name, 'display_name': display_name, 'description': description, 'tensor': tensor, 'content_type': content_type, 'components': components, 'json_config': json_config}, 1)

def _get_json_config(config_dict):
    """Parses and returns JSON string from python dictionary."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary._get_json_config', '_get_json_config(config_dict)', {'json': json, 'config_dict': config_dict}, 1)

def mesh(tag, vertices, colors, faces, config_dict, display_name=None, description=None):
    """Outputs a merged `Summary` protocol buffer with a mesh/point cloud.

      Args:
        tag: A name for this summary operation.
        vertices: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the 3D
          coordinates of vertices.
        faces: Tensor of shape `[dim_1, ..., dim_n, 3]` containing indices of
          vertices within each triangle.
        colors: Tensor of shape `[dim_1, ..., dim_n, 3]` containing colors for each
          vertex.
        display_name: If set, will be used as the display name in TensorBoard.
          Defaults to `name`.
        description: A longform readable description of the summary data. Markdown
          is supported.
        config_dict: Dictionary with ThreeJS classes names and configuration.

      Returns:
        Merged summary for mesh/point cloud representation.
      """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard.summary.mesh', 'mesh(tag, vertices, colors, faces, config_dict, display_name=None, description=None)', {'_get_json_config': _get_json_config, '_get_tensor_summary': _get_tensor_summary, 'Summary': Summary, 'tag': tag, 'vertices': vertices, 'colors': colors, 'faces': faces, 'config_dict': config_dict, 'display_name': display_name, 'description': description}, 1)

