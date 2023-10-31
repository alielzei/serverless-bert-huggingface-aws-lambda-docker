import numpy as np

def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._utils.figure_to_image', 'figure_to_image(figures, close=True)', {'np': np, 'figures': figures, 'close': close}, 1)

def _prepare_video(V):
    """
    Converts a 5D tensor [batchsize, time(frame), channel(color), height, width]
    into 4D tensor with dimension [time(frame), new_width, new_height, channel].
    A batch of images are spreaded to a grid, which forms a frame.
    e.g. Video with batchsize 16 will have a 4x4 grid.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._utils._prepare_video', '_prepare_video(V)', {'np': np, 'V': V}, 1)

def make_grid(I, ncols=8):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._utils.make_grid', 'make_grid(I, ncols=8)', {'np': np, 'I': I, 'ncols': ncols}, 1)

def convert_to_HWC(tensor, input_format):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._utils.convert_to_HWC', 'convert_to_HWC(tensor, input_format)', {'make_grid': make_grid, 'np': np, 'tensor': tensor, 'input_format': input_format}, 1)

