import math
import torch
from torch._six import inf


class __PrinterOptions(object):
    precision = 4
    threshold = 1000
    edgeitems = 3
    linewidth = 80
    sci_mode = None

PRINT_OPTS = __PrinterOptions()

def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None):
    """Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by `_Formatter`
    """
    if profile is not None:
        if profile == 'default':
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == 'short':
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == 'full':
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1e10000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode


class _Formatter(object):
    
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.complex_dtype = tensor.dtype.is_complex
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1
        with torch.no_grad():
            tensor_view = tensor.reshape(-1)
        if not self.floating_dtype:
            for value in tensor_view:
                value_str = '{}'.format(value)
                self.max_width = max(self.max_width, len(value_str))
        else:
            nonzero_finite_vals = torch.masked_select(tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0))
            if nonzero_finite_vals.numel() == 0:
                return
            nonzero_finite_abs = nonzero_finite_vals.abs().double()
            nonzero_finite_min = nonzero_finite_abs.min().double()
            nonzero_finite_max = nonzero_finite_abs.max().double()
            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    self.int_mode = False
                    break
            if self.int_mode:
                if (nonzero_finite_max / nonzero_finite_min > 1000.0 or nonzero_finite_max > 100000000.0):
                    self.sci_mode = True
                    for value in nonzero_finite_vals:
                        value_str = '{{:.{}e}}'.format(PRINT_OPTS.precision).format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = '{:.0f}'.format(value)
                        self.max_width = max(self.max_width, len(value_str) + 1)
            elif (nonzero_finite_max / nonzero_finite_min > 1000.0 or nonzero_finite_max > 100000000.0 or nonzero_finite_min < 0.0001):
                self.sci_mode = True
                for value in nonzero_finite_vals:
                    value_str = '{{:.{}e}}'.format(PRINT_OPTS.precision).format(value)
                    self.max_width = max(self.max_width, len(value_str))
            else:
                for value in nonzero_finite_vals:
                    value_str = '{{:.{}f}}'.format(PRINT_OPTS.precision).format(value)
                    self.max_width = max(self.max_width, len(value_str))
        if PRINT_OPTS.sci_mode is not None:
            self.sci_mode = PRINT_OPTS.sci_mode
    
    def width(self):
        return self.max_width
    
    def format(self, value):
        if self.floating_dtype:
            if self.sci_mode:
                ret = '{{:{}.{}e}}'.format(self.max_width, PRINT_OPTS.precision).format(value)
            elif self.int_mode:
                ret = '{:.0f}'.format(value)
                if not ((math.isinf(value) or math.isnan(value))):
                    ret += '.'
            else:
                ret = '{{:.{}f}}'.format(PRINT_OPTS.precision).format(value)
        elif self.complex_dtype:
            p = PRINT_OPTS.precision
            ret = '({{:.{}f}} {{}} {{:.{}f}}j)'.format(p, p).format(value.real, '+-'[value.imag < 0], abs(value.imag))
        else:
            ret = '{}'.format(value)
        return (self.max_width - len(ret)) * ' ' + ret


def _scalar_str(self, formatter):
    return formatter.format(self.item())

def _vector_str(self, indent, formatter, summarize):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._tensor_str._vector_str', '_vector_str(self, indent, formatter, summarize)', {'math': math, 'PRINT_OPTS': PRINT_OPTS, 'self': self, 'indent': indent, 'formatter': formatter, 'summarize': summarize}, 1)

def _tensor_str_with_formatter(self, indent, formatter, summarize):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._tensor_str._tensor_str_with_formatter', '_tensor_str_with_formatter(self, indent, formatter, summarize)', {'_scalar_str': _scalar_str, '_vector_str': _vector_str, 'PRINT_OPTS': PRINT_OPTS, '_tensor_str_with_formatter': _tensor_str_with_formatter, 'self': self, 'indent': indent, 'formatter': formatter, 'summarize': summarize}, 1)

def _tensor_str(self, indent):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._tensor_str._tensor_str', '_tensor_str(self, indent)', {'PRINT_OPTS': PRINT_OPTS, 'torch': torch, '_Formatter': _Formatter, 'get_summarized_data': get_summarized_data, '_tensor_str_with_formatter': _tensor_str_with_formatter, 'self': self, 'indent': indent}, 1)

def _add_suffixes(tensor_str, suffixes, indent, force_newline):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._tensor_str._add_suffixes', '_add_suffixes(tensor_str, suffixes, indent, force_newline)', {'PRINT_OPTS': PRINT_OPTS, 'tensor_str': tensor_str, 'suffixes': suffixes, 'indent': indent, 'force_newline': force_newline}, 1)

def get_summarized_data(self):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._tensor_str.get_summarized_data', 'get_summarized_data(self)', {'PRINT_OPTS': PRINT_OPTS, 'torch': torch, 'get_summarized_data': get_summarized_data, 'self': self}, 1)

def _str(self):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._tensor_str._str', '_str(self)', {'torch': torch, '_tensor_str': _tensor_str, '_add_suffixes': _add_suffixes, 'self': self}, 1)

