import tempfile
import contextlib
from . import cudart, check_error
DEFAULT_FLAGS = ['gpustarttimestamp', 'gpuendtimestamp', 'gridsize3d', 'threadblocksize', 'streamid', 'enableonstart 0', 'conckerneltrace']

def init(output_file, flags=None, output_mode='key_value'):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.profiler.init', "init(output_file, flags=None, output_mode='key_value')", {'cudart': cudart, 'DEFAULT_FLAGS': DEFAULT_FLAGS, 'tempfile': tempfile, 'check_error': check_error, 'output_file': output_file, 'flags': flags, 'output_mode': output_mode}, 0)

def start():
    check_error(cudart().cudaProfilerStart())

def stop():
    check_error(cudart().cudaProfilerStop())

@contextlib.contextmanager
def profile():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.profiler.profile', 'profile()', {'start': start, 'stop': stop, 'contextlib': contextlib}, 0)

