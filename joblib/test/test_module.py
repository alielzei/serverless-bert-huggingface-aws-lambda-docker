import sys
import joblib
from joblib.testing import check_subprocess_call
from joblib.test.common import with_multiprocessing

def test_version():
    assert hasattr(joblib, '__version__'), 'There are no __version__ argument on the joblib module'

@with_multiprocessing
def test_no_start_method_side_effect_on_import():
    code = 'if True:\n        import joblib\n        import multiprocessing as mp\n        # The following line would raise RuntimeError if the\n        # start_method is already set.\n        mp.set_start_method("loky")\n    '
    check_subprocess_call([sys.executable, '-c', code])

@with_multiprocessing
def test_no_semaphore_tracker_on_import():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_module.test_no_semaphore_tracker_on_import', 'test_no_semaphore_tracker_on_import()', {'sys': sys, 'check_subprocess_call': check_subprocess_call, 'with_multiprocessing': with_multiprocessing}, 0)

@with_multiprocessing
def test_no_resource_tracker_on_import():
    code = 'if True:\n        import joblib\n        from joblib.externals.loky.backend import resource_tracker\n        # The following line would raise RuntimeError if the\n        # start_method is already set.\n        msg = "loky.resource_tracker has been spawned on import"\n        assert resource_tracker._resource_tracker._fd is None, msg\n    '
    check_subprocess_call([sys.executable, '-c', code])

