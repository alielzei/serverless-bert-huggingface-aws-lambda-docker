import mmap
from joblib.backports import make_memmap, concurrency_safe_rename
from joblib.test.common import with_numpy
from joblib.testing import parametrize
from joblib import Parallel, delayed

@with_numpy
def test_memmap(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_backports.test_memmap', 'test_memmap(tmpdir)', {'mmap': mmap, 'make_memmap': make_memmap, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@parametrize('dst_content', [None, 'dst content'])
@parametrize('backend', [None, 'threading'])
def test_concurrency_safe_rename(tmpdir, dst_content, backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_backports.test_concurrency_safe_rename', 'test_concurrency_safe_rename(tmpdir, dst_content, backend)', {'Parallel': Parallel, 'delayed': delayed, 'concurrency_safe_rename': concurrency_safe_rename, 'parametrize': parametrize, 'tmpdir': tmpdir, 'dst_content': dst_content, 'backend': backend}, 0)

