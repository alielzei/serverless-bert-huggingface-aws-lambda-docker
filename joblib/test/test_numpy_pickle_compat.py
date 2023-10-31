"""Test the old numpy pickler, compatibility version."""

from joblib import numpy_pickle_compat

def test_z_file(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_numpy_pickle_compat.test_z_file', 'test_z_file(tmpdir)', {'numpy_pickle_compat': numpy_pickle_compat, 'tmpdir': tmpdir}, 0)

