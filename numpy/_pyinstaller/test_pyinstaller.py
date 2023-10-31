import subprocess
from pathlib import Path
import pytest

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::ResourceWarning')
@pytest.mark.parametrize('mode', ['--onedir', '--onefile'])
@pytest.mark.slow
def test_pyinstaller(mode, tmp_path):
    """Compile and run pyinstaller-smoke.py using PyInstaller."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy._pyinstaller.test_pyinstaller.test_pyinstaller', 'test_pyinstaller(mode, tmp_path)', {'Path': Path, '__file__': __file__, 'subprocess': subprocess, 'pytest': pytest, 'mode': mode, 'tmp_path': tmp_path}, 0)

