import os
import sys
import tempfile

def run_command(cmd):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.f2py.diagnose.run_command', 'run_command(cmd)', {'os': os, 'cmd': cmd}, 0)

def run():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.f2py.diagnose.run', 'run()', {'os': os, 'tempfile': tempfile, 'sys': sys}, 0)
if __name__ == '__main__':
    run()

