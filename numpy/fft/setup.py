import sys

def configuration(parent_package='', top_path=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.fft.setup.configuration', "configuration(parent_package='', top_path=None)", {'sys': sys, 'parent_package': parent_package, 'top_path': top_path}, 1)
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

