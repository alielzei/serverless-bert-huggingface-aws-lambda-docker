
def configuration(parent_package='', top_path=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing.setup.configuration', "configuration(parent_package='', top_path=None)", {'parent_package': parent_package, 'top_path': top_path}, 1)
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='NumPy Developers', maintainer_email='numpy-dev@numpy.org', description='NumPy test module', url='https://www.numpy.org', license='NumPy License (BSD Style)', configuration=configuration)

