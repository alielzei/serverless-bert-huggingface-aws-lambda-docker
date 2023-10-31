
def configuration(parent_package='', top_path=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.setup.configuration', "configuration(parent_package='', top_path=None)", {'parent_package': parent_package, 'top_path': top_path}, 1)
if __name__ == '__main__':
    print('This is the wrong setup.py file to run')

