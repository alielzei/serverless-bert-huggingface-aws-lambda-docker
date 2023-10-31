from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from textwrap import dedent
from subprocess import call

def parse_lines(lines):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.core.nomnigraph.op_gen.parse_lines', 'parse_lines(lines)', {'lines': lines}, 2)

def gen_class(op, op_def):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.core.nomnigraph.op_gen.gen_class', 'gen_class(op, op_def)', {'dedent': dedent, 'op': op, 'op_def': op_def}, 1)

def gen_classes(ops, op_list):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.core.nomnigraph.op_gen.gen_classes', 'gen_classes(ops, op_list)', {'gen_class': gen_class, 'ops': ops, 'op_list': op_list}, 1)

def gen_enum(op_list):
    return ',\n'.join([op for op in op_list]) + '\n'

def gen_names(op_list):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.core.nomnigraph.op_gen.gen_names', 'gen_names(op_list)', {'dedent': dedent, 'op_list': op_list}, 1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate op files.')
    parser.add_argument('--install_dir', help='installation directory')
    parser.add_argument('--source_def', help='ops.def', action='append')
    args = parser.parse_args()
    install_dir = args.install_dir
    sources = args.source_def
    lines = []
    for source in sources:
        with open(source, 'rb') as f:
            lines_tmp = f.readlines()
            lines += [l.strip().decode('utf-8') for l in lines_tmp]
    (ops, op_list) = parse_lines(lines)
    with open(install_dir + '/OpClasses.h', 'wb') as f:
        f.write(gen_classes(ops, op_list).encode('utf-8'))
    with open(install_dir + '/OpNames.h', 'wb') as f:
        f.write(gen_names(op_list).encode('utf-8'))
    with open(install_dir + '/OpEnum.h', 'wb') as f:
        f.write(gen_enum(op_list).encode('utf-8'))
    try:
        cmd = ['clang-format', '-i', install_dir + '/OpClasses.h']
        call(cmd)
        cmd = ['clang-format', '-i', install_dir + '/OpNames.h']
        call(cmd)
        cmd = ['clang-format', '-i', install_dir + '/OpEnum.h']
        call(cmd)
    except Exception:
        pass

