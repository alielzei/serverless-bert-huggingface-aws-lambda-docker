import sys
import yaml
import argparse
import os
from copy import deepcopy
parser = argparse.ArgumentParser()
parser.add_argument('--template_dir', default='.', help='where template.h is')
parser.add_argument('--yaml_dir', default='aten/src/ATen/ATen', help='where ATen yaml files are')
parser.add_argument('--output_prefix', default='', help='')
parser.add_argument('--install_dir', default='.', help='where to put generated file')
parser.add_argument('--aten_root', default='', help='root directory of aten')
(args, _) = parser.parse_known_args()
if args.aten_root:
    if not os.path.exists(args.aten_root):
        raise ValueError('aten_root ({}) does not exist'.format(args.aten_root))
    sys.path.append(os.path.join(args.aten_root, 'src', 'ATen'))
    from code_template import CodeTemplate as CT
else:
    from src.ATen.code_template import CodeTemplate as CT
OP_TEMPLATE = CT.from_file(os.path.join(args.template_dir, 'aten_op_template.h'))
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def write(filename, s):
    with open(filename, 'w') as f:
        f.write(s)

def read(filename):
    with open(filename, 'r') as f:
        return f.read()

def value_has_tensors(v):
    return ('Tensor' in v['dynamic_type'] and 'Sparse' not in v['dynamic_type'])

def value_is_tensor_type(v):
    return (value_has_tensors(v) and v['dynamic_type'] != 'TensorList')
RETURN_MAP = {'Tensor': 'assignTo(Output(${offset}),${output});', 'Scalar': 'assignTo(Output(${offset}),${output}.type(), ${output});', 'bool': 'assignToValue<int64_t>(Output(${offset}),${output});', 'int64_t': 'assignToValue<int64_t>(Output(${offset}),${output});', 'std::vector<Tensor>': 'assignListStartingAt(${offset}, ${output});'}
ARGUMENT_MAP = {'Scalar': 'at::Scalar ${arg} = readScalarAttribute("${arg}");', 'bool': 'bool ${arg} = readAttribute<int64_t>("${arg}");', 'int': 'int ${arg} = readAttribute<int64_t>("${arg}");', 'double': 'double ${arg} = readAttribute<float>("${arg}");', 'int64_t': 'int64_t ${arg} = readAttribute<int64_t>("${arg}");', 'IntArrayRef': 'auto ${arg} = readIntArrayRef("${arg}");', 'std::array<bool,2>': 'auto ${arg} = readBoolMask<2>("${arg}");', 'std::array<bool,3>': 'auto ${arg} = readBoolMask<3>("${arg}");'}

def expand(o):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.aten.gen_op.expand', 'expand(o)', {'deepcopy': deepcopy, 'o': o}, 1)

def supports(o, factory_methods):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.aten.gen_op.supports', 'supports(o, factory_methods)', {'value_has_tensors': value_has_tensors, 'RETURN_MAP': RETURN_MAP, 'ARGUMENT_MAP': ARGUMENT_MAP, 'o': o, 'factory_methods': factory_methods}, 1)
OPTION_TEMPLATE = CT('case ${key}: { // ${name}\n    ${initialization}\n    run_op = [=] {\n        at::AutoNonVariableTypeMode guard;\n        ${statements}\n        auto the_result = ${invocation};\n        ${assignments}\n        return true;\n    };\n} break;\n')
ASSIGN_CHECK_SIZE_TEMPLATE = CT('  if(OutputSize() > ${offset}) {${assignment}}\n')

def get_output(o, i):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.aten.gen_op.get_output', 'get_output(o, i)', {'o': o, 'i': i}, 1)

def attribute_names(o):
    return sorted([a['name'] for a in o['arguments'] if not value_has_tensors(a)])

def required_attribute_names(o):
    return sorted([a['name'] for a in o['arguments'] if (not value_has_tensors(a) and 'default' not in a)])

def self_as_first_argument(arguments):
    return [a for a in arguments if a['name'] == 'self'] + [a for a in arguments if a['name'] != 'self']

def get_num_inputs(o):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.aten.gen_op.get_num_inputs', 'get_num_inputs(o)', {'value_has_tensors': value_has_tensors, 'o': o}, 1)

def find_factory_methods(decls):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.aten.gen_op.find_factory_methods', 'find_factory_methods(decls)', {'decls': decls}, 1)

def emit_assignments(o, env):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.aten.gen_op.emit_assignments', 'emit_assignments(o, env)', {'RETURN_MAP': RETURN_MAP, 'value_is_tensor_type': value_is_tensor_type, 'CT': CT, 'get_output': get_output, 'ASSIGN_CHECK_SIZE_TEMPLATE': ASSIGN_CHECK_SIZE_TEMPLATE, 'o': o, 'env': env}, 0)
if __name__ == '__main__':
    decls = yaml.load(read(os.path.join(args.yaml_dir, 'Declarations.yaml')), Loader=Loader)
    factory_methods = find_factory_methods(decls)
    filtered = [expanded for o in decls for expanded in expand(o) if supports(expanded, factory_methods)]
    top_env = {'mappings': [], 'implementations': []}
    seen = set()
    key = 0
    for o in filtered:
        attr_names = attribute_names(o)
        num_inputs = get_num_inputs(o)
        descriptor = '-'.join([o['name']] + attr_names + [num_inputs])
        if descriptor in seen:
            continue
        seen.add(descriptor)
        top_env['mappings'].append('{{ "{}", {} }},'.format(descriptor, key))
        env = {'name': o['name'], 'statements': [], 'arguments': [], 'assignments': [], 'initialization': [], 'key': str(key)}
        if ('namespace' not in o['method_of'] and 'Tensor' not in o['method_of']):
            assert 'Type' in o['method_of']
        static_tensor_inputs = sum(((arg['type'] != 'TensorList' and value_is_tensor_type(arg)) for arg in o['arguments']))
        has_tensorlist = any((arg['type'] == 'TensorList' for arg in o['arguments']))
        if has_tensorlist:
            tensorlist_idx = [i for (i, arg) in enumerate(o['arguments']) if arg['type'] == 'TensorList'][0]
        real_inputs = 0
        for (i, arg) in enumerate(o['arguments']):
            env['arguments'].append(arg['name'])
            view_length = ('InputSize()' if (has_tensorlist and i < tensorlist_idx) else static_tensor_inputs)
            if arg['type'] == 'TensorList':
                env['statements'].append('auto {} = peekSlice({}, InputSize() - {}, InputSize());'.format(arg['name'], real_inputs, static_tensor_inputs))
            elif value_is_tensor_type(arg):
                env['statements'].append('auto {} = peek({}, {});'.format(arg['name'], real_inputs, view_length))
                real_inputs += 1
            else:
                init = CT(ARGUMENT_MAP[arg['type']]).substitute(env, arg=arg['name'])
                env['initialization'].append(init)
        emit_assignments(o, env)
        if 'namespace' in o['method_of']:
            env['invocation'] = CT('at::${name}(${arguments})').substitute(env)
        else:
            assert 'Tensor' in o['method_of']
            env['invocation'] = 'self.{}({})'.format(o['name'], ', '.join(env['arguments'][1:]))
        top_env['implementations'].append(OPTION_TEMPLATE.substitute(env))
        key += 1
    write(os.path.join(args.install_dir, args.output_prefix + 'aten_op.h'), OP_TEMPLATE.substitute(top_env))

