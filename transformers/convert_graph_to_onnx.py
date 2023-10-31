from argparse import ArgumentParser
from os import listdir, makedirs
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from packaging.version import Version, parse
from transformers import is_tf_available, is_torch_available
from transformers.file_utils import ModelOutput
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding
ORT_QUANTIZE_MINIMUM_VERSION = parse('1.4.0')
SUPPORTED_PIPELINES = ['feature-extraction', 'ner', 'sentiment-analysis', 'fill-mask', 'question-answering', 'text-generation', 'translation_en_to_fr', 'translation_en_to_de', 'translation_en_to_ro']


class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """
    
    def __init__(self):
        super().__init__('ONNX Converter')
        self.add_argument('--pipeline', type=str, choices=SUPPORTED_PIPELINES, default='feature-extraction')
        self.add_argument('--model', type=str, required=True, help="Model's id or path (ex: bert-base-cased)")
        self.add_argument('--tokenizer', type=str, help="Tokenizer's id or path (ex: bert-base-cased)")
        self.add_argument('--framework', type=str, choices=['pt', 'tf'], help='Framework for loading the model')
        self.add_argument('--opset', type=int, default=11, help='ONNX opset to use')
        self.add_argument('--check-loading', action='store_true', help='Check ONNX is able to load the model')
        self.add_argument('--use-external-format', action='store_true', help='Allow exporting model >= than 2Gb')
        self.add_argument('--quantize', action='store_true', help='Quantize the neural network to be run with int8')
        self.add_argument('output')


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Append a string-identifier at the end (before the extension,  if any) to the provided filepath.
    Args:
        filename: pathlib.Path The actual path object we would like to add an identifier suffix
        identifier: The suffix to add

    Returns: String with concatenated indentifier at the end of the filename
    """
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)

def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough.
    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.check_onnxruntime_requirements', 'check_onnxruntime_requirements(minimum_version)', {'parse': parse, 'ORT_QUANTIZE_MINIMUM_VERSION': ORT_QUANTIZE_MINIMUM_VERSION, 'minimum_version': minimum_version}, 0)

def ensure_valid_input(model, tokens, input_names):
    """
    Ensure input are presented in the correct order, without any None
    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs

    Returns: Tuple

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.ensure_valid_input', 'ensure_valid_input(model, tokens, input_names)', {'model': model, 'tokens': tokens, 'input_names': input_names}, 2)

def infer_shapes(nlp: Pipeline, framework: str) -> Tuple[(List[str], List[str], Dict, BatchEncoding)]:
    """
    Attempt to infer the static vs dynamic axes for each input and output tensors for a specific model.
    Args:
        nlp: The pipeline object holding the model to be exported
        framework: The framework identifier to dispatch to the correct inference scheme (pt/tf)

    Returns:
        - List of the inferred input variable names
        - List of the inferred output variable names
        - Dictionary with input/output variables names as key and shape tensor as value
        - a BatchEncoding reference which was used to infer all the above information
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.infer_shapes', 'infer_shapes(nlp, framework)', {'ModelOutput': ModelOutput, 'nlp': nlp, 'framework': framework, 'Tuple': Tuple}, 1)

def load_graph_from_args(pipeline_name: str, framework: str, model: str, tokenizer: Optional[str] = None) -> Pipeline:
    """
    Convert the set of arguments provided through the CLI to an actual pipeline reference (tokenizer + model)
    Args:
        pipeline_name: The kind of pipeline to use (ner, question-answering, etc.)
        framework: The actual model to convert the pipeline from ("pt" or "tf")
        model: The model name which will be loaded by the pipeline
        tokenizer: The tokenizer name which will be loaded by the pipeline, default to the model's value

    Returns: Pipeline object

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.load_graph_from_args', 'load_graph_from_args(pipeline_name, framework, model, tokenizer=None)', {'is_torch_available': is_torch_available, 'is_tf_available': is_tf_available, 'pipeline': pipeline, 'pipeline_name': pipeline_name, 'framework': framework, 'model': model, 'tokenizer': tokenizer, 'Optional': Optional, 'str': str}, 1)

def convert_pytorch(nlp: Pipeline, opset: int, output: Path, use_external_format: bool):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR)
    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB

    Returns:

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.convert_pytorch', 'convert_pytorch(nlp, opset, output, use_external_format)', {'is_torch_available': is_torch_available, 'infer_shapes': infer_shapes, 'ensure_valid_input': ensure_valid_input, 'nlp': nlp, 'opset': opset, 'output': output, 'use_external_format': use_external_format}, 0)

def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR)
    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model

    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.convert_tensorflow', 'convert_tensorflow(nlp, opset, output)', {'is_tf_available': is_tf_available, 'infer_shapes': infer_shapes, 'nlp': nlp, 'opset': opset, 'output': output}, 0)

def convert(framework: str, model: str, output: Path, opset: int, tokenizer: Optional[str] = None, use_external_format: bool = False, pipeline_name: str = 'feature-extraction'):
    """
    Convert the pipeline object to the ONNX Intermediate Representation (IR) format.
    Args:
        framework: The framework the pipeline is backed by ("pt" or "tf")
        model: The name of the model to load for the pipeline
        output: The path where the ONNX graph will be stored
        opset: The actual version of the ONNX operator set to use
        tokenizer: The name of the model to load for the pipeline, default to the model's name if not provided
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB (PyTorch only)
        pipeline_name: The kind of pipeline to instantiate (ner, question-answering, etc.)

    Returns:

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.convert', "convert(framework, model, output, opset, tokenizer=None, use_external_format=False, pipeline_name='feature-extraction')", {'load_graph_from_args': load_graph_from_args, 'makedirs': makedirs, 'listdir': listdir, 'convert_pytorch': convert_pytorch, 'convert_tensorflow': convert_tensorflow, 'framework': framework, 'model': model, 'output': output, 'opset': opset, 'tokenizer': tokenizer, 'use_external_format': use_external_format, 'pipeline_name': pipeline_name, 'Optional': Optional, 'str': str}, 0)

def optimize(onnx_model_path: Path) -> Path:
    """
    Load the model at the specified path and let onnxruntime look at transformations on the graph
    to enable all the optimizations possible
    Args:
        onnx_model_path: filepath where the model binary description is stored

    Returns: Path where the optimized model binary description has been saved

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.optimize', 'optimize(onnx_model_path)', {'generate_identified_filename': generate_identified_filename, 'onnx_model_path': onnx_model_path}, 1)

def quantize(onnx_model_path: Path) -> Path:
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU.
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored

    Returns: The Path generated for the quantized
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.quantize', 'quantize(onnx_model_path)', {'generate_identified_filename': generate_identified_filename, 'onnx_model_path': onnx_model_path}, 1)

def verify(path: Path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_graph_to_onnx.verify', 'verify(path)', {'path': path}, 0)
if __name__ == '__main__':
    parser = OnnxConverterArgumentParser()
    args = parser.parse_args()
    args.output = Path(args.output).absolute()
    try:
        print('\n====== Converting model to ONNX ======')
        convert(args.framework, args.model, args.output, args.opset, args.tokenizer, args.use_external_format, args.pipeline)
        if args.quantize:
            check_onnxruntime_requirements(ORT_QUANTIZE_MINIMUM_VERSION)
            if args.framework == 'tf':
                print('\t Using TensorFlow might not provide the same optimization level compared to PyTorch.\n\t For TensorFlow users you can try optimizing the model directly through onnxruntime_tools.\n\t For more information, please refer to the onnxruntime documentation:\n\t\thttps://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers\n')
            print('\n====== Optimizing ONNX model ======')
            args.optimized_output = optimize(args.output)
            args.quantized_output = quantize(args.optimized_output)
        if args.check_loading:
            print('\n====== Check exported ONNX model(s) ======')
            verify(args.output)
            if hasattr(args, 'optimized_output'):
                verify(args.optimized_output)
            if hasattr(args, 'quantized_output'):
                verify(args.quantized_output)
    except Exception as e:
        print(f'Error while converting the model: {e}')
        exit(1)

