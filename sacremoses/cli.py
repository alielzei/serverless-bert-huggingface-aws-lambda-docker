import os
from copy import deepcopy
from functools import partial
from functools import update_wrapper
import click
from sacremoses.tokenize import MosesTokenizer, MosesDetokenizer
from sacremoses.truecase import MosesTruecaser, MosesDetruecaser
from sacremoses.normalize import MosesPunctNormalizer
from sacremoses.util import parallelize_preprocess
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(chain=True, context_settings=CONTEXT_SETTINGS)
@click.option('--language', '-l', default='en', help='Use language specific rules when tokenizing')
@click.option('--processes', '-j', default=1, help='No. of processes.')
@click.option('--encoding', '-e', default='utf8', help='Specify encoding of file.')
@click.option('--quiet', '-q', is_flag=True, default=False, help='Disable progress bar.')
@click.version_option()
def cli(language, encoding, processes, quiet):
    pass
result_callback = (cli.resultcallback if int(click.__version__.split('.')[0]) < 8 else cli.result_callback)

@result_callback()
def process_pipeline(processors, encoding, **kwargs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('sacremoses.cli.process_pipeline', 'process_pipeline(processors, encoding, **kwargs)', {'click': click, 'result_callback': result_callback, 'processors': processors, 'encoding': encoding, 'kwargs': kwargs}, 0)

def processor(f, **kwargs):
    """Helper decorator to rewrite a function so that
    it returns another function from it.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.cli.processor', 'processor(f, **kwargs)', {'partial': partial, 'update_wrapper': update_wrapper, 'f': f, 'kwargs': kwargs}, 1)

def parallel_or_not(iterator, func, processes, quiet):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('sacremoses.cli.parallel_or_not', 'parallel_or_not(iterator, func, processes, quiet)', {'parallelize_preprocess': parallelize_preprocess, 'iterator': iterator, 'func': func, 'processes': processes, 'quiet': quiet}, 0)

@cli.command('tokenize')
@click.option('--aggressive-dash-splits', '-a', default=False, is_flag=True, help='Triggers dash split rules.')
@click.option('--xml-escape', '-x', default=True, is_flag=True, help='Escape special characters for XML.')
@click.option('--protected-patterns', '-p', help='Specify file with patters to be protected in tokenisation. Special values: :basic: :web:')
@click.option('--custom-nb-prefixes', '-c', help='Specify a custom non-breaking prefixes file, add prefixes to the default ones from the specified language.')
@processor
def tokenize_file(iterator, language, processes, quiet, xml_escape, aggressive_dash_splits, protected_patterns, custom_nb_prefixes):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.cli.tokenize_file', 'tokenize_file(iterator, language, processes, quiet, xml_escape, aggressive_dash_splits, protected_patterns, custom_nb_prefixes)', {'MosesTokenizer': MosesTokenizer, 'partial': partial, 'parallel_or_not': parallel_or_not, 'cli': cli, 'click': click, 'processor': processor, 'iterator': iterator, 'language': language, 'processes': processes, 'quiet': quiet, 'xml_escape': xml_escape, 'aggressive_dash_splits': aggressive_dash_splits, 'protected_patterns': protected_patterns, 'custom_nb_prefixes': custom_nb_prefixes}, 1)

@cli.command('detokenize')
@click.option('--xml-unescape', '-x', default=True, is_flag=True, help='Unescape special characters for XML.')
@processor
def detokenize_file(iterator, language, processes, quiet, xml_unescape):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.cli.detokenize_file', 'detokenize_file(iterator, language, processes, quiet, xml_unescape)', {'MosesDetokenizer': MosesDetokenizer, 'partial': partial, 'parallel_or_not': parallel_or_not, 'cli': cli, 'click': click, 'processor': processor, 'iterator': iterator, 'language': language, 'processes': processes, 'quiet': quiet, 'xml_unescape': xml_unescape}, 1)

@cli.command('normalize')
@click.option('--normalize-quote-commas', '-q', default=True, is_flag=True, help='Normalize quotations and commas.')
@click.option('--normalize-numbers', '-d', default=True, is_flag=True, help='Normalize number.')
@click.option('--replace-unicode-puncts', '-p', default=False, is_flag=True, help='Replace unicode punctuations BEFORE normalization.')
@click.option('--remove-control-chars', '-c', default=False, is_flag=True, help='Remove control characters AFTER normalization.')
@processor
def normalize_file(iterator, language, processes, quiet, normalize_quote_commas, normalize_numbers, replace_unicode_puncts, remove_control_chars):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.cli.normalize_file', 'normalize_file(iterator, language, processes, quiet, normalize_quote_commas, normalize_numbers, replace_unicode_puncts, remove_control_chars)', {'MosesPunctNormalizer': MosesPunctNormalizer, 'partial': partial, 'parallel_or_not': parallel_or_not, 'cli': cli, 'click': click, 'processor': processor, 'iterator': iterator, 'language': language, 'processes': processes, 'quiet': quiet, 'normalize_quote_commas': normalize_quote_commas, 'normalize_numbers': normalize_numbers, 'replace_unicode_puncts': replace_unicode_puncts, 'remove_control_chars': remove_control_chars}, 1)

@cli.command('train-truecase')
@click.option('--modelfile', '-m', required=True, help='Filename to save the modelfile.')
@click.option('--is-asr', '-a', default=False, is_flag=True, help='A flag to indicate that model is for ASR.')
@click.option('--possibly-use-first-token', '-p', default=False, is_flag=True, help='Use the first token as part of truecasing.')
@processor
def train_truecaser(iterator, language, processes, quiet, modelfile, is_asr, possibly_use_first_token):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('sacremoses.cli.train_truecaser', 'train_truecaser(iterator, language, processes, quiet, modelfile, is_asr, possibly_use_first_token)', {'MosesTruecaser': MosesTruecaser, 'cli': cli, 'click': click, 'processor': processor, 'iterator': iterator, 'language': language, 'processes': processes, 'quiet': quiet, 'modelfile': modelfile, 'is_asr': is_asr, 'possibly_use_first_token': possibly_use_first_token}, 0)

@cli.command('truecase')
@click.option('--modelfile', '-m', required=True, help='Filename to save/load the modelfile.')
@click.option('--is-asr', '-a', default=False, is_flag=True, help='A flag to indicate that model is for ASR.')
@click.option('--possibly-use-first-token', '-p', default=False, is_flag=True, help='Use the first token as part of truecase training.')
@processor
def truecase_file(iterator, language, processes, quiet, modelfile, is_asr, possibly_use_first_token):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.cli.truecase_file', 'truecase_file(iterator, language, processes, quiet, modelfile, is_asr, possibly_use_first_token)', {'os': os, 'deepcopy': deepcopy, 'MosesTruecaser': MosesTruecaser, 'partial': partial, 'parallel_or_not': parallel_or_not, 'cli': cli, 'click': click, 'processor': processor, 'iterator': iterator, 'language': language, 'processes': processes, 'quiet': quiet, 'modelfile': modelfile, 'is_asr': is_asr, 'possibly_use_first_token': possibly_use_first_token}, 1)

@cli.command('detruecase')
@click.option('--is-headline', '-a', default=False, is_flag=True, help='Whether the file are headlines.')
@processor
def detruecase_file(iterator, language, processes, quiet, is_headline):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sacremoses.cli.detruecase_file', 'detruecase_file(iterator, language, processes, quiet, is_headline)', {'MosesDetruecaser': MosesDetruecaser, 'partial': partial, 'parallel_or_not': parallel_or_not, 'cli': cli, 'click': click, 'processor': processor, 'iterator': iterator, 'language': language, 'processes': processes, 'quiet': quiet, 'is_headline': is_headline}, 1)

