from __future__ import absolute_import, division, print_function, unicode_literals
import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import torch
import warnings
import zipfile
if sys.version_info[0] == 2:
    from urlparse import urlparse
    from urllib2 import urlopen
else:
    from urllib.request import urlopen
    from urllib.parse import urlparse
try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        
        
        class tqdm(object):
            
            def __init__(self, total=None, disable=False, unit=None, unit_scale=None, unit_divisor=None):
                self.total = total
                self.disable = disable
                self.n = 0
            
            def update(self, n):
                if self.disable:
                    return
                self.n += n
                if self.total is None:
                    sys.stderr.write('\r{0:.1f} bytes'.format(self.n))
                else:
                    sys.stderr.write('\r{0:.1f}%'.format(100 * self.n / float(self.total)))
                sys.stderr.flush()
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return
                sys.stderr.write('\n')
        
HASH_REGEX = re.compile('-([a-f0-9]*)\\.')
MASTER_BRANCH = 'master'
ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
VAR_DEPENDENCY = 'dependencies'
MODULE_HUBCONF = 'hubconf.py'
READ_DATA_CHUNK = 8192
hub_dir = None

def import_module(name, path):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub.import_module', 'import_module(name, path)', {'sys': sys, 'importlib': importlib, 'name': name, 'path': path}, 1)

def _remove_if_exists(path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.hub._remove_if_exists', '_remove_if_exists(path)', {'os': os, 'shutil': shutil, 'path': path}, 0)

def _git_archive_link(repo_owner, repo_name, branch):
    return 'https://github.com/{}/{}/archive/{}.zip'.format(repo_owner, repo_name, branch)

def _load_attr_from_module(module, func_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub._load_attr_from_module', '_load_attr_from_module(module, func_name)', {'module': module, 'func_name': func_name}, 1)

def _get_torch_home():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub._get_torch_home', '_get_torch_home()', {'hub_dir': hub_dir, 'os': os, 'ENV_TORCH_HOME': ENV_TORCH_HOME, 'ENV_XDG_CACHE_HOME': ENV_XDG_CACHE_HOME, 'DEFAULT_CACHE_DIR': DEFAULT_CACHE_DIR}, 1)

def _setup_hubdir():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.hub._setup_hubdir', '_setup_hubdir()', {'os': os, 'warnings': warnings, '_get_torch_home': _get_torch_home}, 0)

def _parse_repo_info(github):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub._parse_repo_info', '_parse_repo_info(github)', {'MASTER_BRANCH': MASTER_BRANCH, 'github': github}, 3)

def _get_cache_or_reload(github, force_reload, verbose=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub._get_cache_or_reload', '_get_cache_or_reload(github, force_reload, verbose=True)', {'_parse_repo_info': _parse_repo_info, 'os': os, 'hub_dir': hub_dir, 'sys': sys, '_remove_if_exists': _remove_if_exists, '_git_archive_link': _git_archive_link, 'download_url_to_file': download_url_to_file, 'zipfile': zipfile, 'shutil': shutil, 'github': github, 'force_reload': force_reload, 'verbose': verbose}, 1)

def _check_module_exists(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub._check_module_exists', '_check_module_exists(name)', {'sys': sys, 'importlib': importlib, 'name': name}, 1)

def _check_dependencies(m):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.hub._check_dependencies', '_check_dependencies(m)', {'_load_attr_from_module': _load_attr_from_module, 'VAR_DEPENDENCY': VAR_DEPENDENCY, '_check_module_exists': _check_module_exists, 'm': m}, 0)

def _load_entry_from_hubconf(m, model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub._load_entry_from_hubconf', '_load_entry_from_hubconf(m, model)', {'_check_dependencies': _check_dependencies, '_load_attr_from_module': _load_attr_from_module, 'm': m, 'model': model}, 1)

def set_dir(d):
    """
    Optionally set hub_dir to a local dir to save downloaded models & weights.

    If ``set_dir`` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if the environment
    variable is not set.


    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    global hub_dir
    hub_dir = d

def list(github, force_reload=False):
    """
    List all entrypoints available in `github` hubconf.

    Args:
        github (string): a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub.list', 'list(github, force_reload=False)', {'_setup_hubdir': _setup_hubdir, '_get_cache_or_reload': _get_cache_or_reload, 'sys': sys, 'import_module': import_module, 'MODULE_HUBCONF': MODULE_HUBCONF, 'github': github, 'force_reload': force_reload}, 1)

def help(github, model, force_reload=False):
    """
    Show the docstring of entrypoint `model`.

    Args:
        github (string): a string with format <repo_owner/repo_name[:tag_name]> with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Example:
        >>> print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub.help', 'help(github, model, force_reload=False)', {'_setup_hubdir': _setup_hubdir, '_get_cache_or_reload': _get_cache_or_reload, 'sys': sys, 'import_module': import_module, 'MODULE_HUBCONF': MODULE_HUBCONF, '_load_entry_from_hubconf': _load_entry_from_hubconf, 'github': github, 'model': model, 'force_reload': force_reload}, 1)

def load(github, model, *args, **kwargs):
    """
    Load a model from a github repo, with pretrained weights.

    Args:
        github (string): a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        *args (optional): the corresponding args for callable `model`.
        force_reload (bool, optional): whether to force a fresh download of github repo unconditionally.
            Default is `False`.
        verbose (bool, optional): If False, mute messages about hitting local caches. Note that the message
            about first download is cannot be muted.
            Default is `True`.
        **kwargs (optional): the corresponding kwargs for callable `model`.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub.load', 'load(github, model, *args, **kwargs)', {'_setup_hubdir': _setup_hubdir, '_get_cache_or_reload': _get_cache_or_reload, 'sys': sys, 'import_module': import_module, 'MODULE_HUBCONF': MODULE_HUBCONF, '_load_entry_from_hubconf': _load_entry_from_hubconf, 'github': github, 'model': model, 'args': args, 'kwargs': kwargs}, 1)

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.hub.download_url_to_file', 'download_url_to_file(url, dst, hash_prefix=None, progress=True)', {'urlopen': urlopen, 'os': os, 'tempfile': tempfile, 'hashlib': hashlib, 'tqdm': tqdm, 'shutil': shutil, 'url': url, 'dst': dst, 'hash_prefix': hash_prefix, 'progress': progress}, 0)

def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
    warnings.warn('torch.hub._download_url_to_file has been renamed to            torch.hub.download_url_to_file to be a public API,            _download_url_to_file will be removed in after 1.3 release')
    download_url_to_file(url, dst, hash_prefix, progress)

def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False):
    """Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.hub.load_state_dict_from_url', 'load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False)', {'os': os, 'warnings': warnings, '_get_torch_home': _get_torch_home, 'errno': errno, 'urlparse': urlparse, 'sys': sys, 'HASH_REGEX': HASH_REGEX, 'download_url_to_file': download_url_to_file, 'zipfile': zipfile, 'torch': torch, 'url': url, 'model_dir': model_dir, 'map_location': map_location, 'progress': progress, 'check_hash': check_hash}, 1)

