""" Logging utilities. """

import logging
import os
import threading
from logging import CRITICAL
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import NOTSET
from logging import WARN
from logging import WARNING
from typing import Optional
_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None
log_levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
_default_log_level = logging.WARNING

def _get_default_logging_level():
    """
    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level.
    If it is not - fall back to ``_default_log_level``
    """
    env_level_str = os.getenv('TRANSFORMERS_VERBOSITY', None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, has to be one of: {', '.join(log_levels.keys())}")
    return _default_log_level

def _get_library_name() -> str:
    return __name__.split('.')[0]

def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())

def _configure_library_root_logger() -> None:
    global _default_handler
    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler()
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False

def _reset_library_root_logger() -> None:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.utils.logging._reset_library_root_logger', '_reset_library_root_logger()', {'_lock': _lock, '_get_library_root_logger': _get_library_root_logger, 'logging': logging}, 1)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.utils.logging.get_logger', 'get_logger(name=None)', {'_get_library_name': _get_library_name, '_configure_library_root_logger': _configure_library_root_logger, 'logging': logging, 'name': name, 'Optional': Optional, 'str': str, 'logging': logging}, 1)

def get_verbosity() -> int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        :obj:`int`: The logging level.

    .. note::

        🤗 Transformers has following logging levels:

        - 50: ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
        - 40: ``transformers.logging.ERROR``
        - 30: ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
        - 20: ``transformers.logging.INFO``
        - 10: ``transformers.logging.DEBUG``
    """
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()

def set_verbosity(verbosity: int) -> None:
    """
    Set the vebosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (:obj:`int`):
            Logging level, e.g., one of:

            - ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
            - ``transformers.logging.ERROR``
            - ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
            - ``transformers.logging.INFO``
            - ``transformers.logging.DEBUG``
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)

def set_verbosity_info():
    """Set the verbosity to the :obj:`INFO` level."""
    return set_verbosity(INFO)

def set_verbosity_warning():
    """Set the verbosity to the :obj:`WARNING` level."""
    return set_verbosity(WARNING)

def set_verbosity_debug():
    """Set the verbosity to the :obj:`DEBUG` level."""
    return set_verbosity(DEBUG)

def set_verbosity_error():
    """Set the verbosity to the :obj:`ERROR` level."""
    return set_verbosity(ERROR)

def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.utils.logging.disable_default_handler', 'disable_default_handler()', {'_configure_library_root_logger': _configure_library_root_logger, '_default_handler': _default_handler, '_get_library_root_logger': _get_library_root_logger}, 0)

def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.utils.logging.enable_default_handler', 'enable_default_handler()', {'_configure_library_root_logger': _configure_library_root_logger, '_default_handler': _default_handler, '_get_library_root_logger': _get_library_root_logger}, 0)

def disable_propagation() -> None:
    """Disable propagation of the library log outputs.
    Note that log propagation is disabled by default.
    """
    _configure_library_root_logger()
    _get_library_root_logger().propagate = False

def enable_propagation() -> None:
    """Enable propagation of the library log outputs.
    Please disable the HuggingFace Transformers's default handler to prevent double logging if the root logger has
    been configured.
    """
    _configure_library_root_logger()
    _get_library_root_logger().propagate = True

def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:

    ::

        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE

    All handlers currently bound to the root logger are affected by this method.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.utils.logging.enable_explicit_format', 'enable_explicit_format()', {'_get_library_root_logger': _get_library_root_logger, 'logging': logging}, 0)

def reset_format() -> None:
    """
    Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.utils.logging.reset_format', 'reset_format()', {'_get_library_root_logger': _get_library_root_logger}, 0)

