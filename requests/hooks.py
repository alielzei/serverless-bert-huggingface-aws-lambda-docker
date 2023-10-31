"""
requests.hooks
~~~~~~~~~~~~~~

This module provides the capabilities for the Requests hooks system.

Available hooks:

``response``:
    The response generated from a Request.
"""

HOOKS = ['response']

def default_hooks():
    return {event: [] for event in HOOKS}

def dispatch_hook(key, hooks, hook_data, **kwargs):
    """Dispatches a hook dictionary on a given piece of data."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.hooks.dispatch_hook', 'dispatch_hook(key, hooks, hook_data, **kwargs)', {'key': key, 'hooks': hooks, 'hook_data': hook_data, 'kwargs': kwargs}, 1)

