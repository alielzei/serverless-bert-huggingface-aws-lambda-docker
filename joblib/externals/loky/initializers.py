import warnings

def _viztracer_init(init_kwargs):
    """Initialize viztracer's profiler in worker processes"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.initializers._viztracer_init', '_viztracer_init(init_kwargs)', {'init_kwargs': init_kwargs}, 0)

def _make_viztracer_initializer_and_initargs():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.initializers._make_viztracer_initializer_and_initargs', '_make_viztracer_initializer_and_initargs()', {'_viztracer_init': _viztracer_init, 'warnings': warnings}, 2)


class _ChainedInitializer:
    """Compound worker initializer

    This is meant to be used in conjunction with _chain_initializers to
    produce  the necessary chained_args list to be passed to __call__.
    """
    
    def __init__(self, initializers):
        self._initializers = initializers
    
    def __call__(self, *chained_args):
        for (initializer, args) in zip(self._initializers, chained_args):
            initializer(*args)


def _chain_initializers(initializer_and_args):
    """Convenience helper to combine a sequence of initializers.

    If some initializers are None, they are filtered out.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.initializers._chain_initializers', '_chain_initializers(initializer_and_args)', {'_ChainedInitializer': _ChainedInitializer, 'initializer_and_args': initializer_and_args}, 2)

def _prepare_initializer(initializer, initargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.initializers._prepare_initializer', '_prepare_initializer(initializer, initargs)', {'_chain_initializers': _chain_initializers, '_make_viztracer_initializer_and_initargs': _make_viztracer_initializer_and_initargs, 'initializer': initializer, 'initargs': initargs}, 1)

