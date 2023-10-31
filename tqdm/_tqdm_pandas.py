import sys
__author__ = 'github.com/casperdcl'
__all__ = ['tqdm_pandas']

def tqdm_pandas(tclass, **tqdm_kwargs):
    """
    Registers the given `tqdm` instance with
    `pandas.core.groupby.DataFrameGroupBy.progress_apply`.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('tqdm._tqdm_pandas.tqdm_pandas', 'tqdm_pandas(tclass, **tqdm_kwargs)', {'sys': sys, 'tclass': tclass, 'tqdm_kwargs': tqdm_kwargs}, 0)

