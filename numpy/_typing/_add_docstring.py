"""A module for creating docstrings for sphinx ``data`` domains."""

import re
import textwrap
from ._generic_alias import NDArray
_docstrings_list = []

def add_newdoc(name: str, value: str, doc: str) -> None:
    """Append ``_docstrings_list`` with a docstring for `name`.

    Parameters
    ----------
    name : str
        The name of the object.
    value : str
        A string-representation of the object.
    doc : str
        The docstring of the object.

    """
    _docstrings_list.append((name, value, doc))

def _parse_docstrings() -> str:
    """Convert all docstrings in ``_docstrings_list`` into a single
    sphinx-legible text block.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy._typing._add_docstring._parse_docstrings', '_parse_docstrings()', {'_docstrings_list': _docstrings_list, 'textwrap': textwrap, 're': re}, 1)
add_newdoc('ArrayLike', 'typing.Union[...]', '\n    A `~typing.Union` representing objects that can be coerced\n    into an `~numpy.ndarray`.\n\n    Among others this includes the likes of:\n\n    * Scalars.\n    * (Nested) sequences.\n    * Objects implementing the `~class.__array__` protocol.\n\n    .. versionadded:: 1.20\n\n    See Also\n    --------\n    :term:`array_like`:\n        Any scalar or sequence that can be interpreted as an ndarray.\n\n    Examples\n    --------\n    .. code-block:: python\n\n        >>> import numpy as np\n        >>> import numpy.typing as npt\n\n        >>> def as_array(a: npt.ArrayLike) -> np.ndarray:\n        ...     return np.array(a)\n\n    ')
add_newdoc('DTypeLike', 'typing.Union[...]', '\n    A `~typing.Union` representing objects that can be coerced\n    into a `~numpy.dtype`.\n\n    Among others this includes the likes of:\n\n    * :class:`type` objects.\n    * Character codes or the names of :class:`type` objects.\n    * Objects with the ``.dtype`` attribute.\n\n    .. versionadded:: 1.20\n\n    See Also\n    --------\n    :ref:`Specifying and constructing data types <arrays.dtypes.constructing>`\n        A comprehensive overview of all objects that can be coerced\n        into data types.\n\n    Examples\n    --------\n    .. code-block:: python\n\n        >>> import numpy as np\n        >>> import numpy.typing as npt\n\n        >>> def as_dtype(d: npt.DTypeLike) -> np.dtype:\n        ...     return np.dtype(d)\n\n    ')
add_newdoc('NDArray', repr(NDArray), '\n    A :term:`generic <generic type>` version of\n    `np.ndarray[Any, np.dtype[+ScalarType]] <numpy.ndarray>`.\n\n    Can be used during runtime for typing arrays with a given dtype\n    and unspecified shape.\n\n    .. versionadded:: 1.21\n\n    Examples\n    --------\n    .. code-block:: python\n\n        >>> import numpy as np\n        >>> import numpy.typing as npt\n\n        >>> print(npt.NDArray)\n        numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]\n\n        >>> print(npt.NDArray[np.float64])\n        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]\n\n        >>> NDArrayInt = npt.NDArray[np.int_]\n        >>> a: NDArrayInt = np.arange(10)\n\n        >>> def func(a: npt.ArrayLike) -> npt.NDArray[Any]:\n        ...     return np.array(a)\n\n    ')
_docstrings = _parse_docstrings()

