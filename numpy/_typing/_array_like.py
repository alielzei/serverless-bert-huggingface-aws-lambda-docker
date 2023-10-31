from __future__ import annotations
import sys
from collections.abc import Collection, Callable
from typing import Any, Sequence, Protocol, Union, TypeVar, runtime_checkable
from numpy import ndarray, dtype, generic, bool_, unsignedinteger, integer, floating, complexfloating, number, timedelta64, datetime64, object_, void, str_, bytes_
from ._nested_sequence import _NestedSequence
_T = TypeVar('_T')
_ScalarType = TypeVar('_ScalarType', bound=generic)
_DType = TypeVar('_DType', bound='dtype[Any]')
_DType_co = TypeVar('_DType_co', covariant=True, bound='dtype[Any]')


@runtime_checkable
class _SupportsArray(Protocol[_DType_co]):
    
    def __array__(self) -> ndarray[(Any, _DType_co)]:
        ...



@runtime_checkable
class _SupportsArrayFunc(Protocol):
    """A protocol class representing `~class.__array_function__`."""
    
    def __array_function__(self, func: Callable[(..., Any)], types: Collection[type[Any]], args: tuple[(Any, ...)], kwargs: dict[(str, Any)]) -> object:
        ...

_FiniteNestedSequence = Union[(_T, Sequence[_T], Sequence[Sequence[_T]], Sequence[Sequence[Sequence[_T]]], Sequence[Sequence[Sequence[Sequence[_T]]]])]
_ArrayLike = Union[(_SupportsArray['dtype[_ScalarType]'], _NestedSequence[_SupportsArray['dtype[_ScalarType]']])]
_DualArrayLike = Union[(_SupportsArray[_DType], _NestedSequence[_SupportsArray[_DType]], _T, _NestedSequence[_T])]
if sys.version_info[:2] < (3, 9):
    ArrayLike = _DualArrayLike[(dtype, Union[(bool, int, float, complex, str, bytes)])]
else:
    ArrayLike = _DualArrayLike[(dtype[Any], Union[(bool, int, float, complex, str, bytes)])]
_ArrayLikeBool_co = _DualArrayLike[('dtype[bool_]', bool)]
_ArrayLikeUInt_co = _DualArrayLike[('dtype[Union[bool_, unsignedinteger[Any]]]', bool)]
_ArrayLikeInt_co = _DualArrayLike[('dtype[Union[bool_, integer[Any]]]', Union[(bool, int)])]
_ArrayLikeFloat_co = _DualArrayLike[('dtype[Union[bool_, integer[Any], floating[Any]]]', Union[(bool, int, float)])]
_ArrayLikeComplex_co = _DualArrayLike[('dtype[Union[bool_, integer[Any], floating[Any], complexfloating[Any, Any]]]', Union[(bool, int, float, complex)])]
_ArrayLikeNumber_co = _DualArrayLike[('dtype[Union[bool_, number[Any]]]', Union[(bool, int, float, complex)])]
_ArrayLikeTD64_co = _DualArrayLike[('dtype[Union[bool_, integer[Any], timedelta64]]', Union[(bool, int)])]
_ArrayLikeDT64_co = Union[(_SupportsArray['dtype[datetime64]'], _NestedSequence[_SupportsArray['dtype[datetime64]']])]
_ArrayLikeObject_co = Union[(_SupportsArray['dtype[object_]'], _NestedSequence[_SupportsArray['dtype[object_]']])]
_ArrayLikeVoid_co = Union[(_SupportsArray['dtype[void]'], _NestedSequence[_SupportsArray['dtype[void]']])]
_ArrayLikeStr_co = _DualArrayLike[('dtype[str_]', str)]
_ArrayLikeBytes_co = _DualArrayLike[('dtype[bytes_]', bytes)]
_ArrayLikeInt = _DualArrayLike[('dtype[integer[Any]]', int)]


class _UnknownType:
    ...

_ArrayLikeUnknown = _DualArrayLike[('dtype[_UnknownType]', _UnknownType)]

