from typing import Any, List, Sequence, Tuple, Union, Type, TypeVar, Protocol, TypedDict, runtime_checkable
import numpy as np
from ._shape import _ShapeLike
from ._generic_alias import _DType as DType
from ._char_codes import _BoolCodes, _UInt8Codes, _UInt16Codes, _UInt32Codes, _UInt64Codes, _Int8Codes, _Int16Codes, _Int32Codes, _Int64Codes, _Float16Codes, _Float32Codes, _Float64Codes, _Complex64Codes, _Complex128Codes, _ByteCodes, _ShortCodes, _IntCCodes, _IntPCodes, _IntCodes, _LongLongCodes, _UByteCodes, _UShortCodes, _UIntCCodes, _UIntPCodes, _UIntCodes, _ULongLongCodes, _HalfCodes, _SingleCodes, _DoubleCodes, _LongDoubleCodes, _CSingleCodes, _CDoubleCodes, _CLongDoubleCodes, _DT64Codes, _TD64Codes, _StrCodes, _BytesCodes, _VoidCodes, _ObjectCodes
_SCT = TypeVar('_SCT', bound=np.generic)
_DType_co = TypeVar('_DType_co', covariant=True, bound=DType[Any])
_DTypeLikeNested = Any


class _DTypeDictBase(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DTypeLikeNested]



class _DTypeDict(_DTypeDictBase, total=False):
    offsets: Sequence[int]
    titles: Sequence[Any]
    itemsize: int
    aligned: bool



@runtime_checkable
class _SupportsDType(Protocol[_DType_co]):
    
    @property
    def dtype(self) -> _DType_co:
        ...

_DTypeLike = Union[('np.dtype[_SCT]', Type[_SCT], _SupportsDType['np.dtype[_SCT]'])]
_VoidDTypeLike = Union[(Tuple[(_DTypeLikeNested, int)], Tuple[(_DTypeLikeNested, _ShapeLike)], List[Any], _DTypeDict, Tuple[(_DTypeLikeNested, _DTypeLikeNested)])]
DTypeLike = Union[(DType[Any], None, Type[Any], _SupportsDType[DType[Any]], str, _VoidDTypeLike)]
_DTypeLikeBool = Union[(Type[bool], Type[np.bool_], DType[np.bool_], _SupportsDType[DType[np.bool_]], _BoolCodes)]
_DTypeLikeUInt = Union[(Type[np.unsignedinteger], DType[np.unsignedinteger], _SupportsDType[DType[np.unsignedinteger]], _UInt8Codes, _UInt16Codes, _UInt32Codes, _UInt64Codes, _UByteCodes, _UShortCodes, _UIntCCodes, _UIntPCodes, _UIntCodes, _ULongLongCodes)]
_DTypeLikeInt = Union[(Type[int], Type[np.signedinteger], DType[np.signedinteger], _SupportsDType[DType[np.signedinteger]], _Int8Codes, _Int16Codes, _Int32Codes, _Int64Codes, _ByteCodes, _ShortCodes, _IntCCodes, _IntPCodes, _IntCodes, _LongLongCodes)]
_DTypeLikeFloat = Union[(Type[float], Type[np.floating], DType[np.floating], _SupportsDType[DType[np.floating]], _Float16Codes, _Float32Codes, _Float64Codes, _HalfCodes, _SingleCodes, _DoubleCodes, _LongDoubleCodes)]
_DTypeLikeComplex = Union[(Type[complex], Type[np.complexfloating], DType[np.complexfloating], _SupportsDType[DType[np.complexfloating]], _Complex64Codes, _Complex128Codes, _CSingleCodes, _CDoubleCodes, _CLongDoubleCodes)]
_DTypeLikeDT64 = Union[(Type[np.timedelta64], DType[np.timedelta64], _SupportsDType[DType[np.timedelta64]], _TD64Codes)]
_DTypeLikeTD64 = Union[(Type[np.datetime64], DType[np.datetime64], _SupportsDType[DType[np.datetime64]], _DT64Codes)]
_DTypeLikeStr = Union[(Type[str], Type[np.str_], DType[np.str_], _SupportsDType[DType[np.str_]], _StrCodes)]
_DTypeLikeBytes = Union[(Type[bytes], Type[np.bytes_], DType[np.bytes_], _SupportsDType[DType[np.bytes_]], _BytesCodes)]
_DTypeLikeVoid = Union[(Type[np.void], DType[np.void], _SupportsDType[DType[np.void]], _VoidCodes, _VoidDTypeLike)]
_DTypeLikeObject = Union[(type, DType[np.object_], _SupportsDType[DType[np.object_]], _ObjectCodes)]
_DTypeLikeComplex_co = Union[(_DTypeLikeBool, _DTypeLikeUInt, _DTypeLikeInt, _DTypeLikeFloat, _DTypeLikeComplex)]

