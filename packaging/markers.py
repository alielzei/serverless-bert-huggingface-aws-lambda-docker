import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ._parser import MarkerAtom, MarkerList, Op, Value, Variable, parse_marker as _parse_marker
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name
__all__ = ['InvalidMarker', 'UndefinedComparison', 'UndefinedEnvironmentName', 'Marker', 'default_environment']
Operator = Callable[([str, str], bool)]


class InvalidMarker(ValueError):
    """
    An invalid marker was found, users should refer to PEP 508.
    """
    



class UndefinedComparison(ValueError):
    """
    An invalid operation was attempted on a value that doesn't support it.
    """
    



class UndefinedEnvironmentName(ValueError):
    """
    A name was attempted to be used that does not exist inside of the
    environment.
    """
    


def _normalize_extra_values(results: Any) -> Any:
    """
    Normalize extra values.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.markers._normalize_extra_values', '_normalize_extra_values(results)', {'Variable': Variable, 'canonicalize_name': canonicalize_name, 'Value': Value, 'results': results}, 1)

def _format_marker(marker: Union[(List[str], MarkerAtom, str)], first: Optional[bool] = True) -> str:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.markers._format_marker', '_format_marker(marker, first=True)', {'_format_marker': _format_marker, 'marker': marker, 'first': first, 'Union': Union, 'Optional': Optional, 'bool': bool}, 1)
_operators: Dict[(str, Operator)] = {'in': lambda lhs, rhs: lhs in rhs, 'not in': lambda lhs, rhs: lhs not in rhs, '<': operator.lt, '<=': operator.le, '==': operator.eq, '!=': operator.ne, '>=': operator.ge, '>': operator.gt}

def _eval_op(lhs: str, op: Op, rhs: str) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.markers._eval_op', '_eval_op(lhs, op, rhs)', {'Specifier': Specifier, 'InvalidSpecifier': InvalidSpecifier, 'Optional': Optional, 'Operator': Operator, '_operators': _operators, 'UndefinedComparison': UndefinedComparison, 'lhs': lhs, 'op': op, 'rhs': rhs}, 1)

def _normalize(*values, key: str) -> Tuple[(str, ...)]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.markers._normalize', '_normalize(*values, key: str)', {'canonicalize_name': canonicalize_name, 'key': key, 'values': values, 'Tuple': Tuple}, 1)

def _evaluate_markers(markers: MarkerList, environment: Dict[(str, str)]) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.markers._evaluate_markers', '_evaluate_markers(markers, environment)', {'List': List, '_evaluate_markers': _evaluate_markers, 'Variable': Variable, '_normalize': _normalize, '_eval_op': _eval_op, 'markers': markers, 'environment': environment, 'Dict': Dict}, 1)

def format_full_version(info: 'sys._version_info') -> str:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.markers.format_full_version', 'format_full_version(info)', {'info': info}, 1)

def default_environment() -> Dict[(str, str)]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging.markers.default_environment', 'default_environment()', {'format_full_version': format_full_version, 'sys': sys, 'os': os, 'platform': platform, 'Dict': Dict}, 1)


class Marker:
    
    def __init__(self, marker: str) -> None:
        try:
            self._markers = _normalize_extra_values(_parse_marker(marker))
        except ParserSyntaxError as e:
            raise InvalidMarker(str(e)) from e
    
    def __str__(self) -> str:
        return _format_marker(self._markers)
    
    def __repr__(self) -> str:
        return f"<Marker('{self}')>"
    
    def __hash__(self) -> int:
        return hash((self.__class__.__name__, str(self)))
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Marker):
            return NotImplemented
        return str(self) == str(other)
    
    def evaluate(self, environment: Optional[Dict[(str, str)]] = None) -> bool:
        """Evaluate a marker.

        Return the boolean from evaluating the given marker against the
        environment. environment is an optional argument to override all or
        part of the determined environment.

        The environment is determined from the current Python process.
        """
        current_environment = default_environment()
        current_environment['extra'] = ''
        if environment is not None:
            current_environment.update(environment)
            if current_environment['extra'] is None:
                current_environment['extra'] = ''
        return _evaluate_markers(self._markers, current_environment)


