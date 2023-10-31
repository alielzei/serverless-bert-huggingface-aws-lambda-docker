"""Handwritten parser of dependency specifiers.

The docstring for each __parse_* function contains ENBF-inspired grammar representing
the implementation.
"""

import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer


class Node:
    
    def __init__(self, value: str) -> None:
        self.value = value
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}('{self}')>"
    
    def serialize(self) -> str:
        raise NotImplementedError



class Variable(Node):
    
    def serialize(self) -> str:
        return str(self)



class Value(Node):
    
    def serialize(self) -> str:
        return f'"{self}"'



class Op(Node):
    
    def serialize(self) -> str:
        return str(self)

MarkerVar = Union[(Variable, Value)]
MarkerItem = Tuple[(MarkerVar, Op, MarkerVar)]
MarkerAtom = Any
MarkerList = List[Any]


class ParsedRequirement(NamedTuple):
    name: str
    url: str
    extras: List[str]
    specifier: str
    marker: Optional[MarkerList]


def parse_requirement(source: str) -> ParsedRequirement:
    return _parse_requirement(Tokenizer(source, rules=DEFAULT_RULES))

def _parse_requirement(tokenizer: Tokenizer) -> ParsedRequirement:
    """
    requirement = WS? IDENTIFIER WS? extras WS? requirement_details
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_requirement', '_parse_requirement(tokenizer)', {'_parse_extras': _parse_extras, '_parse_requirement_details': _parse_requirement_details, 'ParsedRequirement': ParsedRequirement, 'tokenizer': tokenizer}, 1)

def _parse_requirement_details(tokenizer: Tokenizer) -> Tuple[(str, str, Optional[MarkerList])]:
    """
    requirement_details = AT URL (WS requirement_marker?)?
                        | specifier WS? (requirement_marker)?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_requirement_details', '_parse_requirement_details(tokenizer)', {'_parse_requirement_marker': _parse_requirement_marker, '_parse_specifier': _parse_specifier, 'tokenizer': tokenizer, 'Tuple': Tuple}, 3)

def _parse_requirement_marker(tokenizer: Tokenizer, *, span_start: int, after: str) -> MarkerList:
    """
    requirement_marker = SEMICOLON marker WS?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_requirement_marker', '_parse_requirement_marker(tokenizer, span_start: int, after: str)', {'_parse_marker': _parse_marker, 'tokenizer': tokenizer, 'span_start': span_start, 'after': after}, 1)

def _parse_extras(tokenizer: Tokenizer) -> List[str]:
    """
    extras = (LEFT_BRACKET wsp* extras_list? wsp* RIGHT_BRACKET)?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_extras', '_parse_extras(tokenizer)', {'_parse_extras_list': _parse_extras_list, 'tokenizer': tokenizer, 'List': List, 'str': str}, 1)

def _parse_extras_list(tokenizer: Tokenizer) -> List[str]:
    """
    extras_list = identifier (wsp* ',' wsp* identifier)*
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_extras_list', '_parse_extras_list(tokenizer)', {'List': List, 'tokenizer': tokenizer, 'List': List, 'str': str}, 1)

def _parse_specifier(tokenizer: Tokenizer) -> str:
    """
    specifier = LEFT_PARENTHESIS WS? version_many WS? RIGHT_PARENTHESIS
              | WS? version_many WS?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_specifier', '_parse_specifier(tokenizer)', {'_parse_version_many': _parse_version_many, 'tokenizer': tokenizer}, 1)

def _parse_version_many(tokenizer: Tokenizer) -> str:
    """
    version_many = (SPECIFIER (WS? COMMA WS? SPECIFIER)*)?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_version_many', '_parse_version_many(tokenizer)', {'tokenizer': tokenizer}, 1)

def parse_marker(source: str) -> MarkerList:
    return _parse_full_marker(Tokenizer(source, rules=DEFAULT_RULES))

def _parse_full_marker(tokenizer: Tokenizer) -> MarkerList:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_full_marker', '_parse_full_marker(tokenizer)', {'_parse_marker': _parse_marker, 'tokenizer': tokenizer}, 1)

def _parse_marker(tokenizer: Tokenizer) -> MarkerList:
    """
    marker = marker_atom (BOOLOP marker_atom)+
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_marker', '_parse_marker(tokenizer)', {'_parse_marker_atom': _parse_marker_atom, 'tokenizer': tokenizer}, 1)

def _parse_marker_atom(tokenizer: Tokenizer) -> MarkerAtom:
    """
    marker_atom = WS? LEFT_PARENTHESIS WS? marker WS? RIGHT_PARENTHESIS WS?
                | WS? marker_item WS?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_marker_atom', '_parse_marker_atom(tokenizer)', {'MarkerAtom': MarkerAtom, '_parse_marker': _parse_marker, '_parse_marker_item': _parse_marker_item, 'tokenizer': tokenizer}, 1)

def _parse_marker_item(tokenizer: Tokenizer) -> MarkerItem:
    """
    marker_item = WS? marker_var WS? marker_op WS? marker_var WS?
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_marker_item', '_parse_marker_item(tokenizer)', {'_parse_marker_var': _parse_marker_var, '_parse_marker_op': _parse_marker_op, 'tokenizer': tokenizer}, 3)

def _parse_marker_var(tokenizer: Tokenizer) -> MarkerVar:
    """
    marker_var = VARIABLE | QUOTED_STRING
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_marker_var', '_parse_marker_var(tokenizer)', {'process_env_var': process_env_var, 'process_python_str': process_python_str, 'tokenizer': tokenizer}, 1)

def process_env_var(env_var: str) -> Variable:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser.process_env_var', 'process_env_var(env_var)', {'Variable': Variable, 'env_var': env_var}, 1)

def process_python_str(python_str: str) -> Value:
    value = ast.literal_eval(python_str)
    return Value(str(value))

def _parse_marker_op(tokenizer: Tokenizer) -> Op:
    """
    marker_op = IN | NOT IN | OP
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('packaging._parser._parse_marker_op', '_parse_marker_op(tokenizer)', {'Op': Op, 'tokenizer': tokenizer}, 1)

