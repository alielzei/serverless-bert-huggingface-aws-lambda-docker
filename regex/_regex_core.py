import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
__all__ = ['A', 'ASCII', 'B', 'BESTMATCH', 'D', 'DEBUG', 'E', 'ENHANCEMATCH', 'F', 'FULLCASE', 'I', 'IGNORECASE', 'L', 'LOCALE', 'M', 'MULTILINE', 'P', 'POSIX', 'R', 'REVERSE', 'S', 'DOTALL', 'T', 'TEMPLATE', 'U', 'UNICODE', 'V0', 'VERSION0', 'V1', 'VERSION1', 'W', 'WORD', 'X', 'VERBOSE', 'error', 'Scanner', 'RegexFlag']


class error(Exception):
    """Exception raised for invalid regular expressions.

    Attributes:

        msg: The unformatted error message
        pattern: The regular expression pattern
        pos: The position in the pattern where compilation failed, or None
        lineno: The line number where compilation failed, unless pos is None
        colno: The column number where compilation failed, unless pos is None
    """
    
    def __init__(self, message, pattern=None, pos=None):
        newline = ('\n' if isinstance(pattern, str) else b'\n')
        self.msg = message
        self.pattern = pattern
        self.pos = pos
        if (pattern is not None and pos is not None):
            self.lineno = pattern.count(newline, 0, pos) + 1
            self.colno = pos - pattern.rfind(newline, 0, pos)
            message = '{} at position {}'.format(message, pos)
            if newline in pattern:
                message += ' (line {}, column {})'.format(self.lineno, self.colno)
        Exception.__init__(self, message)



class _UnscopedFlagSet(Exception):
    pass



class ParseError(Exception):
    pass



class _FirstSetError(Exception):
    pass



class RegexFlag(enum.IntFlag):
    A = ASCII = 128
    B = BESTMATCH = 4096
    D = DEBUG = 512
    E = ENHANCEMATCH = 32768
    F = FULLCASE = 16384
    I = IGNORECASE = 2
    L = LOCALE = 4
    M = MULTILINE = 8
    P = POSIX = 65536
    R = REVERSE = 1024
    S = DOTALL = 16
    U = UNICODE = 32
    V0 = VERSION0 = 8192
    V1 = VERSION1 = 256
    W = WORD = 2048
    X = VERBOSE = 64
    T = TEMPLATE = 1
    
    def __repr__(self):
        if self._name_ is not None:
            return 'regex.%s' % self._name_
        value = self._value_
        members = []
        negative = value < 0
        if negative:
            value = ~value
        for m in self.__class__:
            if value & m._value_:
                value &= ~m._value_
                members.append('regex.%s' % m._name_)
        if value:
            members.append(hex(value))
        res = '|'.join(members)
        if negative:
            if len(members) > 1:
                res = '~(%s)' % res
            else:
                res = '~%s' % res
        return res
    __str__ = object.__str__

globals().update(RegexFlag.__members__)
DEFAULT_VERSION = VERSION1
_ALL_VERSIONS = VERSION0 | VERSION1
_ALL_ENCODINGS = ASCII | LOCALE | UNICODE
DEFAULT_FLAGS = {VERSION0: 0, VERSION1: FULLCASE}
GLOBAL_FLAGS = _ALL_VERSIONS | BESTMATCH | DEBUG | ENHANCEMATCH | POSIX | REVERSE
SCOPED_FLAGS = FULLCASE | IGNORECASE | MULTILINE | DOTALL | WORD | VERBOSE | _ALL_ENCODINGS
ALPHA = frozenset(string.ascii_letters)
DIGITS = frozenset(string.digits)
ALNUM = ALPHA | DIGITS
OCT_DIGITS = frozenset(string.octdigits)
HEX_DIGITS = frozenset(string.hexdigits)
SPECIAL_CHARS = frozenset('()|?*+{^$.[\\#') | frozenset([''])
NAMED_CHAR_PART = ALNUM | frozenset(' -')
PROPERTY_NAME_PART = ALNUM | frozenset(' &_-.')
SET_OPS = ('||', '~~', '&&', '--')
BYTES_PER_CODE = _regex.get_code_size()
BITS_PER_CODE = BYTES_PER_CODE * 8
UNLIMITED = (1 << BITS_PER_CODE) - 1
REGEX_FLAGS = {'a': ASCII, 'b': BESTMATCH, 'e': ENHANCEMATCH, 'f': FULLCASE, 'i': IGNORECASE, 'L': LOCALE, 'm': MULTILINE, 'p': POSIX, 'r': REVERSE, 's': DOTALL, 'u': UNICODE, 'V0': VERSION0, 'V1': VERSION1, 'w': WORD, 'x': VERBOSE}
CASE_FLAGS = FULLCASE | IGNORECASE
NOCASE = 0
FULLIGNORECASE = FULLCASE | IGNORECASE
FULL_CASE_FOLDING = UNICODE | FULLIGNORECASE
CASE_FLAGS_COMBINATIONS = {0: 0, FULLCASE: 0, IGNORECASE: IGNORECASE, FULLIGNORECASE: FULLIGNORECASE}
HEX_ESCAPES = {'x': 2, 'u': 4, 'U': 8}
OPCODES = '\nFAILURE\nSUCCESS\nANY\nANY_ALL\nANY_ALL_REV\nANY_REV\nANY_U\nANY_U_REV\nATOMIC\nBOUNDARY\nBRANCH\nCALL_REF\nCHARACTER\nCHARACTER_IGN\nCHARACTER_IGN_REV\nCHARACTER_REV\nCONDITIONAL\nDEFAULT_BOUNDARY\nDEFAULT_END_OF_WORD\nDEFAULT_START_OF_WORD\nEND\nEND_OF_LINE\nEND_OF_LINE_U\nEND_OF_STRING\nEND_OF_STRING_LINE\nEND_OF_STRING_LINE_U\nEND_OF_WORD\nFUZZY\nGRAPHEME_BOUNDARY\nGREEDY_REPEAT\nGROUP\nGROUP_CALL\nGROUP_EXISTS\nKEEP\nLAZY_REPEAT\nLOOKAROUND\nNEXT\nPROPERTY\nPROPERTY_IGN\nPROPERTY_IGN_REV\nPROPERTY_REV\nPRUNE\nRANGE\nRANGE_IGN\nRANGE_IGN_REV\nRANGE_REV\nREF_GROUP\nREF_GROUP_FLD\nREF_GROUP_FLD_REV\nREF_GROUP_IGN\nREF_GROUP_IGN_REV\nREF_GROUP_REV\nSEARCH_ANCHOR\nSET_DIFF\nSET_DIFF_IGN\nSET_DIFF_IGN_REV\nSET_DIFF_REV\nSET_INTER\nSET_INTER_IGN\nSET_INTER_IGN_REV\nSET_INTER_REV\nSET_SYM_DIFF\nSET_SYM_DIFF_IGN\nSET_SYM_DIFF_IGN_REV\nSET_SYM_DIFF_REV\nSET_UNION\nSET_UNION_IGN\nSET_UNION_IGN_REV\nSET_UNION_REV\nSKIP\nSTART_OF_LINE\nSTART_OF_LINE_U\nSTART_OF_STRING\nSTART_OF_WORD\nSTRING\nSTRING_FLD\nSTRING_FLD_REV\nSTRING_IGN\nSTRING_IGN_REV\nSTRING_REV\nFUZZY_EXT\n'


class Namespace:
    pass

OP = Namespace()
for (i, op) in enumerate(OPCODES.split()):
    setattr(OP, op, i)

def _shrink_cache(cache_dict, args_dict, locale_sensitive, max_length, divisor=5):
    """Make room in the given cache.

    Args:
        cache_dict: The cache dictionary to modify.
        args_dict: The dictionary of named list args used by patterns.
        max_length: Maximum # of entries in cache_dict before it is shrunk.
        divisor: Cache will shrink to max_length - 1/divisor*max_length items.
    """
    cache_keys = tuple(cache_dict.keys())
    overage = len(cache_keys) - max_length
    if overage < 0:
        return
    number_to_toss = max_length // divisor + overage
    import random
    if not hasattr(random, 'sample'):
        return
    for doomed_key in random.sample(cache_keys, number_to_toss):
        try:
            del cache_dict[doomed_key]
        except KeyError:
            pass
    args_dict.clear()
    sensitivity_dict = {}
    for (pattern, pattern_type, flags, args, default_version, locale) in tuple(cache_dict):
        args_dict[(pattern, pattern_type, flags, default_version, locale)] = args
        try:
            sensitivity_dict[(pattern_type, pattern)] = locale_sensitive[(pattern_type, pattern)]
        except KeyError:
            pass
    locale_sensitive.clear()
    locale_sensitive.update(sensitivity_dict)

def _fold_case(info, string):
    """Folds the case of a string."""
    flags = info.flags
    if flags & _ALL_ENCODINGS == 0:
        flags |= info.guess_encoding
    return _regex.fold_case(flags, string)

def is_cased_i(info, char):
    """Checks whether a character is cased."""
    return len(_regex.get_all_cases(info.flags, char)) > 1

def is_cased_f(flags, char):
    """Checks whether a character is cased."""
    return len(_regex.get_all_cases(flags, char)) > 1

def _compile_firstset(info, fs):
    """Compiles the firstset for the pattern."""
    reverse = bool(info.flags & REVERSE)
    fs = _check_firstset(info, reverse, fs)
    if not fs:
        return []
    return fs.compile(reverse)

def _check_firstset(info, reverse, fs):
    """Checks the firstset for the pattern."""
    if (not fs or None in fs):
        return None
    members = set()
    case_flags = NOCASE
    for i in fs:
        if (isinstance(i, Character) and not i.positive):
            return None
        case_flags |= i.case_flags
        members.add(i.with_flags(case_flags=NOCASE))
    if case_flags == FULLCASE | IGNORECASE:
        return None
    fs = SetUnion(info, list(members), case_flags=case_flags & ~FULLCASE, zerowidth=True)
    fs = fs.optimise(info, reverse, in_set=True)
    return fs

def _flatten_code(code):
    """Flattens the code from a list of tuples."""
    flat_code = []
    for c in code:
        flat_code.extend(c)
    return flat_code

def make_case_flags(info):
    """Makes the case flags."""
    flags = info.flags & CASE_FLAGS
    if info.flags & ASCII:
        flags &= ~FULLCASE
    return flags

def make_character(info, value, in_set=False):
    """Makes a character literal."""
    if in_set:
        return Character(value)
    return Character(value, case_flags=make_case_flags(info))

def make_ref_group(info, name, position):
    """Makes a group reference."""
    return RefGroup(info, name, position, case_flags=make_case_flags(info))

def make_string_set(info, name):
    """Makes a string set."""
    return StringSet(info, name, case_flags=make_case_flags(info))

def make_property(info, prop, in_set):
    """Makes a property."""
    if in_set:
        return prop
    return prop.with_flags(case_flags=make_case_flags(info))

def _parse_pattern(source, info):
    """Parses a pattern, eg. 'a|b|c'."""
    branches = [parse_sequence(source, info)]
    while source.match('|'):
        branches.append(parse_sequence(source, info))
    if len(branches) == 1:
        return branches[0]
    return Branch(branches)

def parse_sequence(source, info):
    """Parses a sequence, eg. 'abc'."""
    sequence = [None]
    case_flags = make_case_flags(info)
    while True:
        saved_pos = source.pos
        ch = source.get()
        if ch in SPECIAL_CHARS:
            if ch in ')|':
                source.pos = saved_pos
                break
            elif ch == '\\':
                sequence.append(parse_escape(source, info, False))
            elif ch == '(':
                element = parse_paren(source, info)
                if element is None:
                    case_flags = make_case_flags(info)
                else:
                    sequence.append(element)
            elif ch == '.':
                if info.flags & DOTALL:
                    sequence.append(AnyAll())
                elif info.flags & WORD:
                    sequence.append(AnyU())
                else:
                    sequence.append(Any())
            elif ch == '[':
                sequence.append(parse_set(source, info))
            elif ch == '^':
                if info.flags & MULTILINE:
                    if info.flags & WORD:
                        sequence.append(StartOfLineU())
                    else:
                        sequence.append(StartOfLine())
                else:
                    sequence.append(StartOfString())
            elif ch == '$':
                if info.flags & MULTILINE:
                    if info.flags & WORD:
                        sequence.append(EndOfLineU())
                    else:
                        sequence.append(EndOfLine())
                elif info.flags & WORD:
                    sequence.append(EndOfStringLineU())
                else:
                    sequence.append(EndOfStringLine())
            elif ch in '?*+{':
                counts = parse_quantifier(source, info, ch)
                if counts:
                    apply_quantifier(source, info, counts, case_flags, ch, saved_pos, sequence)
                    sequence.append(None)
                else:
                    constraints = parse_fuzzy(source, info, ch, case_flags)
                    if constraints:
                        apply_constraint(source, info, constraints, case_flags, saved_pos, sequence)
                        sequence.append(None)
                    else:
                        sequence.append(Character(ord(ch), case_flags=case_flags))
            else:
                sequence.append(Character(ord(ch), case_flags=case_flags))
        else:
            sequence.append(Character(ord(ch), case_flags=case_flags))
    sequence = [item for item in sequence if item is not None]
    return Sequence(sequence)

def apply_quantifier(source, info, counts, case_flags, ch, saved_pos, sequence):
    element = sequence.pop()
    if element is None:
        if sequence:
            raise error('multiple repeat', source.string, saved_pos)
        raise error('nothing to repeat', source.string, saved_pos)
    if isinstance(element, (GreedyRepeat, LazyRepeat, PossessiveRepeat)):
        raise error('multiple repeat', source.string, saved_pos)
    (min_count, max_count) = counts
    saved_pos = source.pos
    ch = source.get()
    if ch == '?':
        repeated = LazyRepeat
    elif ch == '+':
        repeated = PossessiveRepeat
    else:
        source.pos = saved_pos
        repeated = GreedyRepeat
    if (not element.is_empty() and ((min_count != 1 or max_count != 1))):
        element = repeated(element, min_count, max_count)
    sequence.append(element)

def apply_constraint(source, info, constraints, case_flags, saved_pos, sequence):
    element = sequence.pop()
    if element is None:
        raise error('nothing for fuzzy constraint', source.string, saved_pos)
    if isinstance(element, Group):
        element.subpattern = Fuzzy(element.subpattern, constraints)
        sequence.append(element)
    else:
        sequence.append(Fuzzy(element, constraints))
_QUANTIFIERS = {'?': (0, 1), '*': (0, None), '+': (1, None)}

def parse_quantifier(source, info, ch):
    """Parses a quantifier."""
    q = _QUANTIFIERS.get(ch)
    if q:
        return q
    if ch == '{':
        counts = parse_limited_quantifier(source)
        if counts:
            return counts
    return None

def is_above_limit(count):
    """Checks whether a count is above the maximum."""
    return (count is not None and count >= UNLIMITED)

def parse_limited_quantifier(source):
    """Parses a limited quantifier."""
    saved_pos = source.pos
    min_count = parse_count(source)
    if source.match(','):
        max_count = parse_count(source)
        min_count = int((min_count or 0))
        max_count = (int(max_count) if max_count else None)
    else:
        if not min_count:
            source.pos = saved_pos
            return None
        min_count = max_count = int(min_count)
    if not source.match('}'):
        source.pos = saved_pos
        return None
    if (is_above_limit(min_count) or is_above_limit(max_count)):
        raise error('repeat count too big', source.string, saved_pos)
    if (max_count is not None and min_count > max_count):
        raise error('min repeat greater than max repeat', source.string, saved_pos)
    return (min_count, max_count)

def parse_fuzzy(source, info, ch, case_flags):
    """Parses a fuzzy setting, if present."""
    saved_pos = source.pos
    if ch != '{':
        return None
    constraints = {}
    try:
        parse_fuzzy_item(source, constraints)
        while source.match(','):
            parse_fuzzy_item(source, constraints)
    except ParseError:
        source.pos = saved_pos
        return None
    if source.match(':'):
        constraints['test'] = parse_fuzzy_test(source, info, case_flags)
    if not source.match('}'):
        raise error('expected }', source.string, source.pos)
    return constraints

def parse_fuzzy_item(source, constraints):
    """Parses a fuzzy setting item."""
    saved_pos = source.pos
    try:
        parse_cost_constraint(source, constraints)
    except ParseError:
        source.pos = saved_pos
        parse_cost_equation(source, constraints)

def parse_cost_constraint(source, constraints):
    """Parses a cost constraint."""
    saved_pos = source.pos
    ch = source.get()
    if ch in ALPHA:
        constraint = parse_constraint(source, constraints, ch)
        max_inc = parse_fuzzy_compare(source)
        if max_inc is None:
            constraints[constraint] = (0, None)
        else:
            cost_pos = source.pos
            max_cost = parse_cost_limit(source)
            if not max_inc:
                max_cost -= 1
            if max_cost < 0:
                raise error('bad fuzzy cost limit', source.string, cost_pos)
            constraints[constraint] = (0, max_cost)
    elif ch in DIGITS:
        source.pos = saved_pos
        cost_pos = source.pos
        min_cost = parse_cost_limit(source)
        min_inc = parse_fuzzy_compare(source)
        if min_inc is None:
            raise ParseError()
        constraint = parse_constraint(source, constraints, source.get())
        max_inc = parse_fuzzy_compare(source)
        if max_inc is None:
            raise ParseError()
        cost_pos = source.pos
        max_cost = parse_cost_limit(source)
        if not min_inc:
            min_cost += 1
        if not max_inc:
            max_cost -= 1
        if not 0 <= min_cost <= max_cost:
            raise error('bad fuzzy cost limit', source.string, cost_pos)
        constraints[constraint] = (min_cost, max_cost)
    else:
        raise ParseError()

def parse_cost_limit(source):
    """Parses a cost limit."""
    cost_pos = source.pos
    digits = parse_count(source)
    try:
        return int(digits)
    except ValueError:
        pass
    raise error('bad fuzzy cost limit', source.string, cost_pos)

def parse_constraint(source, constraints, ch):
    """Parses a constraint."""
    if ch not in 'deis':
        raise ParseError()
    if ch in constraints:
        raise ParseError()
    return ch

def parse_fuzzy_compare(source):
    """Parses a cost comparator."""
    if source.match('<='):
        return True
    elif source.match('<'):
        return False
    else:
        return None

def parse_cost_equation(source, constraints):
    """Parses a cost equation."""
    if 'cost' in constraints:
        raise error('more than one cost equation', source.string, source.pos)
    cost = {}
    parse_cost_term(source, cost)
    while source.match('+'):
        parse_cost_term(source, cost)
    max_inc = parse_fuzzy_compare(source)
    if max_inc is None:
        raise ParseError()
    max_cost = int(parse_count(source))
    if not max_inc:
        max_cost -= 1
    if max_cost < 0:
        raise error('bad fuzzy cost limit', source.string, source.pos)
    cost['max'] = max_cost
    constraints['cost'] = cost

def parse_cost_term(source, cost):
    """Parses a cost equation term."""
    coeff = parse_count(source)
    ch = source.get()
    if ch not in 'dis':
        raise ParseError()
    if ch in cost:
        raise error('repeated fuzzy cost', source.string, source.pos)
    cost[ch] = int((coeff or 1))

def parse_fuzzy_test(source, info, case_flags):
    saved_pos = source.pos
    ch = source.get()
    if ch in SPECIAL_CHARS:
        if ch == '\\':
            return parse_escape(source, info, False)
        elif ch == '.':
            if info.flags & DOTALL:
                return AnyAll()
            elif info.flags & WORD:
                return AnyU()
            else:
                return Any()
        elif ch == '[':
            return parse_set(source, info)
        else:
            raise error('expected character set', source.string, saved_pos)
    elif ch:
        return Character(ord(ch), case_flags=case_flags)
    else:
        raise error('expected character set', source.string, saved_pos)

def parse_count(source):
    """Parses a quantifier's count, which can be empty."""
    return source.get_while(DIGITS)

def parse_paren(source, info):
    """Parses a parenthesised subpattern or a flag. Returns FLAGS if it's an
    inline flag.
    """
    saved_pos = source.pos
    ch = source.get(True)
    if ch == '?':
        saved_pos_2 = source.pos
        ch = source.get(True)
        if ch == '<':
            saved_pos_3 = source.pos
            ch = source.get()
            if ch in ('=', '!'):
                return parse_lookaround(source, info, True, ch == '=')
            source.pos = saved_pos_3
            name = parse_name(source)
            group = info.open_group(name)
            source.expect('>')
            saved_flags = info.flags
            try:
                subpattern = _parse_pattern(source, info)
                source.expect(')')
            finally:
                info.flags = saved_flags
                source.ignore_space = bool(info.flags & VERBOSE)
            info.close_group()
            return Group(info, group, subpattern)
        if ch in ('=', '!'):
            return parse_lookaround(source, info, False, ch == '=')
        if ch == 'P':
            return parse_extension(source, info)
        if ch == '#':
            return parse_comment(source)
        if ch == '(':
            return parse_conditional(source, info)
        if ch == '>':
            return parse_atomic(source, info)
        if ch == '|':
            return parse_common(source, info)
        if (ch == 'R' or '0' <= ch <= '9'):
            return parse_call_group(source, info, ch, saved_pos_2)
        if ch == '&':
            return parse_call_named_group(source, info, saved_pos_2)
        source.pos = saved_pos_2
        return parse_flags_subpattern(source, info)
    if ch == '*':
        saved_pos_2 = source.pos
        word = source.get_while(set(')>'), include=False)
        if word[:1].isalpha():
            verb = VERBS.get(word)
            if not verb:
                raise error('unknown verb', source.string, saved_pos_2)
            source.expect(')')
            return verb
    source.pos = saved_pos
    group = info.open_group()
    saved_flags = info.flags
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    info.close_group()
    return Group(info, group, subpattern)

def parse_extension(source, info):
    """Parses a Python extension."""
    saved_pos = source.pos
    ch = source.get()
    if ch == '<':
        name = parse_name(source)
        group = info.open_group(name)
        source.expect('>')
        saved_flags = info.flags
        try:
            subpattern = _parse_pattern(source, info)
            source.expect(')')
        finally:
            info.flags = saved_flags
            source.ignore_space = bool(info.flags & VERBOSE)
        info.close_group()
        return Group(info, group, subpattern)
    if ch == '=':
        name = parse_name(source, allow_numeric=True)
        source.expect(')')
        if info.is_open_group(name):
            raise error('cannot refer to an open group', source.string, saved_pos)
        return make_ref_group(info, name, saved_pos)
    if (ch == '>' or ch == '&'):
        return parse_call_named_group(source, info, saved_pos)
    source.pos = saved_pos
    raise error('unknown extension', source.string, saved_pos)

def parse_comment(source):
    """Parses a comment."""
    while True:
        saved_pos = source.pos
        c = source.get(True)
        if (not c or c == ')'):
            break
        if c == '\\':
            c = source.get(True)
    source.pos = saved_pos
    source.expect(')')
    return None

def parse_lookaround(source, info, behind, positive):
    """Parses a lookaround."""
    saved_flags = info.flags
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    return LookAround(behind, positive, subpattern)

def parse_conditional(source, info):
    """Parses a conditional subpattern."""
    saved_flags = info.flags
    saved_pos = source.pos
    ch = source.get()
    if ch == '?':
        ch = source.get()
        if ch in ('=', '!'):
            return parse_lookaround_conditional(source, info, False, ch == '=')
        if ch == '<':
            ch = source.get()
            if ch in ('=', '!'):
                return parse_lookaround_conditional(source, info, True, ch == '=')
        source.pos = saved_pos
        raise error('expected lookaround conditional', source.string, source.pos)
    source.pos = saved_pos
    try:
        group = parse_name(source, True)
        source.expect(')')
        yes_branch = parse_sequence(source, info)
        if source.match('|'):
            no_branch = parse_sequence(source, info)
        else:
            no_branch = Sequence()
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    if (yes_branch.is_empty() and no_branch.is_empty()):
        return Sequence()
    return Conditional(info, group, yes_branch, no_branch, saved_pos)

def parse_lookaround_conditional(source, info, behind, positive):
    saved_flags = info.flags
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    yes_branch = parse_sequence(source, info)
    if source.match('|'):
        no_branch = parse_sequence(source, info)
    else:
        no_branch = Sequence()
    source.expect(')')
    return LookAroundConditional(behind, positive, subpattern, yes_branch, no_branch)

def parse_atomic(source, info):
    """Parses an atomic subpattern."""
    saved_flags = info.flags
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    return Atomic(subpattern)

def parse_common(source, info):
    """Parses a common groups branch."""
    initial_group_count = info.group_count
    branches = [parse_sequence(source, info)]
    final_group_count = info.group_count
    while source.match('|'):
        info.group_count = initial_group_count
        branches.append(parse_sequence(source, info))
        final_group_count = max(final_group_count, info.group_count)
    info.group_count = final_group_count
    source.expect(')')
    if len(branches) == 1:
        return branches[0]
    return Branch(branches)

def parse_call_group(source, info, ch, pos):
    """Parses a call to a group."""
    if ch == 'R':
        group = '0'
    else:
        group = ch + source.get_while(DIGITS)
    source.expect(')')
    return CallGroup(info, group, pos)

def parse_call_named_group(source, info, pos):
    """Parses a call to a named group."""
    group = parse_name(source)
    source.expect(')')
    return CallGroup(info, group, pos)

def parse_flag_set(source):
    """Parses a set of inline flags."""
    flags = 0
    try:
        while True:
            saved_pos = source.pos
            ch = source.get()
            if ch == 'V':
                ch += source.get()
            flags |= REGEX_FLAGS[ch]
    except KeyError:
        source.pos = saved_pos
    return flags

def parse_flags(source, info):
    """Parses flags being turned on/off."""
    flags_on = parse_flag_set(source)
    if source.match('-'):
        flags_off = parse_flag_set(source)
        if not flags_off:
            raise error("bad inline flags: no flags after '-'", source.string, source.pos)
    else:
        flags_off = 0
    if flags_on & LOCALE:
        info.inline_locale = True
    return (flags_on, flags_off)

def parse_subpattern(source, info, flags_on, flags_off):
    """Parses a subpattern with scoped flags."""
    saved_flags = info.flags
    info.flags = (info.flags | flags_on) & ~flags_off
    source.ignore_space = bool(info.flags & VERBOSE)
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    return subpattern

def parse_flags_subpattern(source, info):
    """Parses a flags subpattern. It could be inline flags or a subpattern
    possibly with local flags. If it's a subpattern, then that's returned;
    if it's a inline flags, then None is returned.
    """
    (flags_on, flags_off) = parse_flags(source, info)
    if flags_off & GLOBAL_FLAGS:
        raise error('bad inline flags: cannot turn off global flag', source.string, source.pos)
    if flags_on & flags_off:
        raise error('bad inline flags: flag turned on and off', source.string, source.pos)
    new_global_flags = flags_on & ~info.global_flags & GLOBAL_FLAGS
    if new_global_flags:
        info.global_flags |= new_global_flags
        raise _UnscopedFlagSet(info.global_flags)
    flags_on &= ~GLOBAL_FLAGS
    if source.match(':'):
        return parse_subpattern(source, info, flags_on, flags_off)
    if source.match(')'):
        parse_positional_flags(source, info, flags_on, flags_off)
        return None
    raise error('unknown extension', source.string, source.pos)

def parse_positional_flags(source, info, flags_on, flags_off):
    """Parses positional flags."""
    version = (info.flags & _ALL_VERSIONS or DEFAULT_VERSION)
    if version == VERSION0:
        if flags_off:
            raise error('bad inline flags: cannot turn flags off', source.string, source.pos)
        new_global_flags = flags_on & ~info.global_flags
        if new_global_flags:
            info.global_flags |= new_global_flags
            raise _UnscopedFlagSet(info.global_flags)
    else:
        info.flags = (info.flags | flags_on) & ~flags_off
    source.ignore_space = bool(info.flags & VERBOSE)

def parse_name(source, allow_numeric=False, allow_group_0=False):
    """Parses a name."""
    name = source.get_while(set(')>'), include=False)
    if not name:
        raise error('missing group name', source.string, source.pos)
    if name.isdigit():
        min_group = (0 if allow_group_0 else 1)
        if (not allow_numeric or int(name) < min_group):
            raise error('bad character in group name', source.string, source.pos)
    elif not name.isidentifier():
        raise error('bad character in group name', source.string, source.pos)
    return name

def is_octal(string):
    """Checks whether a string is octal."""
    return all((ch in OCT_DIGITS for ch in string))

def is_decimal(string):
    """Checks whether a string is decimal."""
    return all((ch in DIGITS for ch in string))

def is_hexadecimal(string):
    """Checks whether a string is hexadecimal."""
    return all((ch in HEX_DIGITS for ch in string))

def parse_escape(source, info, in_set):
    """Parses an escape sequence."""
    saved_ignore = source.ignore_space
    source.ignore_space = False
    ch = source.get()
    source.ignore_space = saved_ignore
    if not ch:
        raise error('bad escape (end of pattern)', source.string, source.pos)
    if ch in HEX_ESCAPES:
        return parse_hex_escape(source, info, ch, HEX_ESCAPES[ch], in_set, ch)
    elif (ch == 'g' and not in_set):
        saved_pos = source.pos
        try:
            return parse_group_ref(source, info)
        except error:
            source.pos = saved_pos
        return make_character(info, ord(ch), in_set)
    elif (ch == 'G' and not in_set):
        return SearchAnchor()
    elif (ch == 'L' and not in_set):
        return parse_string_set(source, info)
    elif ch == 'N':
        return parse_named_char(source, info, in_set)
    elif ch in 'pP':
        return parse_property(source, info, ch == 'p', in_set)
    elif (ch == 'X' and not in_set):
        return Grapheme()
    elif ch in ALPHA:
        if not in_set:
            if info.flags & WORD:
                value = WORD_POSITION_ESCAPES.get(ch)
            else:
                value = POSITION_ESCAPES.get(ch)
            if value:
                return value
        value = CHARSET_ESCAPES.get(ch)
        if value:
            return value
        value = CHARACTER_ESCAPES.get(ch)
        if value:
            return Character(ord(value))
        raise error('bad escape \\%s' % ch, source.string, source.pos)
    elif ch in DIGITS:
        return parse_numeric_escape(source, info, ch, in_set)
    else:
        return make_character(info, ord(ch), in_set)

def parse_numeric_escape(source, info, ch, in_set):
    """Parses a numeric escape sequence."""
    if (in_set or ch == '0'):
        return parse_octal_escape(source, info, [ch], in_set)
    digits = ch
    saved_pos = source.pos
    ch = source.get()
    if ch in DIGITS:
        digits += ch
        saved_pos = source.pos
        ch = source.get()
        if (is_octal(digits) and ch in OCT_DIGITS):
            encoding = info.flags & _ALL_ENCODINGS
            if (encoding == ASCII or encoding == LOCALE):
                octal_mask = 255
            else:
                octal_mask = 511
            value = int(digits + ch, 8) & octal_mask
            return make_character(info, value)
    source.pos = saved_pos
    if info.is_open_group(digits):
        raise error('cannot refer to an open group', source.string, source.pos)
    return make_ref_group(info, digits, source.pos)

def parse_octal_escape(source, info, digits, in_set):
    """Parses an octal escape sequence."""
    saved_pos = source.pos
    ch = source.get()
    while (len(digits) < 3 and ch in OCT_DIGITS):
        digits.append(ch)
        saved_pos = source.pos
        ch = source.get()
    source.pos = saved_pos
    try:
        value = int(''.join(digits), 8)
        return make_character(info, value, in_set)
    except ValueError:
        if digits[0] in OCT_DIGITS:
            raise error('incomplete escape \\%s' % ''.join(digits), source.string, source.pos)
        else:
            raise error('bad escape \\%s' % digits[0], source.string, source.pos)

def parse_hex_escape(source, info, esc, expected_len, in_set, type):
    """Parses a hex escape sequence."""
    saved_pos = source.pos
    digits = []
    for i in range(expected_len):
        ch = source.get()
        if ch not in HEX_DIGITS:
            raise error('incomplete escape \\%s%s' % (type, ''.join(digits)), source.string, saved_pos)
        digits.append(ch)
    try:
        value = int(''.join(digits), 16)
    except ValueError:
        pass
    else:
        if value < 1114112:
            return make_character(info, value, in_set)
    raise error('bad hex escape \\%s%s' % (esc, ''.join(digits)), source.string, saved_pos)

def parse_group_ref(source, info):
    """Parses a group reference."""
    source.expect('<')
    saved_pos = source.pos
    name = parse_name(source, True)
    source.expect('>')
    if info.is_open_group(name):
        raise error('cannot refer to an open group', source.string, source.pos)
    return make_ref_group(info, name, saved_pos)

def parse_string_set(source, info):
    """Parses a string set reference."""
    source.expect('<')
    name = parse_name(source, True)
    source.expect('>')
    if (name is None or name not in info.kwargs):
        raise error('undefined named list', source.string, source.pos)
    return make_string_set(info, name)

def parse_named_char(source, info, in_set):
    """Parses a named character."""
    saved_pos = source.pos
    if source.match('{'):
        name = source.get_while(NAMED_CHAR_PART)
        if source.match('}'):
            try:
                value = unicodedata.lookup(name)
                return make_character(info, ord(value), in_set)
            except KeyError:
                raise error('undefined character name', source.string, source.pos)
    source.pos = saved_pos
    return make_character(info, ord('N'), in_set)

def parse_property(source, info, positive, in_set):
    """Parses a Unicode property."""
    saved_pos = source.pos
    ch = source.get()
    if ch == '{':
        negate = source.match('^')
        (prop_name, name) = parse_property_name(source)
        if source.match('}'):
            prop = lookup_property(prop_name, name, positive != negate, source)
            return make_property(info, prop, in_set)
    elif (ch and ch in 'CLMNPSZ'):
        prop = lookup_property(None, ch, positive, source)
        return make_property(info, prop, in_set)
    source.pos = saved_pos
    ch = ('p' if positive else 'P')
    return make_character(info, ord(ch), in_set)

def parse_property_name(source):
    """Parses a property name, which may be qualified."""
    name = source.get_while(PROPERTY_NAME_PART)
    saved_pos = source.pos
    ch = source.get()
    if (ch and ch in ':='):
        prop_name = name
        name = source.get_while(ALNUM | set(' &_-./')).strip()
        if name:
            saved_pos = source.pos
        else:
            (prop_name, name) = (None, prop_name)
    else:
        prop_name = None
    source.pos = saved_pos
    return (prop_name, name)

def parse_set(source, info):
    """Parses a character set."""
    version = (info.flags & _ALL_VERSIONS or DEFAULT_VERSION)
    saved_ignore = source.ignore_space
    source.ignore_space = False
    negate = source.match('^')
    try:
        if version == VERSION0:
            item = parse_set_imp_union(source, info)
        else:
            item = parse_set_union(source, info)
        if not source.match(']'):
            raise error('missing ]', source.string, source.pos)
    finally:
        source.ignore_space = saved_ignore
    if negate:
        item = item.with_flags(positive=not item.positive)
    item = item.with_flags(case_flags=make_case_flags(info))
    return item

def parse_set_union(source, info):
    """Parses a set union ([x||y])."""
    items = [parse_set_symm_diff(source, info)]
    while source.match('||'):
        items.append(parse_set_symm_diff(source, info))
    if len(items) == 1:
        return items[0]
    return SetUnion(info, items)

def parse_set_symm_diff(source, info):
    """Parses a set symmetric difference ([x~~y])."""
    items = [parse_set_inter(source, info)]
    while source.match('~~'):
        items.append(parse_set_inter(source, info))
    if len(items) == 1:
        return items[0]
    return SetSymDiff(info, items)

def parse_set_inter(source, info):
    """Parses a set intersection ([x&&y])."""
    items = [parse_set_diff(source, info)]
    while source.match('&&'):
        items.append(parse_set_diff(source, info))
    if len(items) == 1:
        return items[0]
    return SetInter(info, items)

def parse_set_diff(source, info):
    """Parses a set difference ([x--y])."""
    items = [parse_set_imp_union(source, info)]
    while source.match('--'):
        items.append(parse_set_imp_union(source, info))
    if len(items) == 1:
        return items[0]
    return SetDiff(info, items)

def parse_set_imp_union(source, info):
    """Parses a set implicit union ([xy])."""
    version = (info.flags & _ALL_VERSIONS or DEFAULT_VERSION)
    items = [parse_set_member(source, info)]
    while True:
        saved_pos = source.pos
        if source.match(']'):
            source.pos = saved_pos
            break
        if (version == VERSION1 and any((source.match(op) for op in SET_OPS))):
            source.pos = saved_pos
            break
        items.append(parse_set_member(source, info))
    if len(items) == 1:
        return items[0]
    return SetUnion(info, items)

def parse_set_member(source, info):
    """Parses a member in a character set."""
    start = parse_set_item(source, info)
    saved_pos1 = source.pos
    if (not isinstance(start, Character) or not start.positive or not source.match('-')):
        return start
    version = (info.flags & _ALL_VERSIONS or DEFAULT_VERSION)
    saved_pos2 = source.pos
    if (version == VERSION1 and source.match('-')):
        source.pos = saved_pos1
        return start
    if source.match(']'):
        source.pos = saved_pos2
        return SetUnion(info, [start, Character(ord('-'))])
    end = parse_set_item(source, info)
    if (not isinstance(end, Character) or not end.positive):
        return SetUnion(info, [start, Character(ord('-')), end])
    if start.value > end.value:
        raise error('bad character range', source.string, source.pos)
    if start.value == end.value:
        return start
    return Range(start.value, end.value)

def parse_set_item(source, info):
    """Parses an item in a character set."""
    version = (info.flags & _ALL_VERSIONS or DEFAULT_VERSION)
    if source.match('\\'):
        return parse_escape(source, info, True)
    saved_pos = source.pos
    if source.match('[:'):
        try:
            return parse_posix_class(source, info)
        except ParseError:
            source.pos = saved_pos
    if (version == VERSION1 and source.match('[')):
        negate = source.match('^')
        item = parse_set_union(source, info)
        if not source.match(']'):
            raise error('missing ]', source.string, source.pos)
        if negate:
            item = item.with_flags(positive=not item.positive)
        return item
    ch = source.get()
    if not ch:
        raise error('unterminated character set', source.string, source.pos)
    return Character(ord(ch))

def parse_posix_class(source, info):
    """Parses a POSIX character class."""
    negate = source.match('^')
    (prop_name, name) = parse_property_name(source)
    if not source.match(':]'):
        raise ParseError()
    return lookup_property(prop_name, name, not negate, source, posix=True)

def float_to_rational(flt):
    """Converts a float to a rational pair."""
    int_part = int(flt)
    error = flt - int_part
    if abs(error) < 0.0001:
        return (int_part, 1)
    (den, num) = float_to_rational(1.0 / error)
    return (int_part * den + num, den)

def numeric_to_rational(numeric):
    """Converts a numeric string to a rational string, if possible."""
    if numeric[:1] == '-':
        (sign, numeric) = (numeric[0], numeric[1:])
    else:
        sign = ''
    parts = numeric.split('/')
    if len(parts) == 2:
        (num, den) = float_to_rational(float(parts[0]) / float(parts[1]))
    elif len(parts) == 1:
        (num, den) = float_to_rational(float(parts[0]))
    else:
        raise ValueError()
    result = '{}{}/{}'.format(sign, num, den)
    if result.endswith('/1'):
        return result[:-2]
    return result

def standardise_name(name):
    """Standardises a property or value name."""
    try:
        return numeric_to_rational(''.join(name))
    except (ValueError, ZeroDivisionError):
        return ''.join((ch for ch in name if ch not in '_- ')).upper()
_POSIX_CLASSES = set('ALNUM DIGIT PUNCT XDIGIT'.split())
_BINARY_VALUES = set('YES Y NO N TRUE T FALSE F'.split())

def lookup_property(property, value, positive, source=None, posix=False):
    """Looks up a property."""
    property = (standardise_name(property) if property else None)
    value = standardise_name(value)
    if (property, value) == ('GENERALCATEGORY', 'ASSIGNED'):
        (property, value, positive) = ('GENERALCATEGORY', 'UNASSIGNED', not positive)
    if (posix and not property and value.upper() in _POSIX_CLASSES):
        value = 'POSIX' + value
    if property:
        prop = PROPERTIES.get(property)
        if not prop:
            if not source:
                raise error('unknown property')
            raise error('unknown property', source.string, source.pos)
        (prop_id, value_dict) = prop
        val_id = value_dict.get(value)
        if val_id is None:
            if not source:
                raise error('unknown property value')
            raise error('unknown property value', source.string, source.pos)
        return Property(prop_id << 16 | val_id, positive)
    for property in ('GC', 'SCRIPT', 'BLOCK'):
        (prop_id, value_dict) = PROPERTIES.get(property)
        val_id = value_dict.get(value)
        if val_id is not None:
            return Property(prop_id << 16 | val_id, positive)
    prop = PROPERTIES.get(value)
    if prop:
        (prop_id, value_dict) = prop
        if set(value_dict) == _BINARY_VALUES:
            return Property(prop_id << 16 | 1, positive)
        return Property(prop_id << 16, not positive)
    if value.startswith('IS'):
        prop = PROPERTIES.get(value[2:])
        if prop:
            (prop_id, value_dict) = prop
            if 'YES' in value_dict:
                return Property(prop_id << 16 | 1, positive)
    for (prefix, property) in (('IS', 'SCRIPT'), ('IN', 'BLOCK')):
        if value.startswith(prefix):
            (prop_id, value_dict) = PROPERTIES.get(property)
            val_id = value_dict.get(value[2:])
            if val_id is not None:
                return Property(prop_id << 16 | val_id, positive)
    if not source:
        raise error('unknown property')
    raise error('unknown property', source.string, source.pos)

def _compile_replacement(source, pattern, is_unicode):
    """Compiles a replacement template escape sequence."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('regex._regex_core._compile_replacement', '_compile_replacement(source, pattern, is_unicode)', {'ALPHA': ALPHA, 'CHARACTER_ESCAPES': CHARACTER_ESCAPES, 'HEX_ESCAPES': HEX_ESCAPES, 'parse_repl_hex_escape': parse_repl_hex_escape, 'compile_repl_group': compile_repl_group, 'parse_repl_named_char': parse_repl_named_char, 'error': error, 'OCT_DIGITS': OCT_DIGITS, 'DIGITS': DIGITS, 'is_octal': is_octal, 'source': source, 'pattern': pattern, 'is_unicode': is_unicode}, 2)

def parse_repl_hex_escape(source, expected_len, type):
    """Parses a hex escape sequence in a replacement string."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('regex._regex_core.parse_repl_hex_escape', 'parse_repl_hex_escape(source, expected_len, type)', {'HEX_DIGITS': HEX_DIGITS, 'error': error, 'source': source, 'expected_len': expected_len, 'type': type}, 1)

def parse_repl_named_char(source):
    """Parses a named character in a replacement string."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('regex._regex_core.parse_repl_named_char', 'parse_repl_named_char(source)', {'ALPHA': ALPHA, 'unicodedata': unicodedata, 'error': error, 'source': source}, 1)

def compile_repl_group(source, pattern):
    """Compiles a replacement template group reference."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('regex._regex_core.compile_repl_group', 'compile_repl_group(source, pattern)', {'parse_name': parse_name, 'error': error, 'source': source, 'pattern': pattern}, 1)
INDENT = '  '
POSITIVE_OP = 1
ZEROWIDTH_OP = 2
FUZZY_OP = 4
REVERSE_OP = 8
REQUIRED_OP = 16
POS_TEXT = {False: 'NON-MATCH', True: 'MATCH'}
CASE_TEXT = {NOCASE: '', IGNORECASE: ' SIMPLE_IGNORE_CASE', FULLCASE: '', FULLIGNORECASE: ' FULL_IGNORE_CASE'}

def make_sequence(items):
    if len(items) == 1:
        return items[0]
    return Sequence(items)


class RegexBase:
    
    def __init__(self):
        self._key = self.__class__
    
    def with_flags(self, positive=None, case_flags=None, zerowidth=None):
        if positive is None:
            positive = self.positive
        else:
            positive = bool(positive)
        if case_flags is None:
            case_flags = self.case_flags
        else:
            case_flags = CASE_FLAGS_COMBINATIONS[case_flags & CASE_FLAGS]
        if zerowidth is None:
            zerowidth = self.zerowidth
        else:
            zerowidth = bool(zerowidth)
        if (positive == self.positive and case_flags == self.case_flags and zerowidth == self.zerowidth):
            return self
        return self.rebuild(positive, case_flags, zerowidth)
    
    def fix_groups(self, pattern, reverse, fuzzy):
        pass
    
    def optimise(self, info, reverse):
        return self
    
    def pack_characters(self, info):
        return self
    
    def remove_captures(self):
        return self
    
    def is_atomic(self):
        return True
    
    def can_be_affix(self):
        return True
    
    def contains_group(self):
        return False
    
    def get_firstset(self, reverse):
        raise _FirstSetError()
    
    def has_simple_start(self):
        return False
    
    def compile(self, reverse=False, fuzzy=False):
        return self._compile(reverse, fuzzy)
    
    def is_empty(self):
        return False
    
    def __hash__(self):
        return hash(self._key)
    
    def __eq__(self, other):
        return (type(self) is type(other) and self._key == other._key)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def get_required_string(self, reverse):
        return (self.max_width(), None)



class ZeroWidthBase(RegexBase):
    
    def __init__(self, positive=True):
        RegexBase.__init__(self)
        self.positive = bool(positive)
        self._key = (self.__class__, self.positive)
    
    def get_firstset(self, reverse):
        return set([None])
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if fuzzy:
            flags |= FUZZY_OP
        if reverse:
            flags |= REVERSE_OP
        return [(self._opcode, flags)]
    
    def dump(self, indent, reverse):
        print('{}{} {}'.format(INDENT * indent, self._op_name, POS_TEXT[self.positive]))
    
    def max_width(self):
        return 0



class Any(RegexBase):
    _opcode = {False: OP.ANY, True: OP.ANY_REV}
    _op_name = 'ANY'
    
    def has_simple_start(self):
        return True
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if fuzzy:
            flags |= FUZZY_OP
        return [(self._opcode[reverse], flags)]
    
    def dump(self, indent, reverse):
        print('{}{}'.format(INDENT * indent, self._op_name))
    
    def max_width(self):
        return 1



class AnyAll(Any):
    _opcode = {False: OP.ANY_ALL, True: OP.ANY_ALL_REV}
    _op_name = 'ANY_ALL'



class AnyU(Any):
    _opcode = {False: OP.ANY_U, True: OP.ANY_U_REV}
    _op_name = 'ANY_U'



class Atomic(RegexBase):
    
    def __init__(self, subpattern):
        RegexBase.__init__(self)
        self.subpattern = subpattern
    
    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, fuzzy)
    
    def optimise(self, info, reverse):
        self.subpattern = self.subpattern.optimise(info, reverse)
        if self.subpattern.is_empty():
            return self.subpattern
        return self
    
    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self
    
    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        return self
    
    def can_be_affix(self):
        return self.subpattern.can_be_affix()
    
    def contains_group(self):
        return self.subpattern.contains_group()
    
    def get_firstset(self, reverse):
        return self.subpattern.get_firstset(reverse)
    
    def has_simple_start(self):
        return self.subpattern.has_simple_start()
    
    def _compile(self, reverse, fuzzy):
        return [(OP.ATOMIC, )] + self.subpattern.compile(reverse, fuzzy) + [(OP.END, )]
    
    def dump(self, indent, reverse):
        print('{}ATOMIC'.format(INDENT * indent))
        self.subpattern.dump(indent + 1, reverse)
    
    def is_empty(self):
        return self.subpattern.is_empty()
    
    def __eq__(self, other):
        return (type(self) is type(other) and self.subpattern == other.subpattern)
    
    def max_width(self):
        return self.subpattern.max_width()
    
    def get_required_string(self, reverse):
        return self.subpattern.get_required_string(reverse)



class Boundary(ZeroWidthBase):
    _opcode = OP.BOUNDARY
    _op_name = 'BOUNDARY'



class Branch(RegexBase):
    
    def __init__(self, branches):
        RegexBase.__init__(self)
        self.branches = branches
    
    def fix_groups(self, pattern, reverse, fuzzy):
        for b in self.branches:
            b.fix_groups(pattern, reverse, fuzzy)
    
    def optimise(self, info, reverse):
        if not self.branches:
            return Sequence([])
        branches = Branch._flatten_branches(info, reverse, self.branches)
        if reverse:
            (suffix, branches) = Branch._split_common_suffix(info, branches)
            prefix = []
        else:
            (prefix, branches) = Branch._split_common_prefix(info, branches)
            suffix = []
        branches = Branch._reduce_to_set(info, reverse, branches)
        if len(branches) > 1:
            sequence = [Branch(branches)]
            if (not prefix or not suffix):
                firstset = self._add_precheck(info, reverse, branches)
                if firstset:
                    if reverse:
                        sequence.append(firstset)
                    else:
                        sequence.insert(0, firstset)
        else:
            sequence = branches
        return make_sequence(prefix + sequence + suffix)
    
    def _add_precheck(self, info, reverse, branches):
        charset = set()
        pos = (-1 if reverse else 0)
        for branch in branches:
            if (type(branch) is Literal and branch.case_flags == NOCASE):
                charset.add(branch.characters[pos])
            else:
                return
        if not charset:
            return None
        return _check_firstset(info, reverse, [Character(c) for c in charset])
    
    def pack_characters(self, info):
        self.branches = [b.pack_characters(info) for b in self.branches]
        return self
    
    def remove_captures(self):
        self.branches = [b.remove_captures() for b in self.branches]
        return self
    
    def is_atomic(self):
        return all((b.is_atomic() for b in self.branches))
    
    def can_be_affix(self):
        return all((b.can_be_affix() for b in self.branches))
    
    def contains_group(self):
        return any((b.contains_group() for b in self.branches))
    
    def get_firstset(self, reverse):
        fs = set()
        for b in self.branches:
            fs |= b.get_firstset(reverse)
        return (fs or set([None]))
    
    def _compile(self, reverse, fuzzy):
        code = [(OP.BRANCH, )]
        for b in self.branches:
            code.extend(b.compile(reverse, fuzzy))
            code.append((OP.NEXT, ))
        code[-1] = (OP.END, )
        return code
    
    def dump(self, indent, reverse):
        print('{}BRANCH'.format(INDENT * indent))
        self.branches[0].dump(indent + 1, reverse)
        for b in self.branches[1:]:
            print('{}OR'.format(INDENT * indent))
            b.dump(indent + 1, reverse)
    
    @staticmethod
    def _flatten_branches(info, reverse, branches):
        new_branches = []
        for b in branches:
            b = b.optimise(info, reverse)
            if isinstance(b, Branch):
                new_branches.extend(b.branches)
            else:
                new_branches.append(b)
        return new_branches
    
    @staticmethod
    def _split_common_prefix(info, branches):
        alternatives = []
        for b in branches:
            if isinstance(b, Sequence):
                alternatives.append(b.items)
            else:
                alternatives.append([b])
        max_count = min((len(a) for a in alternatives))
        prefix = alternatives[0]
        pos = 0
        end_pos = max_count
        while (pos < end_pos and prefix[pos].can_be_affix() and all((a[pos] == prefix[pos] for a in alternatives))):
            pos += 1
        count = pos
        if info.flags & UNICODE:
            count = pos
            while (count > 0 and not all((Branch._can_split(a, count) for a in alternatives))):
                count -= 1
        if count == 0:
            return ([], branches)
        new_branches = []
        for a in alternatives:
            new_branches.append(make_sequence(a[count:]))
        return (prefix[:count], new_branches)
    
    @staticmethod
    def _split_common_suffix(info, branches):
        alternatives = []
        for b in branches:
            if isinstance(b, Sequence):
                alternatives.append(b.items)
            else:
                alternatives.append([b])
        max_count = min((len(a) for a in alternatives))
        suffix = alternatives[0]
        pos = -1
        end_pos = -1 - max_count
        while (pos > end_pos and suffix[pos].can_be_affix() and all((a[pos] == suffix[pos] for a in alternatives))):
            pos -= 1
        count = -1 - pos
        if info.flags & UNICODE:
            while (count > 0 and not all((Branch._can_split_rev(a, count) for a in alternatives))):
                count -= 1
        if count == 0:
            return ([], branches)
        new_branches = []
        for a in alternatives:
            new_branches.append(make_sequence(a[:-count]))
        return (suffix[-count:], new_branches)
    
    @staticmethod
    def _can_split(items, count):
        if not Branch._is_full_case(items, count - 1):
            return True
        if not Branch._is_full_case(items, count):
            return True
        if Branch._is_folded(items[count - 1:count + 1]):
            return False
        if (Branch._is_full_case(items, count + 2) and Branch._is_folded(items[count - 1:count + 2])):
            return False
        if (Branch._is_full_case(items, count - 2) and Branch._is_folded(items[count - 2:count + 1])):
            return False
        return True
    
    @staticmethod
    def _can_split_rev(items, count):
        end = len(items)
        if not Branch._is_full_case(items, end - count):
            return True
        if not Branch._is_full_case(items, end - count - 1):
            return True
        if Branch._is_folded(items[end - count - 1:end - count + 1]):
            return False
        if (Branch._is_full_case(items, end - count + 2) and Branch._is_folded(items[end - count - 1:end - count + 2])):
            return False
        if (Branch._is_full_case(items, end - count - 2) and Branch._is_folded(items[end - count - 2:end - count + 1])):
            return False
        return True
    
    @staticmethod
    def _merge_common_prefixes(info, reverse, branches):
        prefixed = defaultdict(list)
        order = {}
        new_branches = []
        for b in branches:
            if Branch._is_simple_character(b):
                prefixed[b.value].append([b])
                order.setdefault(b.value, len(order))
            elif (isinstance(b, Sequence) and b.items and Branch._is_simple_character(b.items[0])):
                prefixed[b.items[0].value].append(b.items)
                order.setdefault(b.items[0].value, len(order))
            else:
                Branch._flush_char_prefix(info, reverse, prefixed, order, new_branches)
                new_branches.append(b)
        Branch._flush_char_prefix(info, prefixed, order, new_branches)
        return new_branches
    
    @staticmethod
    def _is_simple_character(c):
        return (isinstance(c, Character) and c.positive and not c.case_flags)
    
    @staticmethod
    def _reduce_to_set(info, reverse, branches):
        new_branches = []
        items = set()
        case_flags = NOCASE
        for b in branches:
            if isinstance(b, (Character, Property, SetBase)):
                if b.case_flags != case_flags:
                    Branch._flush_set_members(info, reverse, items, case_flags, new_branches)
                    case_flags = b.case_flags
                items.add(b.with_flags(case_flags=NOCASE))
            else:
                Branch._flush_set_members(info, reverse, items, case_flags, new_branches)
                new_branches.append(b)
        Branch._flush_set_members(info, reverse, items, case_flags, new_branches)
        return new_branches
    
    @staticmethod
    def _flush_char_prefix(info, reverse, prefixed, order, new_branches):
        if not prefixed:
            return
        for (value, branches) in sorted(prefixed.items(), key=lambda pair: order[pair[0]]):
            if len(branches) == 1:
                new_branches.append(make_sequence(branches[0]))
            else:
                subbranches = []
                optional = False
                for b in branches:
                    if len(b) > 1:
                        subbranches.append(make_sequence(b[1:]))
                    elif not optional:
                        subbranches.append(Sequence())
                        optional = True
                sequence = Sequence([Character(value), Branch(subbranches)])
                new_branches.append(sequence.optimise(info, reverse))
        prefixed.clear()
        order.clear()
    
    @staticmethod
    def _flush_set_members(info, reverse, items, case_flags, new_branches):
        if not items:
            return
        if len(items) == 1:
            item = list(items)[0]
        else:
            item = SetUnion(info, list(items)).optimise(info, reverse)
        new_branches.append(item.with_flags(case_flags=case_flags))
        items.clear()
    
    @staticmethod
    def _is_full_case(items, i):
        if not 0 <= i < len(items):
            return False
        item = items[i]
        return (isinstance(item, Character) and item.positive and item.case_flags & FULLIGNORECASE == FULLIGNORECASE)
    
    @staticmethod
    def _is_folded(items):
        if len(items) < 2:
            return False
        for i in items:
            if (not isinstance(i, Character) or not i.positive or not i.case_flags):
                return False
        folded = ''.join((chr(i.value) for i in items))
        folded = _regex.fold_case(FULL_CASE_FOLDING, folded)
        expanding_chars = _regex.get_expand_on_folding()
        for c in expanding_chars:
            if folded == _regex.fold_case(FULL_CASE_FOLDING, c):
                return True
        return False
    
    def is_empty(self):
        return all((b.is_empty() for b in self.branches))
    
    def __eq__(self, other):
        return (type(self) is type(other) and self.branches == other.branches)
    
    def max_width(self):
        return max((b.max_width() for b in self.branches))



class CallGroup(RegexBase):
    
    def __init__(self, info, group, position):
        RegexBase.__init__(self)
        self.info = info
        self.group = group
        self.position = position
        self._key = (self.__class__, self.group)
    
    def fix_groups(self, pattern, reverse, fuzzy):
        try:
            self.group = int(self.group)
        except ValueError:
            try:
                self.group = self.info.group_index[self.group]
            except KeyError:
                raise error('invalid group reference', pattern, self.position)
        if not 0 <= self.group <= self.info.group_count:
            raise error('unknown group', pattern, self.position)
        if (self.group > 0 and self.info.open_group_count[self.group] > 1):
            raise error('ambiguous group reference', pattern, self.position)
        self.info.group_calls.append((self, reverse, fuzzy))
        self._key = (self.__class__, self.group)
    
    def remove_captures(self):
        raise error('group reference not allowed', pattern, self.position)
    
    def _compile(self, reverse, fuzzy):
        return [(OP.GROUP_CALL, self.call_ref)]
    
    def dump(self, indent, reverse):
        print('{}GROUP_CALL {}'.format(INDENT * indent, self.group))
    
    def __eq__(self, other):
        return (type(self) is type(other) and self.group == other.group)
    
    def max_width(self):
        return UNLIMITED
    
    def __del__(self):
        self.info = None



class CallRef(RegexBase):
    
    def __init__(self, ref, parsed):
        self.ref = ref
        self.parsed = parsed
    
    def _compile(self, reverse, fuzzy):
        return [(OP.CALL_REF, self.ref)] + self.parsed._compile(reverse, fuzzy) + [(OP.END, )]



class Character(RegexBase):
    _opcode = {(NOCASE, False): OP.CHARACTER, (IGNORECASE, False): OP.CHARACTER_IGN, (FULLCASE, False): OP.CHARACTER, (FULLIGNORECASE, False): OP.CHARACTER_IGN, (NOCASE, True): OP.CHARACTER_REV, (IGNORECASE, True): OP.CHARACTER_IGN_REV, (FULLCASE, True): OP.CHARACTER_REV, (FULLIGNORECASE, True): OP.CHARACTER_IGN_REV}
    
    def __init__(self, value, positive=True, case_flags=NOCASE, zerowidth=False):
        RegexBase.__init__(self)
        self.value = value
        self.positive = bool(positive)
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self.zerowidth = bool(zerowidth)
        if (self.positive and self.case_flags & FULLIGNORECASE == FULLIGNORECASE):
            self.folded = _regex.fold_case(FULL_CASE_FOLDING, chr(self.value))
        else:
            self.folded = chr(self.value)
        self._key = (self.__class__, self.value, self.positive, self.case_flags, self.zerowidth)
    
    def rebuild(self, positive, case_flags, zerowidth):
        return Character(self.value, positive, case_flags, zerowidth)
    
    def optimise(self, info, reverse, in_set=False):
        return self
    
    def get_firstset(self, reverse):
        return set([self])
    
    def has_simple_start(self):
        return True
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if self.zerowidth:
            flags |= ZEROWIDTH_OP
        if fuzzy:
            flags |= FUZZY_OP
        code = PrecompiledCode([self._opcode[(self.case_flags, reverse)], flags, self.value])
        if len(self.folded) > 1:
            code = Branch([code, String([ord(c) for c in self.folded], case_flags=self.case_flags)])
        return code.compile(reverse, fuzzy)
    
    def dump(self, indent, reverse):
        display = ascii(chr(self.value)).lstrip('bu')
        print('{}CHARACTER {} {}{}'.format(INDENT * indent, POS_TEXT[self.positive], display, CASE_TEXT[self.case_flags]))
    
    def matches(self, ch):
        return ch == self.value == self.positive
    
    def max_width(self):
        return len(self.folded)
    
    def get_required_string(self, reverse):
        if not self.positive:
            return (1, None)
        self.folded_characters = tuple((ord(c) for c in self.folded))
        return (0, self)



class Conditional(RegexBase):
    
    def __init__(self, info, group, yes_item, no_item, position):
        RegexBase.__init__(self)
        self.info = info
        self.group = group
        self.yes_item = yes_item
        self.no_item = no_item
        self.position = position
    
    def fix_groups(self, pattern, reverse, fuzzy):
        try:
            self.group = int(self.group)
        except ValueError:
            try:
                self.group = self.info.group_index[self.group]
            except KeyError:
                if self.group == 'DEFINE':
                    self.group = 0
                else:
                    raise error('unknown group', pattern, self.position)
        if not 0 <= self.group <= self.info.group_count:
            raise error('invalid group reference', pattern, self.position)
        self.yes_item.fix_groups(pattern, reverse, fuzzy)
        self.no_item.fix_groups(pattern, reverse, fuzzy)
    
    def optimise(self, info, reverse):
        yes_item = self.yes_item.optimise(info, reverse)
        no_item = self.no_item.optimise(info, reverse)
        return Conditional(info, self.group, yes_item, no_item, self.position)
    
    def pack_characters(self, info):
        self.yes_item = self.yes_item.pack_characters(info)
        self.no_item = self.no_item.pack_characters(info)
        return self
    
    def remove_captures(self):
        self.yes_item = self.yes_item.remove_captures()
        self.no_item = self.no_item.remove_captures()
    
    def is_atomic(self):
        return (self.yes_item.is_atomic() and self.no_item.is_atomic())
    
    def can_be_affix(self):
        return (self.yes_item.can_be_affix() and self.no_item.can_be_affix())
    
    def contains_group(self):
        return (self.yes_item.contains_group() or self.no_item.contains_group())
    
    def get_firstset(self, reverse):
        return self.yes_item.get_firstset(reverse) | self.no_item.get_firstset(reverse)
    
    def _compile(self, reverse, fuzzy):
        code = [(OP.GROUP_EXISTS, self.group)]
        code.extend(self.yes_item.compile(reverse, fuzzy))
        add_code = self.no_item.compile(reverse, fuzzy)
        if add_code:
            code.append((OP.NEXT, ))
            code.extend(add_code)
        code.append((OP.END, ))
        return code
    
    def dump(self, indent, reverse):
        print('{}GROUP_EXISTS {}'.format(INDENT * indent, self.group))
        self.yes_item.dump(indent + 1, reverse)
        if not self.no_item.is_empty():
            print('{}OR'.format(INDENT * indent))
            self.no_item.dump(indent + 1, reverse)
    
    def is_empty(self):
        return (self.yes_item.is_empty() and self.no_item.is_empty())
    
    def __eq__(self, other):
        return (type(self) is type(other) and (self.group, self.yes_item, self.no_item) == (other.group, other.yes_item, other.no_item))
    
    def max_width(self):
        return max(self.yes_item.max_width(), self.no_item.max_width())
    
    def __del__(self):
        self.info = None



class DefaultBoundary(ZeroWidthBase):
    _opcode = OP.DEFAULT_BOUNDARY
    _op_name = 'DEFAULT_BOUNDARY'



class DefaultEndOfWord(ZeroWidthBase):
    _opcode = OP.DEFAULT_END_OF_WORD
    _op_name = 'DEFAULT_END_OF_WORD'



class DefaultStartOfWord(ZeroWidthBase):
    _opcode = OP.DEFAULT_START_OF_WORD
    _op_name = 'DEFAULT_START_OF_WORD'



class EndOfLine(ZeroWidthBase):
    _opcode = OP.END_OF_LINE
    _op_name = 'END_OF_LINE'



class EndOfLineU(EndOfLine):
    _opcode = OP.END_OF_LINE_U
    _op_name = 'END_OF_LINE_U'



class EndOfString(ZeroWidthBase):
    _opcode = OP.END_OF_STRING
    _op_name = 'END_OF_STRING'



class EndOfStringLine(ZeroWidthBase):
    _opcode = OP.END_OF_STRING_LINE
    _op_name = 'END_OF_STRING_LINE'



class EndOfStringLineU(EndOfStringLine):
    _opcode = OP.END_OF_STRING_LINE_U
    _op_name = 'END_OF_STRING_LINE_U'



class EndOfWord(ZeroWidthBase):
    _opcode = OP.END_OF_WORD
    _op_name = 'END_OF_WORD'



class Failure(ZeroWidthBase):
    _op_name = 'FAILURE'
    
    def _compile(self, reverse, fuzzy):
        return [(OP.FAILURE, )]



class Fuzzy(RegexBase):
    
    def __init__(self, subpattern, constraints=None):
        RegexBase.__init__(self)
        if constraints is None:
            constraints = {}
        self.subpattern = subpattern
        self.constraints = constraints
        if 'cost' in constraints:
            for e in 'dis':
                if e in constraints['cost']:
                    constraints.setdefault(e, (0, None))
        if set(constraints) & set('dis'):
            for e in 'dis':
                constraints.setdefault(e, (0, 0))
        else:
            for e in 'dis':
                constraints.setdefault(e, (0, None))
        constraints.setdefault('e', (0, None))
        if 'cost' in constraints:
            for e in 'dis':
                constraints['cost'].setdefault(e, 0)
        else:
            constraints['cost'] = {'d': 1, 'i': 1, 's': 1, 'max': constraints['e'][1]}
    
    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, True)
    
    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self
    
    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        return self
    
    def is_atomic(self):
        return self.subpattern.is_atomic()
    
    def contains_group(self):
        return self.subpattern.contains_group()
    
    def _compile(self, reverse, fuzzy):
        arguments = []
        for e in 'dise':
            v = self.constraints[e]
            arguments.append(v[0])
            arguments.append((UNLIMITED if v[1] is None else v[1]))
        for e in 'dis':
            arguments.append(self.constraints['cost'][e])
        v = self.constraints['cost']['max']
        arguments.append((UNLIMITED if v is None else v))
        flags = 0
        if reverse:
            flags |= REVERSE_OP
        test = self.constraints.get('test')
        if test:
            return [(OP.FUZZY_EXT, flags) + tuple(arguments)] + test.compile(reverse, True) + [(OP.NEXT, )] + self.subpattern.compile(reverse, True) + [(OP.END, )]
        return [(OP.FUZZY, flags) + tuple(arguments)] + self.subpattern.compile(reverse, True) + [(OP.END, )]
    
    def dump(self, indent, reverse):
        constraints = self._constraints_to_string()
        if constraints:
            constraints = ' ' + constraints
        print('{}FUZZY{}'.format(INDENT * indent, constraints))
        self.subpattern.dump(indent + 1, reverse)
    
    def is_empty(self):
        return self.subpattern.is_empty()
    
    def __eq__(self, other):
        return (type(self) is type(other) and self.subpattern == other.subpattern and self.constraints == other.constraints)
    
    def max_width(self):
        return UNLIMITED
    
    def _constraints_to_string(self):
        constraints = []
        for name in 'ids':
            (min, max) = self.constraints[name]
            if max == 0:
                continue
            con = ''
            if min > 0:
                con = '{}<='.format(min)
            con += name
            if max is not None:
                con += '<={}'.format(max)
            constraints.append(con)
        cost = []
        for name in 'ids':
            coeff = self.constraints['cost'][name]
            if coeff > 0:
                cost.append('{}{}'.format(coeff, name))
        limit = self.constraints['cost']['max']
        if (limit is not None and limit > 0):
            cost = '{}<={}'.format('+'.join(cost), limit)
            constraints.append(cost)
        return ','.join(constraints)



class Grapheme(RegexBase):
    
    def _compile(self, reverse, fuzzy):
        grapheme_matcher = Atomic(Sequence([LazyRepeat(AnyAll(), 1, None), GraphemeBoundary()]))
        return grapheme_matcher.compile(reverse, fuzzy)
    
    def dump(self, indent, reverse):
        print('{}GRAPHEME'.format(INDENT * indent))
    
    def max_width(self):
        return UNLIMITED



class GraphemeBoundary:
    
    def compile(self, reverse, fuzzy):
        return [(OP.GRAPHEME_BOUNDARY, 1)]



class GreedyRepeat(RegexBase):
    _opcode = OP.GREEDY_REPEAT
    _op_name = 'GREEDY_REPEAT'
    
    def __init__(self, subpattern, min_count, max_count):
        RegexBase.__init__(self)
        self.subpattern = subpattern
        self.min_count = min_count
        self.max_count = max_count
    
    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, fuzzy)
    
    def optimise(self, info, reverse):
        subpattern = self.subpattern.optimise(info, reverse)
        return type(self)(subpattern, self.min_count, self.max_count)
    
    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self
    
    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        return self
    
    def is_atomic(self):
        return (self.min_count == self.max_count and self.subpattern.is_atomic())
    
    def can_be_affix(self):
        return False
    
    def contains_group(self):
        return self.subpattern.contains_group()
    
    def get_firstset(self, reverse):
        fs = self.subpattern.get_firstset(reverse)
        if self.min_count == 0:
            fs.add(None)
        return fs
    
    def _compile(self, reverse, fuzzy):
        repeat = [self._opcode, self.min_count]
        if self.max_count is None:
            repeat.append(UNLIMITED)
        else:
            repeat.append(self.max_count)
        subpattern = self.subpattern.compile(reverse, fuzzy)
        if not subpattern:
            return []
        return [tuple(repeat)] + subpattern + [(OP.END, )]
    
    def dump(self, indent, reverse):
        if self.max_count is None:
            limit = 'INF'
        else:
            limit = self.max_count
        print('{}{} {} {}'.format(INDENT * indent, self._op_name, self.min_count, limit))
        self.subpattern.dump(indent + 1, reverse)
    
    def is_empty(self):
        return self.subpattern.is_empty()
    
    def __eq__(self, other):
        return (type(self) is type(other) and (self.subpattern, self.min_count, self.max_count) == (other.subpattern, other.min_count, other.max_count))
    
    def max_width(self):
        if self.max_count is None:
            return UNLIMITED
        return self.subpattern.max_width() * self.max_count
    
    def get_required_string(self, reverse):
        max_count = (UNLIMITED if self.max_count is None else self.max_count)
        if self.min_count == 0:
            w = self.subpattern.max_width() * max_count
            return (min(w, UNLIMITED), None)
        (ofs, req) = self.subpattern.get_required_string(reverse)
        if req:
            return (ofs, req)
        w = self.subpattern.max_width() * max_count
        return (min(w, UNLIMITED), None)



class PossessiveRepeat(GreedyRepeat):
    
    def is_atomic(self):
        return True
    
    def _compile(self, reverse, fuzzy):
        subpattern = self.subpattern.compile(reverse, fuzzy)
        if not subpattern:
            return []
        repeat = [self._opcode, self.min_count]
        if self.max_count is None:
            repeat.append(UNLIMITED)
        else:
            repeat.append(self.max_count)
        return [(OP.ATOMIC, ), tuple(repeat)] + subpattern + [(OP.END, ), (OP.END, )]
    
    def dump(self, indent, reverse):
        print('{}ATOMIC'.format(INDENT * indent))
        if self.max_count is None:
            limit = 'INF'
        else:
            limit = self.max_count
        print('{}{} {} {}'.format(INDENT * (indent + 1), self._op_name, self.min_count, limit))
        self.subpattern.dump(indent + 2, reverse)



class Group(RegexBase):
    
    def __init__(self, info, group, subpattern):
        RegexBase.__init__(self)
        self.info = info
        self.group = group
        self.subpattern = subpattern
        self.call_ref = None
    
    def fix_groups(self, pattern, reverse, fuzzy):
        self.info.defined_groups[self.group] = (self, reverse, fuzzy)
        self.subpattern.fix_groups(pattern, reverse, fuzzy)
    
    def optimise(self, info, reverse):
        subpattern = self.subpattern.optimise(info, reverse)
        return Group(self.info, self.group, subpattern)
    
    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self
    
    def remove_captures(self):
        return self.subpattern.remove_captures()
    
    def is_atomic(self):
        return self.subpattern.is_atomic()
    
    def can_be_affix(self):
        return False
    
    def contains_group(self):
        return True
    
    def get_firstset(self, reverse):
        return self.subpattern.get_firstset(reverse)
    
    def has_simple_start(self):
        return self.subpattern.has_simple_start()
    
    def _compile(self, reverse, fuzzy):
        code = []
        public_group = private_group = self.group
        if private_group < 0:
            public_group = self.info.private_groups[private_group]
            private_group = self.info.group_count - private_group
        key = (self.group, reverse, fuzzy)
        ref = self.info.call_refs.get(key)
        if ref is not None:
            code += [(OP.CALL_REF, ref)]
        code += [(OP.GROUP, int(not reverse), private_group, public_group)]
        code += self.subpattern.compile(reverse, fuzzy)
        code += [(OP.END, )]
        if ref is not None:
            code += [(OP.END, )]
        return code
    
    def dump(self, indent, reverse):
        group = self.group
        if group < 0:
            group = private_groups[group]
        print('{}GROUP {}'.format(INDENT * indent, group))
        self.subpattern.dump(indent + 1, reverse)
    
    def __eq__(self, other):
        return (type(self) is type(other) and (self.group, self.subpattern) == (other.group, other.subpattern))
    
    def max_width(self):
        return self.subpattern.max_width()
    
    def get_required_string(self, reverse):
        return self.subpattern.get_required_string(reverse)
    
    def __del__(self):
        self.info = None



class Keep(ZeroWidthBase):
    _opcode = OP.KEEP
    _op_name = 'KEEP'



class LazyRepeat(GreedyRepeat):
    _opcode = OP.LAZY_REPEAT
    _op_name = 'LAZY_REPEAT'



class LookAround(RegexBase):
    _dir_text = {False: 'AHEAD', True: 'BEHIND'}
    
    def __init__(self, behind, positive, subpattern):
        RegexBase.__init__(self)
        self.behind = bool(behind)
        self.positive = bool(positive)
        self.subpattern = subpattern
    
    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, self.behind, fuzzy)
    
    def optimise(self, info, reverse):
        subpattern = self.subpattern.optimise(info, self.behind)
        if (self.positive and subpattern.is_empty()):
            return subpattern
        return LookAround(self.behind, self.positive, subpattern)
    
    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self
    
    def remove_captures(self):
        return self.subpattern.remove_captures()
    
    def is_atomic(self):
        return self.subpattern.is_atomic()
    
    def can_be_affix(self):
        return self.subpattern.can_be_affix()
    
    def contains_group(self):
        return self.subpattern.contains_group()
    
    def get_firstset(self, reverse):
        if (self.positive and self.behind == reverse):
            return self.subpattern.get_firstset(reverse)
        return set([None])
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if fuzzy:
            flags |= FUZZY_OP
        if reverse:
            flags |= REVERSE_OP
        return [(OP.LOOKAROUND, flags, int(not self.behind))] + self.subpattern.compile(self.behind) + [(OP.END, )]
    
    def dump(self, indent, reverse):
        print('{}LOOK{} {}'.format(INDENT * indent, self._dir_text[self.behind], POS_TEXT[self.positive]))
        self.subpattern.dump(indent + 1, self.behind)
    
    def is_empty(self):
        return (self.positive and self.subpattern.is_empty())
    
    def __eq__(self, other):
        return (type(self) is type(other) and (self.behind, self.positive, self.subpattern) == (other.behind, other.positive, other.subpattern))
    
    def max_width(self):
        return 0



class LookAroundConditional(RegexBase):
    _dir_text = {False: 'AHEAD', True: 'BEHIND'}
    
    def __init__(self, behind, positive, subpattern, yes_item, no_item):
        RegexBase.__init__(self)
        self.behind = bool(behind)
        self.positive = bool(positive)
        self.subpattern = subpattern
        self.yes_item = yes_item
        self.no_item = no_item
    
    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, fuzzy)
        self.yes_item.fix_groups(pattern, reverse, fuzzy)
        self.no_item.fix_groups(pattern, reverse, fuzzy)
    
    def optimise(self, info, reverse):
        subpattern = self.subpattern.optimise(info, self.behind)
        yes_item = self.yes_item.optimise(info, self.behind)
        no_item = self.no_item.optimise(info, self.behind)
        return LookAroundConditional(self.behind, self.positive, subpattern, yes_item, no_item)
    
    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        self.yes_item = self.yes_item.pack_characters(info)
        self.no_item = self.no_item.pack_characters(info)
        return self
    
    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        self.yes_item = self.yes_item.remove_captures()
        self.no_item = self.no_item.remove_captures()
    
    def is_atomic(self):
        return (self.subpattern.is_atomic() and self.yes_item.is_atomic() and self.no_item.is_atomic())
    
    def can_be_affix(self):
        return (self.subpattern.can_be_affix() and self.yes_item.can_be_affix() and self.no_item.can_be_affix())
    
    def contains_group(self):
        return (self.subpattern.contains_group() or self.yes_item.contains_group() or self.no_item.contains_group())
    
    def _compile(self, reverse, fuzzy):
        code = [(OP.CONDITIONAL, int(self.positive), int(not self.behind))]
        code.extend(self.subpattern.compile(self.behind, fuzzy))
        code.append((OP.NEXT, ))
        code.extend(self.yes_item.compile(reverse, fuzzy))
        add_code = self.no_item.compile(reverse, fuzzy)
        if add_code:
            code.append((OP.NEXT, ))
            code.extend(add_code)
        code.append((OP.END, ))
        return code
    
    def dump(self, indent, reverse):
        print('{}CONDITIONAL {} {}'.format(INDENT * indent, self._dir_text[self.behind], POS_TEXT[self.positive]))
        self.subpattern.dump(indent + 1, self.behind)
        print('{}EITHER'.format(INDENT * indent))
        self.yes_item.dump(indent + 1, reverse)
        if not self.no_item.is_empty():
            print('{}OR'.format(INDENT * indent))
            self.no_item.dump(indent + 1, reverse)
    
    def is_empty(self):
        return ((self.subpattern.is_empty() and self.yes_item.is_empty()) or self.no_item.is_empty())
    
    def __eq__(self, other):
        return (type(self) is type(other) and (self.subpattern, self.yes_item, self.no_item) == (other.subpattern, other.yes_item, other.no_item))
    
    def max_width(self):
        return max(self.yes_item.max_width(), self.no_item.max_width())
    
    def get_required_string(self, reverse):
        return (self.max_width(), None)



class PrecompiledCode(RegexBase):
    
    def __init__(self, code):
        self.code = code
    
    def _compile(self, reverse, fuzzy):
        return [tuple(self.code)]



class Property(RegexBase):
    _opcode = {(NOCASE, False): OP.PROPERTY, (IGNORECASE, False): OP.PROPERTY_IGN, (FULLCASE, False): OP.PROPERTY, (FULLIGNORECASE, False): OP.PROPERTY_IGN, (NOCASE, True): OP.PROPERTY_REV, (IGNORECASE, True): OP.PROPERTY_IGN_REV, (FULLCASE, True): OP.PROPERTY_REV, (FULLIGNORECASE, True): OP.PROPERTY_IGN_REV}
    
    def __init__(self, value, positive=True, case_flags=NOCASE, zerowidth=False):
        RegexBase.__init__(self)
        self.value = value
        self.positive = bool(positive)
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self.zerowidth = bool(zerowidth)
        self._key = (self.__class__, self.value, self.positive, self.case_flags, self.zerowidth)
    
    def rebuild(self, positive, case_flags, zerowidth):
        return Property(self.value, positive, case_flags, zerowidth)
    
    def optimise(self, info, reverse, in_set=False):
        return self
    
    def get_firstset(self, reverse):
        return set([self])
    
    def has_simple_start(self):
        return True
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if self.zerowidth:
            flags |= ZEROWIDTH_OP
        if fuzzy:
            flags |= FUZZY_OP
        return [(self._opcode[(self.case_flags, reverse)], flags, self.value)]
    
    def dump(self, indent, reverse):
        prop = PROPERTY_NAMES[self.value >> 16]
        (name, value) = (prop[0], prop[1][self.value & 65535])
        print('{}PROPERTY {} {}:{}{}'.format(INDENT * indent, POS_TEXT[self.positive], name, value, CASE_TEXT[self.case_flags]))
    
    def matches(self, ch):
        return _regex.has_property_value(self.value, ch) == self.positive
    
    def max_width(self):
        return 1



class Prune(ZeroWidthBase):
    _op_name = 'PRUNE'
    
    def _compile(self, reverse, fuzzy):
        return [(OP.PRUNE, )]



class Range(RegexBase):
    _opcode = {(NOCASE, False): OP.RANGE, (IGNORECASE, False): OP.RANGE_IGN, (FULLCASE, False): OP.RANGE, (FULLIGNORECASE, False): OP.RANGE_IGN, (NOCASE, True): OP.RANGE_REV, (IGNORECASE, True): OP.RANGE_IGN_REV, (FULLCASE, True): OP.RANGE_REV, (FULLIGNORECASE, True): OP.RANGE_IGN_REV}
    _op_name = 'RANGE'
    
    def __init__(self, lower, upper, positive=True, case_flags=NOCASE, zerowidth=False):
        RegexBase.__init__(self)
        self.lower = lower
        self.upper = upper
        self.positive = bool(positive)
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self.zerowidth = bool(zerowidth)
        self._key = (self.__class__, self.lower, self.upper, self.positive, self.case_flags, self.zerowidth)
    
    def rebuild(self, positive, case_flags, zerowidth):
        return Range(self.lower, self.upper, positive, case_flags, zerowidth)
    
    def optimise(self, info, reverse, in_set=False):
        if (not self.positive or not self.case_flags & IGNORECASE or in_set):
            return self
        if (not info.flags & UNICODE or self.case_flags & FULLIGNORECASE != FULLIGNORECASE):
            return self
        expanding_chars = _regex.get_expand_on_folding()
        items = []
        for ch in expanding_chars:
            if self.lower <= ord(ch) <= self.upper:
                folded = _regex.fold_case(FULL_CASE_FOLDING, ch)
                items.append(String([ord(c) for c in folded], case_flags=self.case_flags))
        if not items:
            return self
        if len(items) < self.upper - self.lower + 1:
            items.insert(0, self)
        return Branch(items)
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if self.zerowidth:
            flags |= ZEROWIDTH_OP
        if fuzzy:
            flags |= FUZZY_OP
        return [(self._opcode[(self.case_flags, reverse)], flags, self.lower, self.upper)]
    
    def dump(self, indent, reverse):
        display_lower = ascii(chr(self.lower)).lstrip('bu')
        display_upper = ascii(chr(self.upper)).lstrip('bu')
        print('{}RANGE {} {} {}{}'.format(INDENT * indent, POS_TEXT[self.positive], display_lower, display_upper, CASE_TEXT[self.case_flags]))
    
    def matches(self, ch):
        return self.lower <= ch <= self.upper == self.positive
    
    def max_width(self):
        return 1



class RefGroup(RegexBase):
    _opcode = {(NOCASE, False): OP.REF_GROUP, (IGNORECASE, False): OP.REF_GROUP_IGN, (FULLCASE, False): OP.REF_GROUP, (FULLIGNORECASE, False): OP.REF_GROUP_FLD, (NOCASE, True): OP.REF_GROUP_REV, (IGNORECASE, True): OP.REF_GROUP_IGN_REV, (FULLCASE, True): OP.REF_GROUP_REV, (FULLIGNORECASE, True): OP.REF_GROUP_FLD_REV}
    
    def __init__(self, info, group, position, case_flags=NOCASE):
        RegexBase.__init__(self)
        self.info = info
        self.group = group
        self.position = position
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self._key = (self.__class__, self.group, self.case_flags)
    
    def fix_groups(self, pattern, reverse, fuzzy):
        try:
            self.group = int(self.group)
        except ValueError:
            try:
                self.group = self.info.group_index[self.group]
            except KeyError:
                raise error('unknown group', pattern, self.position)
        if not 1 <= self.group <= self.info.group_count:
            raise error('invalid group reference', pattern, self.position)
        self._key = (self.__class__, self.group, self.case_flags)
    
    def remove_captures(self):
        raise error('group reference not allowed', pattern, self.position)
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if fuzzy:
            flags |= FUZZY_OP
        return [(self._opcode[(self.case_flags, reverse)], flags, self.group)]
    
    def dump(self, indent, reverse):
        print('{}REF_GROUP {}{}'.format(INDENT * indent, self.group, CASE_TEXT[self.case_flags]))
    
    def max_width(self):
        return UNLIMITED
    
    def __del__(self):
        self.info = None



class SearchAnchor(ZeroWidthBase):
    _opcode = OP.SEARCH_ANCHOR
    _op_name = 'SEARCH_ANCHOR'



class Sequence(RegexBase):
    
    def __init__(self, items=None):
        RegexBase.__init__(self)
        if items is None:
            items = []
        self.items = items
    
    def fix_groups(self, pattern, reverse, fuzzy):
        for s in self.items:
            s.fix_groups(pattern, reverse, fuzzy)
    
    def optimise(self, info, reverse):
        items = []
        for s in self.items:
            s = s.optimise(info, reverse)
            if isinstance(s, Sequence):
                items.extend(s.items)
            else:
                items.append(s)
        return make_sequence(items)
    
    def pack_characters(self, info):
        """Packs sequences of characters into strings."""
        items = []
        characters = []
        case_flags = NOCASE
        for s in self.items:
            if (type(s) is Character and s.positive and not s.zerowidth):
                if s.case_flags != case_flags:
                    if (s.case_flags or is_cased_i(info, s.value)):
                        Sequence._flush_characters(info, characters, case_flags, items)
                        case_flags = s.case_flags
                characters.append(s.value)
            elif (type(s) is String or type(s) is Literal):
                if s.case_flags != case_flags:
                    if (s.case_flags or any((is_cased_i(info, c) for c in characters))):
                        Sequence._flush_characters(info, characters, case_flags, items)
                        case_flags = s.case_flags
                characters.extend(s.characters)
            else:
                Sequence._flush_characters(info, characters, case_flags, items)
                items.append(s.pack_characters(info))
        Sequence._flush_characters(info, characters, case_flags, items)
        return make_sequence(items)
    
    def remove_captures(self):
        self.items = [s.remove_captures() for s in self.items]
        return self
    
    def is_atomic(self):
        return all((s.is_atomic() for s in self.items))
    
    def can_be_affix(self):
        return False
    
    def contains_group(self):
        return any((s.contains_group() for s in self.items))
    
    def get_firstset(self, reverse):
        fs = set()
        items = self.items
        if reverse:
            items.reverse()
        for s in items:
            fs |= s.get_firstset(reverse)
            if None not in fs:
                return fs
            fs.discard(None)
        return fs | set([None])
    
    def has_simple_start(self):
        return (bool(self.items) and self.items[0].has_simple_start())
    
    def _compile(self, reverse, fuzzy):
        seq = self.items
        if reverse:
            seq = seq[::-1]
        code = []
        for s in seq:
            code.extend(s.compile(reverse, fuzzy))
        return code
    
    def dump(self, indent, reverse):
        for s in self.items:
            s.dump(indent, reverse)
    
    @staticmethod
    def _flush_characters(info, characters, case_flags, items):
        if not characters:
            return
        if case_flags & IGNORECASE:
            if not any((is_cased_i(info, c) for c in characters)):
                case_flags = NOCASE
        if case_flags & FULLIGNORECASE == FULLIGNORECASE:
            literals = Sequence._fix_full_casefold(characters)
            for item in literals:
                chars = item.characters
                if len(chars) == 1:
                    items.append(Character(chars[0], case_flags=item.case_flags))
                else:
                    items.append(String(chars, case_flags=item.case_flags))
        elif len(characters) == 1:
            items.append(Character(characters[0], case_flags=case_flags))
        else:
            items.append(String(characters, case_flags=case_flags))
        characters[:] = []
    
    @staticmethod
    def _fix_full_casefold(characters):
        expanded = [_regex.fold_case(FULL_CASE_FOLDING, c) for c in _regex.get_expand_on_folding()]
        string = _regex.fold_case(FULL_CASE_FOLDING, ''.join((chr(c) for c in characters))).lower()
        chunks = []
        for e in expanded:
            found = string.find(e)
            while found >= 0:
                chunks.append((found, found + len(e)))
                found = string.find(e, found + 1)
        pos = 0
        literals = []
        for (start, end) in Sequence._merge_chunks(chunks):
            if pos < start:
                literals.append(Literal(characters[pos:start], case_flags=IGNORECASE))
            literals.append(Literal(characters[start:end], case_flags=FULLIGNORECASE))
            pos = end
        if pos < len(characters):
            literals.append(Literal(characters[pos:], case_flags=IGNORECASE))
        return literals
    
    @staticmethod
    def _merge_chunks(chunks):
        if len(chunks) < 2:
            return chunks
        chunks.sort()
        (start, end) = chunks[0]
        new_chunks = []
        for (s, e) in chunks[1:]:
            if s <= end:
                end = max(end, e)
            else:
                new_chunks.append((start, end))
                (start, end) = (s, e)
        new_chunks.append((start, end))
        return new_chunks
    
    def is_empty(self):
        return all((i.is_empty() for i in self.items))
    
    def __eq__(self, other):
        return (type(self) is type(other) and self.items == other.items)
    
    def max_width(self):
        return sum((s.max_width() for s in self.items))
    
    def get_required_string(self, reverse):
        seq = self.items
        if reverse:
            seq = seq[::-1]
        offset = 0
        for s in seq:
            (ofs, req) = s.get_required_string(reverse)
            offset += ofs
            if req:
                return (offset, req)
        return (offset, None)



class SetBase(RegexBase):
    
    def __init__(self, info, items, positive=True, case_flags=NOCASE, zerowidth=False):
        RegexBase.__init__(self)
        self.info = info
        self.items = tuple(items)
        self.positive = bool(positive)
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self.zerowidth = bool(zerowidth)
        self.char_width = 1
        self._key = (self.__class__, self.items, self.positive, self.case_flags, self.zerowidth)
    
    def rebuild(self, positive, case_flags, zerowidth):
        return type(self)(self.info, self.items, positive, case_flags, zerowidth).optimise(self.info, False)
    
    def get_firstset(self, reverse):
        return set([self])
    
    def has_simple_start(self):
        return True
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if self.zerowidth:
            flags |= ZEROWIDTH_OP
        if fuzzy:
            flags |= FUZZY_OP
        code = [(self._opcode[(self.case_flags, reverse)], flags)]
        for m in self.items:
            code.extend(m.compile())
        code.append((OP.END, ))
        return code
    
    def dump(self, indent, reverse):
        print('{}{} {}{}'.format(INDENT * indent, self._op_name, POS_TEXT[self.positive], CASE_TEXT[self.case_flags]))
        for i in self.items:
            i.dump(indent + 1, reverse)
    
    def _handle_case_folding(self, info, in_set):
        if (not self.positive or not self.case_flags & IGNORECASE or in_set):
            return self
        if (not self.info.flags & UNICODE or self.case_flags & FULLIGNORECASE != FULLIGNORECASE):
            return self
        expanding_chars = _regex.get_expand_on_folding()
        items = []
        seen = set()
        for ch in expanding_chars:
            if self.matches(ord(ch)):
                folded = _regex.fold_case(FULL_CASE_FOLDING, ch)
                if folded not in seen:
                    items.append(String([ord(c) for c in folded], case_flags=self.case_flags))
                    seen.add(folded)
        if not items:
            return self
        return Branch([self] + items)
    
    def max_width(self):
        if (not self.positive or not self.case_flags & IGNORECASE):
            return 1
        if (not self.info.flags & UNICODE or self.case_flags & FULLIGNORECASE != FULLIGNORECASE):
            return 1
        expanding_chars = _regex.get_expand_on_folding()
        seen = set()
        for ch in expanding_chars:
            if self.matches(ord(ch)):
                folded = _regex.fold_case(FULL_CASE_FOLDING, ch)
                seen.add(folded)
        if not seen:
            return 1
        return max((len(folded) for folded in seen))
    
    def __del__(self):
        self.info = None



class SetDiff(SetBase):
    _opcode = {(NOCASE, False): OP.SET_DIFF, (IGNORECASE, False): OP.SET_DIFF_IGN, (FULLCASE, False): OP.SET_DIFF, (FULLIGNORECASE, False): OP.SET_DIFF_IGN, (NOCASE, True): OP.SET_DIFF_REV, (IGNORECASE, True): OP.SET_DIFF_IGN_REV, (FULLCASE, True): OP.SET_DIFF_REV, (FULLIGNORECASE, True): OP.SET_DIFF_IGN_REV}
    _op_name = 'SET_DIFF'
    
    def optimise(self, info, reverse, in_set=False):
        items = self.items
        if len(items) > 2:
            items = [items[0], SetUnion(info, items[1:])]
        if len(items) == 1:
            return items[0].with_flags(case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple((m.optimise(info, reverse, in_set=True) for m in items))
        return self._handle_case_folding(info, in_set)
    
    def matches(self, ch):
        m = (self.items[0].matches(ch) and not self.items[1].matches(ch))
        return m == self.positive



class SetInter(SetBase):
    _opcode = {(NOCASE, False): OP.SET_INTER, (IGNORECASE, False): OP.SET_INTER_IGN, (FULLCASE, False): OP.SET_INTER, (FULLIGNORECASE, False): OP.SET_INTER_IGN, (NOCASE, True): OP.SET_INTER_REV, (IGNORECASE, True): OP.SET_INTER_IGN_REV, (FULLCASE, True): OP.SET_INTER_REV, (FULLIGNORECASE, True): OP.SET_INTER_IGN_REV}
    _op_name = 'SET_INTER'
    
    def optimise(self, info, reverse, in_set=False):
        items = []
        for m in self.items:
            m = m.optimise(info, reverse, in_set=True)
            if (isinstance(m, SetInter) and m.positive):
                items.extend(m.items)
            else:
                items.append(m)
        if len(items) == 1:
            return items[0].with_flags(case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple(items)
        return self._handle_case_folding(info, in_set)
    
    def matches(self, ch):
        m = all((i.matches(ch) for i in self.items))
        return m == self.positive



class SetSymDiff(SetBase):
    _opcode = {(NOCASE, False): OP.SET_SYM_DIFF, (IGNORECASE, False): OP.SET_SYM_DIFF_IGN, (FULLCASE, False): OP.SET_SYM_DIFF, (FULLIGNORECASE, False): OP.SET_SYM_DIFF_IGN, (NOCASE, True): OP.SET_SYM_DIFF_REV, (IGNORECASE, True): OP.SET_SYM_DIFF_IGN_REV, (FULLCASE, True): OP.SET_SYM_DIFF_REV, (FULLIGNORECASE, True): OP.SET_SYM_DIFF_IGN_REV}
    _op_name = 'SET_SYM_DIFF'
    
    def optimise(self, info, reverse, in_set=False):
        items = []
        for m in self.items:
            m = m.optimise(info, reverse, in_set=True)
            if (isinstance(m, SetSymDiff) and m.positive):
                items.extend(m.items)
            else:
                items.append(m)
        if len(items) == 1:
            return items[0].with_flags(case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple(items)
        return self._handle_case_folding(info, in_set)
    
    def matches(self, ch):
        m = False
        for i in self.items:
            m = m != i.matches(ch)
        return m == self.positive



class SetUnion(SetBase):
    _opcode = {(NOCASE, False): OP.SET_UNION, (IGNORECASE, False): OP.SET_UNION_IGN, (FULLCASE, False): OP.SET_UNION, (FULLIGNORECASE, False): OP.SET_UNION_IGN, (NOCASE, True): OP.SET_UNION_REV, (IGNORECASE, True): OP.SET_UNION_IGN_REV, (FULLCASE, True): OP.SET_UNION_REV, (FULLIGNORECASE, True): OP.SET_UNION_IGN_REV}
    _op_name = 'SET_UNION'
    
    def optimise(self, info, reverse, in_set=False):
        items = []
        for m in self.items:
            m = m.optimise(info, reverse, in_set=True)
            if (isinstance(m, SetUnion) and m.positive):
                items.extend(m.items)
            else:
                items.append(m)
        if len(items) == 1:
            i = items[0]
            return i.with_flags(positive=i.positive == self.positive, case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple(items)
        return self._handle_case_folding(info, in_set)
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if self.zerowidth:
            flags |= ZEROWIDTH_OP
        if fuzzy:
            flags |= FUZZY_OP
        (characters, others) = (defaultdict(list), [])
        for m in self.items:
            if isinstance(m, Character):
                characters[m.positive].append(m.value)
            else:
                others.append(m)
        code = [(self._opcode[(self.case_flags, reverse)], flags)]
        for (positive, values) in characters.items():
            flags = 0
            if positive:
                flags |= POSITIVE_OP
            if len(values) == 1:
                code.append((OP.CHARACTER, flags, values[0]))
            else:
                code.append((OP.STRING, flags, len(values)) + tuple(values))
        for m in others:
            code.extend(m.compile())
        code.append((OP.END, ))
        return code
    
    def matches(self, ch):
        m = any((i.matches(ch) for i in self.items))
        return m == self.positive



class Skip(ZeroWidthBase):
    _op_name = 'SKIP'
    _opcode = OP.SKIP



class StartOfLine(ZeroWidthBase):
    _opcode = OP.START_OF_LINE
    _op_name = 'START_OF_LINE'



class StartOfLineU(StartOfLine):
    _opcode = OP.START_OF_LINE_U
    _op_name = 'START_OF_LINE_U'



class StartOfString(ZeroWidthBase):
    _opcode = OP.START_OF_STRING
    _op_name = 'START_OF_STRING'



class StartOfWord(ZeroWidthBase):
    _opcode = OP.START_OF_WORD
    _op_name = 'START_OF_WORD'



class String(RegexBase):
    _opcode = {(NOCASE, False): OP.STRING, (IGNORECASE, False): OP.STRING_IGN, (FULLCASE, False): OP.STRING, (FULLIGNORECASE, False): OP.STRING_FLD, (NOCASE, True): OP.STRING_REV, (IGNORECASE, True): OP.STRING_IGN_REV, (FULLCASE, True): OP.STRING_REV, (FULLIGNORECASE, True): OP.STRING_FLD_REV}
    
    def __init__(self, characters, case_flags=NOCASE):
        self.characters = tuple(characters)
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        if self.case_flags & FULLIGNORECASE == FULLIGNORECASE:
            folded_characters = []
            for char in self.characters:
                folded = _regex.fold_case(FULL_CASE_FOLDING, chr(char))
                folded_characters.extend((ord(c) for c in folded))
        else:
            folded_characters = self.characters
        self.folded_characters = tuple(folded_characters)
        self.required = False
        self._key = (self.__class__, self.characters, self.case_flags)
    
    def get_firstset(self, reverse):
        if reverse:
            pos = -1
        else:
            pos = 0
        return set([Character(self.characters[pos], case_flags=self.case_flags)])
    
    def has_simple_start(self):
        return True
    
    def _compile(self, reverse, fuzzy):
        flags = 0
        if fuzzy:
            flags |= FUZZY_OP
        if self.required:
            flags |= REQUIRED_OP
        return [(self._opcode[(self.case_flags, reverse)], flags, len(self.folded_characters)) + self.folded_characters]
    
    def dump(self, indent, reverse):
        display = ascii(''.join((chr(c) for c in self.characters))).lstrip('bu')
        print('{}STRING {}{}'.format(INDENT * indent, display, CASE_TEXT[self.case_flags]))
    
    def max_width(self):
        return len(self.folded_characters)
    
    def get_required_string(self, reverse):
        return (0, self)



class Literal(String):
    
    def dump(self, indent, reverse):
        literal = ''.join((chr(c) for c in self.characters))
        display = ascii(literal).lstrip('bu')
        print('{}LITERAL MATCH {}{}'.format(INDENT * indent, display, CASE_TEXT[self.case_flags]))



class StringSet(Branch):
    
    def __init__(self, info, name, case_flags=NOCASE):
        self.info = info
        self.name = name
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self._key = (self.__class__, self.name, self.case_flags)
        self.set_key = (name, self.case_flags)
        if self.set_key not in info.named_lists_used:
            info.named_lists_used[self.set_key] = len(info.named_lists_used)
        index = self.info.named_lists_used[self.set_key]
        items = self.info.kwargs[self.name]
        case_flags = self.case_flags
        encoding = self.info.flags & _ALL_ENCODINGS
        fold_flags = encoding | case_flags
        choices = []
        for string in items:
            if isinstance(string, str):
                string = [ord(c) for c in string]
            choices.append([Character(c, case_flags=case_flags) for c in string])
        choices.sort(key=len, reverse=True)
        self.branches = [Sequence(choice) for choice in choices]
    
    def dump(self, indent, reverse):
        print('{}STRING_SET {}{}'.format(INDENT * indent, self.name, CASE_TEXT[self.case_flags]))
    
    def __del__(self):
        self.info = None



class Source:
    """Scanner for the regular expression source string."""
    
    def __init__(self, string):
        if isinstance(string, str):
            self.string = string
            self.char_type = chr
        else:
            self.string = string.decode('latin-1')
            self.char_type = lambda c: bytes([c])
        self.pos = 0
        self.ignore_space = False
        self.sep = string[:0]
    
    def get(self, override_ignore=False):
        string = self.string
        pos = self.pos
        try:
            if (self.ignore_space and not override_ignore):
                while True:
                    if string[pos].isspace():
                        pos += 1
                    elif string[pos] == '#':
                        pos = string.index('\n', pos)
                    else:
                        break
            ch = string[pos]
            self.pos = pos + 1
            return ch
        except IndexError:
            self.pos = pos
            return string[:0]
        except ValueError:
            self.pos = len(string)
            return string[:0]
    
    def get_many(self, count=1):
        string = self.string
        pos = self.pos
        try:
            if self.ignore_space:
                substring = []
                while len(substring) < count:
                    while True:
                        if string[pos].isspace():
                            pos += 1
                        elif string[pos] == '#':
                            pos = string.index('\n', pos)
                        else:
                            break
                    substring.append(string[pos])
                    pos += 1
                substring = ''.join(substring)
            else:
                substring = string[pos:pos + count]
                pos += len(substring)
            self.pos = pos
            return substring
        except IndexError:
            self.pos = len(string)
            return ''.join(substring)
        except ValueError:
            self.pos = len(string)
            return ''.join(substring)
    
    def get_while(self, test_set, include=True):
        string = self.string
        pos = self.pos
        if self.ignore_space:
            try:
                substring = []
                while True:
                    if string[pos].isspace():
                        pos += 1
                    elif string[pos] == '#':
                        pos = string.index('\n', pos)
                    elif string[pos] in test_set == include:
                        substring.append(string[pos])
                        pos += 1
                    else:
                        break
                self.pos = pos
            except IndexError:
                self.pos = len(string)
            except ValueError:
                self.pos = len(string)
            return ''.join(substring)
        else:
            try:
                while string[pos] in test_set == include:
                    pos += 1
                substring = string[self.pos:pos]
                self.pos = pos
                return substring
            except IndexError:
                substring = string[self.pos:pos]
                self.pos = pos
                return substring
    
    def skip_while(self, test_set, include=True):
        string = self.string
        pos = self.pos
        try:
            if self.ignore_space:
                while True:
                    if string[pos].isspace():
                        pos += 1
                    elif string[pos] == '#':
                        pos = string.index('\n', pos)
                    elif string[pos] in test_set == include:
                        pos += 1
                    else:
                        break
            else:
                while string[pos] in test_set == include:
                    pos += 1
            self.pos = pos
        except IndexError:
            self.pos = len(string)
        except ValueError:
            self.pos = len(string)
    
    def match(self, substring):
        string = self.string
        pos = self.pos
        if self.ignore_space:
            try:
                for c in substring:
                    while True:
                        if string[pos].isspace():
                            pos += 1
                        elif string[pos] == '#':
                            pos = string.index('\n', pos)
                        else:
                            break
                    if string[pos] != c:
                        return False
                    pos += 1
                self.pos = pos
                return True
            except IndexError:
                return False
            except ValueError:
                return False
        else:
            if not string.startswith(substring, pos):
                return False
            self.pos = pos + len(substring)
            return True
    
    def expect(self, substring):
        if not self.match(substring):
            raise error('missing {}'.format(substring), self.string, self.pos)
    
    def at_end(self):
        string = self.string
        pos = self.pos
        try:
            if self.ignore_space:
                while True:
                    if string[pos].isspace():
                        pos += 1
                    elif string[pos] == '#':
                        pos = string.index('\n', pos)
                    else:
                        break
            return pos >= len(string)
        except IndexError:
            return True
        except ValueError:
            return True



class Info:
    """Info about the regular expression."""
    
    def __init__(self, flags=0, char_type=None, kwargs={}):
        flags |= DEFAULT_FLAGS[(flags & _ALL_VERSIONS or DEFAULT_VERSION)]
        self.flags = flags
        self.global_flags = flags
        self.inline_locale = False
        self.kwargs = kwargs
        self.group_count = 0
        self.group_index = {}
        self.group_name = {}
        self.char_type = char_type
        self.named_lists_used = {}
        self.open_groups = []
        self.open_group_count = {}
        self.defined_groups = {}
        self.group_calls = []
        self.private_groups = {}
    
    def open_group(self, name=None):
        group = self.group_index.get(name)
        if group is None:
            while True:
                self.group_count += 1
                if (name is None or self.group_count not in self.group_name):
                    break
            group = self.group_count
            if name:
                self.group_index[name] = group
                self.group_name[group] = name
        if group in self.open_groups:
            group_alias = -(len(self.private_groups) + 1)
            self.private_groups[group_alias] = group
            group = group_alias
        self.open_groups.append(group)
        self.open_group_count[group] = self.open_group_count.get(group, 0) + 1
        return group
    
    def close_group(self):
        self.open_groups.pop()
    
    def is_open_group(self, name):
        version = (self.flags & _ALL_VERSIONS or DEFAULT_VERSION)
        if version == VERSION1:
            return False
        if name.isdigit():
            group = int(name)
        else:
            group = self.group_index.get(name)
        return group in self.open_groups


def _check_group_features(info, parsed):
    """Checks whether the reverse and fuzzy features of the group calls match
    the groups which they call.
    """
    call_refs = {}
    additional_groups = []
    for (call, reverse, fuzzy) in info.group_calls:
        key = (call.group, reverse, fuzzy)
        ref = call_refs.get(key)
        if ref is None:
            if call.group == 0:
                rev = bool(info.flags & REVERSE)
                fuz = isinstance(parsed, Fuzzy)
                if (rev, fuz) != (reverse, fuzzy):
                    additional_groups.append((CallRef(len(call_refs), parsed), reverse, fuzzy))
            else:
                def_info = info.defined_groups[call.group]
                group = def_info[0]
                if def_info[1:] != (reverse, fuzzy):
                    additional_groups.append((group, reverse, fuzzy))
            ref = len(call_refs)
            call_refs[key] = ref
        call.call_ref = ref
    info.call_refs = call_refs
    info.additional_groups = additional_groups

def _get_required_string(parsed, flags):
    """Gets the required string and related info of a parsed pattern."""
    (req_offset, required) = parsed.get_required_string(bool(flags & REVERSE))
    if required:
        required.required = True
        if req_offset >= UNLIMITED:
            req_offset = -1
        req_flags = required.case_flags
        if not flags & UNICODE:
            req_flags &= ~UNICODE
        req_chars = required.folded_characters
    else:
        req_offset = 0
        req_chars = ()
        req_flags = 0
    return (req_offset, req_chars, req_flags)


class Scanner:
    
    def __init__(self, lexicon, flags=0):
        self.lexicon = lexicon
        patterns = []
        for (phrase, action) in lexicon:
            source = Source(phrase)
            info = Info(flags, source.char_type)
            source.ignore_space = bool(info.flags & VERBOSE)
            parsed = _parse_pattern(source, info)
            if not source.at_end():
                raise error('unbalanced parenthesis', source.string, source.pos)
            patterns.append(parsed.remove_captures())
        info = Info(flags)
        patterns = [Group(info, g + 1, p) for (g, p) in enumerate(patterns)]
        parsed = Branch(patterns)
        reverse = bool(info.flags & REVERSE)
        parsed = parsed.optimise(info, reverse)
        parsed = parsed.pack_characters(info)
        (req_offset, req_chars, req_flags) = _get_required_string(parsed, info.flags)
        _check_group_features(info, parsed)
        if info.call_refs:
            raise error('recursive regex not supported by Scanner', source.string, source.pos)
        reverse = bool(info.flags & REVERSE)
        code = parsed.compile(reverse) + [(OP.SUCCESS, )]
        code = _flatten_code(code)
        if not parsed.has_simple_start():
            try:
                fs_code = _compile_firstset(info, parsed.get_firstset(reverse))
                fs_code = _flatten_code(fs_code)
                code = fs_code + code
            except _FirstSetError:
                pass
        version = (info.flags & _ALL_VERSIONS or DEFAULT_VERSION)
        if version not in (0, VERSION0, VERSION1):
            raise ValueError('VERSION0 and VERSION1 flags are mutually incompatible')
        self.scanner = _regex.compile(None, flags & GLOBAL_FLAGS | version, code, {}, {}, {}, [], req_offset, req_chars, req_flags, len(patterns))
    
    def scan(self, string):
        result = []
        append = result.append
        match = self.scanner.scanner(string).match
        i = 0
        while True:
            m = match()
            if not m:
                break
            j = m.end()
            if i == j:
                break
            action = self.lexicon[m.lastindex - 1][1]
            if hasattr(action, '__call__'):
                self.match = m
                action = action(self, m.group())
            if action is not None:
                append(action)
            i = j
        return (result, string[i:])

PROPERTIES = _regex.get_properties()
PROPERTY_NAMES = {}
for (prop_name, (prop_id, values)) in PROPERTIES.items():
    (name, prop_values) = PROPERTY_NAMES.get(prop_id, ('', {}))
    name = max(name, prop_name, key=len)
    PROPERTY_NAMES[prop_id] = (name, prop_values)
    for (val_name, val_id) in values.items():
        prop_values[val_id] = max(prop_values.get(val_id, ''), val_name, key=len)
CHARACTER_ESCAPES = {'a': '\x07', 'b': '\x08', 'f': '\x0c', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\x0b'}
CHARSET_ESCAPES = {'d': lookup_property(None, 'Digit', True), 'D': lookup_property(None, 'Digit', False), 'h': lookup_property(None, 'Blank', True), 's': lookup_property(None, 'Space', True), 'S': lookup_property(None, 'Space', False), 'w': lookup_property(None, 'Word', True), 'W': lookup_property(None, 'Word', False)}
POSITION_ESCAPES = {'A': StartOfString(), 'b': Boundary(), 'B': Boundary(False), 'K': Keep(), 'm': StartOfWord(), 'M': EndOfWord(), 'Z': EndOfString()}
WORD_POSITION_ESCAPES = dict(POSITION_ESCAPES)
WORD_POSITION_ESCAPES.update({'b': DefaultBoundary(), 'B': DefaultBoundary(False), 'm': DefaultStartOfWord(), 'M': DefaultEndOfWord()})
VERBS = {'FAIL': Failure(), 'F': Failure(), 'PRUNE': Prune(), 'SKIP': Skip()}

