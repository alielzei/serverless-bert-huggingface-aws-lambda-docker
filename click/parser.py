"""
This module started out as largely a copy paste from the stdlib's
optparse module with the features removed that we do not need from
optparse because we implement them in Click on a higher level (for
instance type handling, help formatting and a lot more).

The plan is to remove more and more from here over time.

The reason this is a different module and not optparse from the stdlib
is that there are differences in 2.x and 3.x about the error messages
generated and optparse in the stdlib uses gettext for no good reason
and might cause us issues.

Click uses parts of optparse written by Gregory P. Ward and maintained
by the Python Software Foundation. This is limited to code in parser.py.

Copyright 2001-2006 Gregory P. Ward. All rights reserved.
Copyright 2002-2006 Python Software Foundation. All rights reserved.
"""

import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
if t.TYPE_CHECKING:
    import typing_extensions as te
    from .core import Argument as CoreArgument
    from .core import Context
    from .core import Option as CoreOption
    from .core import Parameter as CoreParameter
V = t.TypeVar('V')
_flag_needs_value = object()

def _unpack_args(args: t.Sequence[str], nargs_spec: t.Sequence[int]) -> t.Tuple[(t.Sequence[t.Union[(str, t.Sequence[t.Optional[str]], None)]], t.List[str])]:
    """Given an iterable of arguments and an iterable of nargs specifications,
    it returns a tuple with all the unpacked arguments at the first index
    and all remaining arguments as the second.

    The nargs specification is the number of arguments that should be consumed
    or `-1` to indicate that this position should eat up all the remainders.

    Missing items are filled with `None`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.parser._unpack_args', '_unpack_args(args, nargs_spec)', {'deque': deque, 't': t, 'V': V, 'args': args, 'nargs_spec': nargs_spec, 't': t, 'str': str, 't': t, 'int': int, 't': t}, 1)

def split_opt(opt: str) -> t.Tuple[(str, str)]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.parser.split_opt', 'split_opt(opt)', {'opt': opt, 't': t}, 2)

def normalize_opt(opt: str, ctx: t.Optional['Context']) -> str:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.parser.normalize_opt', 'normalize_opt(opt, ctx)', {'split_opt': split_opt, 'opt': opt, 'ctx': ctx, 't': t}, 1)

def split_arg_string(string: str) -> t.List[str]:
    """Split an argument string as with :func:`shlex.split`, but don't
    fail if the string is incomplete. Ignores a missing closing quote or
    incomplete escape sequence and uses the partial token as-is.

    .. code-block:: python

        split_arg_string("example 'my file")
        ["example", "my file"]

        split_arg_string("example my")
        ["example", "my"]

    :param string: String to split.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.parser.split_arg_string', 'split_arg_string(string)', {'string': string, 't': t, 'str': str}, 1)


class Option:
    
    def __init__(self, obj: 'CoreOption', opts: t.Sequence[str], dest: t.Optional[str], action: t.Optional[str] = None, nargs: int = 1, const: t.Optional[t.Any] = None):
        self._short_opts = []
        self._long_opts = []
        self.prefixes: t.Set[str] = set()
        for opt in opts:
            (prefix, value) = split_opt(opt)
            if not prefix:
                raise ValueError(f'Invalid start character for option ({opt})')
            self.prefixes.add(prefix[0])
            if (len(prefix) == 1 and len(value) == 1):
                self._short_opts.append(opt)
            else:
                self._long_opts.append(opt)
                self.prefixes.add(prefix)
        if action is None:
            action = 'store'
        self.dest = dest
        self.action = action
        self.nargs = nargs
        self.const = const
        self.obj = obj
    
    @property
    def takes_value(self) -> bool:
        return self.action in ('store', 'append')
    
    def process(self, value: t.Any, state: 'ParsingState') -> None:
        if self.action == 'store':
            state.opts[self.dest] = value
        elif self.action == 'store_const':
            state.opts[self.dest] = self.const
        elif self.action == 'append':
            state.opts.setdefault(self.dest, []).append(value)
        elif self.action == 'append_const':
            state.opts.setdefault(self.dest, []).append(self.const)
        elif self.action == 'count':
            state.opts[self.dest] = state.opts.get(self.dest, 0) + 1
        else:
            raise ValueError(f"unknown action '{self.action}'")
        state.order.append(self.obj)



class Argument:
    
    def __init__(self, obj: 'CoreArgument', dest: t.Optional[str], nargs: int = 1):
        self.dest = dest
        self.nargs = nargs
        self.obj = obj
    
    def process(self, value: t.Union[(t.Optional[str], t.Sequence[t.Optional[str]])], state: 'ParsingState') -> None:
        if self.nargs > 1:
            assert value is not None
            holes = sum((1 for x in value if x is None))
            if holes == len(value):
                value = None
            elif holes != 0:
                raise BadArgumentUsage(_('Argument {name!r} takes {nargs} values.').format(name=self.dest, nargs=self.nargs))
        if (self.nargs == -1 and self.obj.envvar is not None and value == ()):
            value = None
        state.opts[self.dest] = value
        state.order.append(self.obj)



class ParsingState:
    
    def __init__(self, rargs: t.List[str]) -> None:
        self.opts: t.Dict[(str, t.Any)] = {}
        self.largs: t.List[str] = []
        self.rargs = rargs
        self.order: t.List['CoreParameter'] = []



class OptionParser:
    """The option parser is an internal class that is ultimately used to
    parse options and arguments.  It's modelled after optparse and brings
    a similar but vastly simplified API.  It should generally not be used
    directly as the high level Click classes wrap it for you.

    It's not nearly as extensible as optparse or argparse as it does not
    implement features that are implemented on a higher level (such as
    types or defaults).

    :param ctx: optionally the :class:`~click.Context` where this parser
                should go with.
    """
    
    def __init__(self, ctx: t.Optional['Context'] = None) -> None:
        self.ctx = ctx
        self.allow_interspersed_args: bool = True
        self.ignore_unknown_options: bool = False
        if ctx is not None:
            self.allow_interspersed_args = ctx.allow_interspersed_args
            self.ignore_unknown_options = ctx.ignore_unknown_options
        self._short_opt: t.Dict[(str, Option)] = {}
        self._long_opt: t.Dict[(str, Option)] = {}
        self._opt_prefixes = {'-', '--'}
        self._args: t.List[Argument] = []
    
    def add_option(self, obj: 'CoreOption', opts: t.Sequence[str], dest: t.Optional[str], action: t.Optional[str] = None, nargs: int = 1, const: t.Optional[t.Any] = None) -> None:
        """Adds a new option named `dest` to the parser.  The destination
        is not inferred (unlike with optparse) and needs to be explicitly
        provided.  Action can be any of ``store``, ``store_const``,
        ``append``, ``append_const`` or ``count``.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        opts = [normalize_opt(opt, self.ctx) for opt in opts]
        option = Option(obj, opts, dest, action=action, nargs=nargs, const=const)
        self._opt_prefixes.update(option.prefixes)
        for opt in option._short_opts:
            self._short_opt[opt] = option
        for opt in option._long_opts:
            self._long_opt[opt] = option
    
    def add_argument(self, obj: 'CoreArgument', dest: t.Optional[str], nargs: int = 1) -> None:
        """Adds a positional argument named `dest` to the parser.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        self._args.append(Argument(obj, dest=dest, nargs=nargs))
    
    def parse_args(self, args: t.List[str]) -> t.Tuple[(t.Dict[(str, t.Any)], t.List[str], t.List['CoreParameter'])]:
        """Parses positional arguments and returns ``(values, args, order)``
        for the parsed options and arguments as well as the leftover
        arguments if there are any.  The order is a list of objects as they
        appear on the command line.  If arguments appear multiple times they
        will be memorized multiple times as well.
        """
        state = ParsingState(args)
        try:
            self._process_args_for_options(state)
            self._process_args_for_args(state)
        except UsageError:
            if (self.ctx is None or not self.ctx.resilient_parsing):
                raise
        return (state.opts, state.largs, state.order)
    
    def _process_args_for_args(self, state: ParsingState) -> None:
        (pargs, args) = _unpack_args(state.largs + state.rargs, [x.nargs for x in self._args])
        for (idx, arg) in enumerate(self._args):
            arg.process(pargs[idx], state)
        state.largs = args
        state.rargs = []
    
    def _process_args_for_options(self, state: ParsingState) -> None:
        while state.rargs:
            arg = state.rargs.pop(0)
            arglen = len(arg)
            if arg == '--':
                return
            elif (arg[:1] in self._opt_prefixes and arglen > 1):
                self._process_opts(arg, state)
            elif self.allow_interspersed_args:
                state.largs.append(arg)
            else:
                state.rargs.insert(0, arg)
                return
    
    def _match_long_opt(self, opt: str, explicit_value: t.Optional[str], state: ParsingState) -> None:
        if opt not in self._long_opt:
            from difflib import get_close_matches
            possibilities = get_close_matches(opt, self._long_opt)
            raise NoSuchOption(opt, possibilities=possibilities, ctx=self.ctx)
        option = self._long_opt[opt]
        if option.takes_value:
            if explicit_value is not None:
                state.rargs.insert(0, explicit_value)
            value = self._get_value_from_state(opt, option, state)
        elif explicit_value is not None:
            raise BadOptionUsage(opt, _('Option {name!r} does not take a value.').format(name=opt))
        else:
            value = None
        option.process(value, state)
    
    def _match_short_opt(self, arg: str, state: ParsingState) -> None:
        stop = False
        i = 1
        prefix = arg[0]
        unknown_options = []
        for ch in arg[1:]:
            opt = normalize_opt(f'{prefix}{ch}', self.ctx)
            option = self._short_opt.get(opt)
            i += 1
            if not option:
                if self.ignore_unknown_options:
                    unknown_options.append(ch)
                    continue
                raise NoSuchOption(opt, ctx=self.ctx)
            if option.takes_value:
                if i < len(arg):
                    state.rargs.insert(0, arg[i:])
                    stop = True
                value = self._get_value_from_state(opt, option, state)
            else:
                value = None
            option.process(value, state)
            if stop:
                break
        if (self.ignore_unknown_options and unknown_options):
            state.largs.append(f"{prefix}{''.join(unknown_options)}")
    
    def _get_value_from_state(self, option_name: str, option: Option, state: ParsingState) -> t.Any:
        nargs = option.nargs
        if len(state.rargs) < nargs:
            if option.obj._flag_needs_value:
                value = _flag_needs_value
            else:
                raise BadOptionUsage(option_name, ngettext('Option {name!r} requires an argument.', 'Option {name!r} requires {nargs} arguments.', nargs).format(name=option_name, nargs=nargs))
        elif nargs == 1:
            next_rarg = state.rargs[0]
            if (option.obj._flag_needs_value and isinstance(next_rarg, str) and next_rarg[:1] in self._opt_prefixes and len(next_rarg) > 1):
                value = _flag_needs_value
            else:
                value = state.rargs.pop(0)
        else:
            value = tuple(state.rargs[:nargs])
            del state.rargs[:nargs]
        return value
    
    def _process_opts(self, arg: str, state: ParsingState) -> None:
        explicit_value = None
        if '=' in arg:
            (long_opt, explicit_value) = arg.split('=', 1)
        else:
            long_opt = arg
        norm_long_opt = normalize_opt(long_opt, self.ctx)
        try:
            self._match_long_opt(norm_long_opt, explicit_value, state)
        except NoSuchOption:
            if arg[:2] not in self._opt_prefixes:
                self._match_short_opt(arg, state)
                return
            if not self.ignore_unknown_options:
                raise
            state.largs.append(arg)


