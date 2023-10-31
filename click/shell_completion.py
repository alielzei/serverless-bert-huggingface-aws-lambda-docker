import os
import re
import typing as t
from gettext import gettext as _
from .core import Argument
from .core import BaseCommand
from .core import Context
from .core import MultiCommand
from .core import Option
from .core import Parameter
from .core import ParameterSource
from .parser import split_arg_string
from .utils import echo

def shell_complete(cli: BaseCommand, ctx_args: t.MutableMapping[(str, t.Any)], prog_name: str, complete_var: str, instruction: str) -> int:
    """Perform shell completion for the given CLI program.

    :param cli: Command being called.
    :param ctx_args: Extra arguments to pass to
        ``cli.make_context``.
    :param prog_name: Name of the executable in the shell.
    :param complete_var: Name of the environment variable that holds
        the completion instruction.
    :param instruction: Value of ``complete_var`` with the completion
        instruction and shell, in the form ``instruction_shell``.
    :return: Status code to exit with.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.shell_completion.shell_complete', 'shell_complete(cli, ctx_args, prog_name, complete_var, instruction)', {'get_completion_class': get_completion_class, 'echo': echo, 'cli': cli, 'ctx_args': ctx_args, 'prog_name': prog_name, 'complete_var': complete_var, 'instruction': instruction, 't': t}, 1)


class CompletionItem:
    """Represents a completion value and metadata about the value. The
    default metadata is ``type`` to indicate special shell handling,
    and ``help`` if a shell supports showing a help string next to the
    value.

    Arbitrary parameters can be passed when creating the object, and
    accessed using ``item.attr``. If an attribute wasn't passed,
    accessing it returns ``None``.

    :param value: The completion suggestion.
    :param type: Tells the shell script to provide special completion
        support for the type. Click uses ``"dir"`` and ``"file"``.
    :param help: String shown next to the value if supported.
    :param kwargs: Arbitrary metadata. The built-in implementations
        don't use this, but custom type completions paired with custom
        shell support could use it.
    """
    __slots__ = ('value', 'type', 'help', '_info')
    
    def __init__(self, value: t.Any, type: str = 'plain', help: t.Optional[str] = None, **kwargs) -> None:
        self.value: t.Any = value
        self.type: str = type
        self.help: t.Optional[str] = help
        self._info = kwargs
    
    def __getattr__(self, name: str) -> t.Any:
        return self._info.get(name)

_SOURCE_BASH = '%(complete_func)s() {\n    local IFS=$\'\\n\'\n    local response\n\n    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD %(complete_var)s=bash_complete $1)\n\n    for completion in $response; do\n        IFS=\',\' read type value <<< "$completion"\n\n        if [[ $type == \'dir\' ]]; then\n            COMPREPLY=()\n            compopt -o dirnames\n        elif [[ $type == \'file\' ]]; then\n            COMPREPLY=()\n            compopt -o default\n        elif [[ $type == \'plain\' ]]; then\n            COMPREPLY+=($value)\n        fi\n    done\n\n    return 0\n}\n\n%(complete_func)s_setup() {\n    complete -o nosort -F %(complete_func)s %(prog_name)s\n}\n\n%(complete_func)s_setup;\n'
_SOURCE_ZSH = '#compdef %(prog_name)s\n\n%(complete_func)s() {\n    local -a completions\n    local -a completions_with_descriptions\n    local -a response\n    (( ! $+commands[%(prog_name)s] )) && return 1\n\n    response=("${(@f)$(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) %(complete_var)s=zsh_complete %(prog_name)s)}")\n\n    for type key descr in ${response}; do\n        if [[ "$type" == "plain" ]]; then\n            if [[ "$descr" == "_" ]]; then\n                completions+=("$key")\n            else\n                completions_with_descriptions+=("$key":"$descr")\n            fi\n        elif [[ "$type" == "dir" ]]; then\n            _path_files -/\n        elif [[ "$type" == "file" ]]; then\n            _path_files -f\n        fi\n    done\n\n    if [ -n "$completions_with_descriptions" ]; then\n        _describe -V unsorted completions_with_descriptions -U\n    fi\n\n    if [ -n "$completions" ]; then\n        compadd -U -V unsorted -a completions\n    fi\n}\n\nif [[ $zsh_eval_context[-1] == loadautofunc ]]; then\n    # autoload from fpath, call function directly\n    %(complete_func)s "$@"\nelse\n    # eval/source/. command, register function for later\n    compdef %(complete_func)s %(prog_name)s\nfi\n'
_SOURCE_FISH = 'function %(complete_func)s;\n    set -l response (env %(complete_var)s=fish_complete COMP_WORDS=(commandline -cp) COMP_CWORD=(commandline -t) %(prog_name)s);\n\n    for completion in $response;\n        set -l metadata (string split "," $completion);\n\n        if test $metadata[1] = "dir";\n            __fish_complete_directories $metadata[2];\n        else if test $metadata[1] = "file";\n            __fish_complete_path $metadata[2];\n        else if test $metadata[1] = "plain";\n            echo $metadata[2];\n        end;\n    end;\nend;\n\ncomplete --no-files --command %(prog_name)s --arguments "(%(complete_func)s)";\n'


class ShellComplete:
    """Base class for providing shell completion support. A subclass for
    a given shell will override attributes and methods to implement the
    completion instructions (``source`` and ``complete``).

    :param cli: Command being called.
    :param prog_name: Name of the executable in the shell.
    :param complete_var: Name of the environment variable that holds
        the completion instruction.

    .. versionadded:: 8.0
    """
    name: t.ClassVar[str]
    'Name to register the shell as with :func:`add_completion_class`.\n    This is used in completion instructions (``{name}_source`` and\n    ``{name}_complete``).\n    '
    source_template: t.ClassVar[str]
    'Completion script template formatted by :meth:`source`. This must\n    be provided by subclasses.\n    '
    
    def __init__(self, cli: BaseCommand, ctx_args: t.MutableMapping[(str, t.Any)], prog_name: str, complete_var: str) -> None:
        self.cli = cli
        self.ctx_args = ctx_args
        self.prog_name = prog_name
        self.complete_var = complete_var
    
    @property
    def func_name(self) -> str:
        """The name of the shell function defined by the completion
        script.
        """
        safe_name = re.sub('\\W*', '', self.prog_name.replace('-', '_'), flags=re.ASCII)
        return f'_{safe_name}_completion'
    
    def source_vars(self) -> t.Dict[(str, t.Any)]:
        """Vars for formatting :attr:`source_template`.

        By default this provides ``complete_func``, ``complete_var``,
        and ``prog_name``.
        """
        return {'complete_func': self.func_name, 'complete_var': self.complete_var, 'prog_name': self.prog_name}
    
    def source(self) -> str:
        """Produce the shell script that defines the completion
        function. By default this ``%``-style formats
        :attr:`source_template` with the dict returned by
        :meth:`source_vars`.
        """
        return self.source_template % self.source_vars()
    
    def get_completion_args(self) -> t.Tuple[(t.List[str], str)]:
        """Use the env vars defined by the shell script to return a
        tuple of ``args, incomplete``. This must be implemented by
        subclasses.
        """
        raise NotImplementedError
    
    def get_completions(self, args: t.List[str], incomplete: str) -> t.List[CompletionItem]:
        """Determine the context and last complete command or parameter
        from the complete args. Call that object's ``shell_complete``
        method to get the completions for the incomplete value.

        :param args: List of complete args before the incomplete value.
        :param incomplete: Value being completed. May be empty.
        """
        ctx = _resolve_context(self.cli, self.ctx_args, self.prog_name, args)
        (obj, incomplete) = _resolve_incomplete(ctx, args, incomplete)
        return obj.shell_complete(ctx, incomplete)
    
    def format_completion(self, item: CompletionItem) -> str:
        """Format a completion item into the form recognized by the
        shell script. This must be implemented by subclasses.

        :param item: Completion item to format.
        """
        raise NotImplementedError
    
    def complete(self) -> str:
        """Produce the completion data to send back to the shell.

        By default this calls :meth:`get_completion_args`, gets the
        completions, then calls :meth:`format_completion` for each
        completion.
        """
        (args, incomplete) = self.get_completion_args()
        completions = self.get_completions(args, incomplete)
        out = [self.format_completion(item) for item in completions]
        return '\n'.join(out)



class BashComplete(ShellComplete):
    """Shell completion for Bash."""
    name = 'bash'
    source_template = _SOURCE_BASH
    
    @staticmethod
    def _check_version() -> None:
        import subprocess
        output = subprocess.run(['bash', '-c', 'echo "${BASH_VERSION}"'], stdout=subprocess.PIPE)
        match = re.search('^(\\d+)\\.(\\d+)\\.\\d+', output.stdout.decode())
        if match is not None:
            (major, minor) = match.groups()
            if (major < '4' or (major == '4' and minor < '4')):
                echo(_('Shell completion is not supported for Bash versions older than 4.4.'), err=True)
        else:
            echo(_("Couldn't detect Bash version, shell completion is not supported."), err=True)
    
    def source(self) -> str:
        self._check_version()
        return super().source()
    
    def get_completion_args(self) -> t.Tuple[(t.List[str], str)]:
        cwords = split_arg_string(os.environ['COMP_WORDS'])
        cword = int(os.environ['COMP_CWORD'])
        args = cwords[1:cword]
        try:
            incomplete = cwords[cword]
        except IndexError:
            incomplete = ''
        return (args, incomplete)
    
    def format_completion(self, item: CompletionItem) -> str:
        return f'{item.type},{item.value}'



class ZshComplete(ShellComplete):
    """Shell completion for Zsh."""
    name = 'zsh'
    source_template = _SOURCE_ZSH
    
    def get_completion_args(self) -> t.Tuple[(t.List[str], str)]:
        cwords = split_arg_string(os.environ['COMP_WORDS'])
        cword = int(os.environ['COMP_CWORD'])
        args = cwords[1:cword]
        try:
            incomplete = cwords[cword]
        except IndexError:
            incomplete = ''
        return (args, incomplete)
    
    def format_completion(self, item: CompletionItem) -> str:
        return f"{item.type}\n{item.value}\n{(item.help if item.help else '_')}"



class FishComplete(ShellComplete):
    """Shell completion for Fish."""
    name = 'fish'
    source_template = _SOURCE_FISH
    
    def get_completion_args(self) -> t.Tuple[(t.List[str], str)]:
        cwords = split_arg_string(os.environ['COMP_WORDS'])
        incomplete = os.environ['COMP_CWORD']
        args = cwords[1:]
        if (incomplete and args and args[-1] == incomplete):
            args.pop()
        return (args, incomplete)
    
    def format_completion(self, item: CompletionItem) -> str:
        if item.help:
            return f'{item.type},{item.value}\t{item.help}'
        return f'{item.type},{item.value}'

ShellCompleteType = t.TypeVar('ShellCompleteType', bound=t.Type[ShellComplete])
_available_shells: t.Dict[(str, t.Type[ShellComplete])] = {'bash': BashComplete, 'fish': FishComplete, 'zsh': ZshComplete}

def add_completion_class(cls: ShellCompleteType, name: t.Optional[str] = None) -> ShellCompleteType:
    """Register a :class:`ShellComplete` subclass under the given name.
    The name will be provided by the completion instruction environment
    variable during completion.

    :param cls: The completion class that will handle completion for the
        shell.
    :param name: Name to register the class under. Defaults to the
        class's ``name`` attribute.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.shell_completion.add_completion_class', 'add_completion_class(cls, name=None)', {'_available_shells': _available_shells, 'cls': cls, 'name': name, 't': t, 'str': str}, 1)

def get_completion_class(shell: str) -> t.Optional[t.Type[ShellComplete]]:
    """Look up a registered :class:`ShellComplete` subclass by the name
    provided by the completion instruction environment variable. If the
    name isn't registered, returns ``None``.

    :param shell: Name the class is registered under.
    """
    return _available_shells.get(shell)

def _is_incomplete_argument(ctx: Context, param: Parameter) -> bool:
    """Determine if the given parameter is an argument that can still
    accept values.

    :param ctx: Invocation context for the command represented by the
        parsed complete args.
    :param param: Argument object being checked.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.shell_completion._is_incomplete_argument', '_is_incomplete_argument(ctx, param)', {'Argument': Argument, 'ParameterSource': ParameterSource, 'ctx': ctx, 'param': param}, 1)

def _start_of_option(ctx: Context, value: str) -> bool:
    """Check if the value looks like the start of an option."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.shell_completion._start_of_option', '_start_of_option(ctx, value)', {'ctx': ctx, 'value': value}, 1)

def _is_incomplete_option(ctx: Context, args: t.List[str], param: Parameter) -> bool:
    """Determine if the given parameter is an option that needs a value.

    :param args: List of complete args before the incomplete value.
    :param param: Option object being checked.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.shell_completion._is_incomplete_option', '_is_incomplete_option(ctx, args, param)', {'Option': Option, '_start_of_option': _start_of_option, 'ctx': ctx, 'args': args, 'param': param, 't': t, 'str': str}, 1)

def _resolve_context(cli: BaseCommand, ctx_args: t.MutableMapping[(str, t.Any)], prog_name: str, args: t.List[str]) -> Context:
    """Produce the context hierarchy starting with the command and
    traversing the complete arguments. This only follows the commands,
    it doesn't trigger input prompts or callbacks.

    :param cli: Command being called.
    :param prog_name: Name of the executable in the shell.
    :param args: List of complete args before the incomplete value.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.shell_completion._resolve_context', '_resolve_context(cli, ctx_args, prog_name, args)', {'MultiCommand': MultiCommand, 'cli': cli, 'ctx_args': ctx_args, 'prog_name': prog_name, 'args': args, 't': t, 't': t, 'str': str}, 1)

def _resolve_incomplete(ctx: Context, args: t.List[str], incomplete: str) -> t.Tuple[(t.Union[(BaseCommand, Parameter)], str)]:
    """Find the Click object that will handle the completion of the
    incomplete value. Return the object and the incomplete value.

    :param ctx: Invocation context for the command represented by
        the parsed complete args.
    :param args: List of complete args before the incomplete value.
    :param incomplete: Value being completed. May be empty.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.shell_completion._resolve_incomplete', '_resolve_incomplete(ctx, args, incomplete)', {'_start_of_option': _start_of_option, '_is_incomplete_option': _is_incomplete_option, '_is_incomplete_argument': _is_incomplete_argument, 'ctx': ctx, 'args': args, 'incomplete': incomplete, 't': t, 'str': str, 't': t}, 2)

