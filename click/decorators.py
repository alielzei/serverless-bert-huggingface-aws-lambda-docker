import inspect
import types
import typing as t
from functools import update_wrapper
from gettext import gettext as _
from .core import Argument
from .core import Command
from .core import Context
from .core import Group
from .core import Option
from .core import Parameter
from .globals import get_current_context
from .utils import echo
if t.TYPE_CHECKING:
    import typing_extensions as te
    P = te.ParamSpec('P')
R = t.TypeVar('R')
T = t.TypeVar('T')
_AnyCallable = t.Callable[(..., t.Any)]
FC = t.TypeVar('FC', bound=t.Union[(_AnyCallable, Command)])

def pass_context(f: 't.Callable[te.Concatenate[Context, P], R]') -> 't.Callable[P, R]':
    """Marks a callback as wanting to receive the current context
    object as first argument.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.pass_context', 'pass_context(f)', {'get_current_context': get_current_context, 'update_wrapper': update_wrapper, 'f': f}, 1)

def pass_obj(f: 't.Callable[te.Concatenate[t.Any, P], R]') -> 't.Callable[P, R]':
    """Similar to :func:`pass_context`, but only pass the object on the
    context onwards (:attr:`Context.obj`).  This is useful if that object
    represents the state of a nested system.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.pass_obj', 'pass_obj(f)', {'get_current_context': get_current_context, 'update_wrapper': update_wrapper, 'f': f}, 1)

def make_pass_decorator(object_type: t.Type[T], ensure: bool = False) -> t.Callable[(['t.Callable[te.Concatenate[T, P], R]'], 't.Callable[P, R]')]:
    """Given an object type this creates a decorator that will work
    similar to :func:`pass_obj` but instead of passing the object of the
    current context, it will find the innermost context of type
    :func:`object_type`.

    This generates a decorator that works roughly like this::

        from functools import update_wrapper

        def decorator(f):
            @pass_context
            def new_func(ctx, *args, **kwargs):
                obj = ctx.find_object(object_type)
                return ctx.invoke(f, obj, *args, **kwargs)
            return update_wrapper(new_func, f)
        return decorator

    :param object_type: the type of the object to pass.
    :param ensure: if set to `True`, a new object will be created and
                   remembered on the context if it's not there yet.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.make_pass_decorator', 'make_pass_decorator(object_type, ensure=False)', {'get_current_context': get_current_context, 't': t, 'T': T, 'update_wrapper': update_wrapper, 'object_type': object_type, 'ensure': ensure, 't': t, 'T': T, 't': t}, 1)

def pass_meta_key(key: str, *, doc_description: t.Optional[str] = None) -> 't.Callable[[t.Callable[te.Concatenate[t.Any, P], R]], t.Callable[P, R]]':
    """Create a decorator that passes a key from
    :attr:`click.Context.meta` as the first argument to the decorated
    function.

    :param key: Key in ``Context.meta`` to pass.
    :param doc_description: Description of the object being passed,
        inserted into the decorator's docstring. Defaults to "the 'key'
        key from Context.meta".

    .. versionadded:: 8.0
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.pass_meta_key', 'pass_meta_key(key, doc_description: t.Optional[str] = None)', {'R': R, 'get_current_context': get_current_context, 'update_wrapper': update_wrapper, 'key': key, 'doc_description': doc_description}, 1)
CmdType = t.TypeVar('CmdType', bound=Command)

@t.overload
def command(name: _AnyCallable) -> Command:
    ...

@t.overload
def command(name: t.Optional[str], cls: t.Type[CmdType], **attrs) -> t.Callable[([_AnyCallable], CmdType)]:
    ...

@t.overload
def command(name: None = None, *, cls: t.Type[CmdType], **attrs) -> t.Callable[([_AnyCallable], CmdType)]:
    ...

@t.overload
def command(name: t.Optional[str] = ..., cls: None = None, **attrs) -> t.Callable[([_AnyCallable], Command)]:
    ...

def command(name: t.Union[(t.Optional[str], _AnyCallable)] = None, cls: t.Optional[t.Type[CmdType]] = None, **attrs) -> t.Union[(Command, t.Callable[([_AnyCallable], t.Union[(Command, CmdType)])])]:
    """Creates a new :class:`Command` and uses the decorated function as
    callback.  This will also automatically attach all decorated
    :func:`option`\s and :func:`argument`\s as parameters to the command.

    The name of the command defaults to the name of the function with
    underscores replaced by dashes.  If you want to change that, you can
    pass the intended name as the first argument.

    All keyword arguments are forwarded to the underlying command class.
    For the ``params`` argument, any decorated params are appended to
    the end of the list.

    Once decorated the function turns into a :class:`Command` instance
    that can be invoked as a command line utility or be attached to a
    command :class:`Group`.

    :param name: the name of the command.  This defaults to the function
                 name with underscores replaced by dashes.
    :param cls: the command class to instantiate.  This defaults to
                :class:`Command`.

    .. versionchanged:: 8.1
        This decorator can be applied without parentheses.

    .. versionchanged:: 8.1
        The ``params`` argument can be used. Decorated params are
        appended to the end of the list.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.command', 'command(name=None, cls=None, **attrs)', {'t': t, '_AnyCallable': _AnyCallable, 'CmdType': CmdType, 'Command': Command, 'name': name, 'cls': cls, 'attrs': attrs, 't': t, 't': t, 't': t}, 1)
GrpType = t.TypeVar('GrpType', bound=Group)

@t.overload
def group(name: _AnyCallable) -> Group:
    ...

@t.overload
def group(name: t.Optional[str], cls: t.Type[GrpType], **attrs) -> t.Callable[([_AnyCallable], GrpType)]:
    ...

@t.overload
def group(name: None = None, *, cls: t.Type[GrpType], **attrs) -> t.Callable[([_AnyCallable], GrpType)]:
    ...

@t.overload
def group(name: t.Optional[str] = ..., cls: None = None, **attrs) -> t.Callable[([_AnyCallable], Group)]:
    ...

def group(name: t.Union[(str, _AnyCallable, None)] = None, cls: t.Optional[t.Type[GrpType]] = None, **attrs) -> t.Union[(Group, t.Callable[([_AnyCallable], t.Union[(Group, GrpType)])])]:
    """Creates a new :class:`Group` with a function as callback.  This
    works otherwise the same as :func:`command` just that the `cls`
    parameter is set to :class:`Group`.

    .. versionchanged:: 8.1
        This decorator can be applied without parentheses.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.group', 'group(name=None, cls=None, **attrs)', {'t': t, 'GrpType': GrpType, 'Group': Group, 'command': command, 'name': name, 'cls': cls, 'attrs': attrs, 't': t, 't': t, 't': t}, 1)

def _param_memo(f: t.Callable[(..., t.Any)], param: Parameter) -> None:
    import custom_funtemplate
    custom_funtemplate.rewrite_template('click.decorators._param_memo', '_param_memo(f, param)', {'Command': Command, 'f': f, 'param': param, 't': t}, 0)

def argument(*param_decls, cls: t.Optional[t.Type[Argument]] = None, **attrs) -> t.Callable[([FC], FC)]:
    """Attaches an argument to the command.  All positional arguments are
    passed as parameter declarations to :class:`Argument`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Argument` instance manually
    and attaching it to the :attr:`Command.params` list.

    For the default argument class, refer to :class:`Argument` and
    :class:`Parameter` for descriptions of parameters.

    :param cls: the argument class to instantiate.  This defaults to
                :class:`Argument`.
    :param param_decls: Passed as positional arguments to the constructor of
        ``cls``.
    :param attrs: Passed as keyword arguments to the constructor of ``cls``.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.argument', 'argument(*param_decls, cls: t.Optional[t.Type[Argument]] = None, **attrs)', {'Argument': Argument, 'FC': FC, '_param_memo': _param_memo, 'cls': cls, 'param_decls': param_decls, 'attrs': attrs, 't': t}, 1)

def option(*param_decls, cls: t.Optional[t.Type[Option]] = None, **attrs) -> t.Callable[([FC], FC)]:
    """Attaches an option to the command.  All positional arguments are
    passed as parameter declarations to :class:`Option`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Option` instance manually
    and attaching it to the :attr:`Command.params` list.

    For the default option class, refer to :class:`Option` and
    :class:`Parameter` for descriptions of parameters.

    :param cls: the option class to instantiate.  This defaults to
                :class:`Option`.
    :param param_decls: Passed as positional arguments to the constructor of
        ``cls``.
    :param attrs: Passed as keyword arguments to the constructor of ``cls``.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.option', 'option(*param_decls, cls: t.Optional[t.Type[Option]] = None, **attrs)', {'Option': Option, 'FC': FC, '_param_memo': _param_memo, 'cls': cls, 'param_decls': param_decls, 'attrs': attrs, 't': t}, 1)

def confirmation_option(*param_decls, **kwargs) -> t.Callable[([FC], FC)]:
    """Add a ``--yes`` option which shows a prompt before continuing if
    not passed. If the prompt is declined, the program will exit.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--yes"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.confirmation_option', 'confirmation_option(*param_decls, **kwargs)', {'Context': Context, 'Parameter': Parameter, 'option': option, 'param_decls': param_decls, 'kwargs': kwargs, 't': t}, 1)

def password_option(*param_decls, **kwargs) -> t.Callable[([FC], FC)]:
    """Add a ``--password`` option which prompts for a password, hiding
    input and asking to enter the value again for confirmation.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--password"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.password_option', 'password_option(*param_decls, **kwargs)', {'option': option, 'param_decls': param_decls, 'kwargs': kwargs, 't': t}, 1)

def version_option(version: t.Optional[str] = None, *param_decls, package_name: t.Optional[str] = None, prog_name: t.Optional[str] = None, message: t.Optional[str] = None, **kwargs) -> t.Callable[([FC], FC)]:
    """Add a ``--version`` option which immediately prints the version
    number and exits the program.

    If ``version`` is not provided, Click will try to detect it using
    :func:`importlib.metadata.version` to get the version for the
    ``package_name``. On Python < 3.8, the ``importlib_metadata``
    backport must be installed.

    If ``package_name`` is not provided, Click will try to detect it by
    inspecting the stack frames. This will be used to detect the
    version, so it must match the name of the installed package.

    :param version: The version number to show. If not provided, Click
        will try to detect it.
    :param param_decls: One or more option names. Defaults to the single
        value ``"--version"``.
    :param package_name: The package name to detect the version from. If
        not provided, Click will try to detect it.
    :param prog_name: The name of the CLI to show in the message. If not
        provided, it will be detected from the command.
    :param message: The message to show. The values ``%(prog)s``,
        ``%(package)s``, and ``%(version)s`` are available. Defaults to
        ``"%(prog)s, version %(version)s"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    :raise RuntimeError: ``version`` could not be detected.

    .. versionchanged:: 8.0
        Add the ``package_name`` parameter, and the ``%(package)s``
        value for messages.

    .. versionchanged:: 8.0
        Use :mod:`importlib.metadata` instead of ``pkg_resources``. The
        version is detected based on the package name, not the entry
        point name. The Python package name must match the installed
        package name, or be passed with ``package_name=``.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.version_option', 'version_option(version=None, *param_decls, package_name: t.Optional[str] = None, prog_name: t.Optional[str] = None, message: t.Optional[str] = None, **kwargs)', {'_': _, 'inspect': inspect, 'Context': Context, 'Parameter': Parameter, 't': t, 'types': types, 'echo': echo, 'option': option, 'version': version, 'package_name': package_name, 'prog_name': prog_name, 'message': message, 'param_decls': param_decls, 'kwargs': kwargs, 't': t, 'str': str, 't': t}, 1)

def help_option(*param_decls, **kwargs) -> t.Callable[([FC], FC)]:
    """Add a ``--help`` option which immediately prints the help page
    and exits the program.

    This is usually unnecessary, as the ``--help`` option is added to
    each command automatically unless ``add_help_option=False`` is
    passed.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--help"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.decorators.help_option', 'help_option(*param_decls, **kwargs)', {'Context': Context, 'Parameter': Parameter, 'echo': echo, '_': _, 'option': option, 'param_decls': param_decls, 'kwargs': kwargs, 't': t}, 1)

