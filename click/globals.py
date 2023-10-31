import typing as t
from threading import local
if t.TYPE_CHECKING:
    import typing_extensions as te
    from .core import Context
_local = local()

@t.overload
def get_current_context(silent: 'te.Literal[False]' = False) -> 'Context':
    ...

@t.overload
def get_current_context(silent: bool = ...) -> t.Optional['Context']:
    ...

def get_current_context(silent: bool = False) -> t.Optional['Context']:
    """Returns the current click context.  This can be used as a way to
    access the current context object from anywhere.  This is a more implicit
    alternative to the :func:`pass_context` decorator.  This function is
    primarily useful for helpers such as :func:`echo` which might be
    interested in changing its behavior based on the current context.

    To push the current context, :meth:`Context.scope` can be used.

    .. versionadded:: 5.0

    :param silent: if set to `True` the return value is `None` if no context
                   is available.  The default behavior is to raise a
                   :exc:`RuntimeError`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.globals.get_current_context', 'get_current_context(silent=False)', {'t': t, '_local': _local, 'silent': silent, 't': t}, 1)

def push_context(ctx: 'Context') -> None:
    """Pushes a new context to the current stack."""
    _local.__dict__.setdefault('stack', []).append(ctx)

def pop_context() -> None:
    """Removes the top level from the stack."""
    _local.stack.pop()

def resolve_color_default(color: t.Optional[bool] = None) -> t.Optional[bool]:
    """Internal helper to get the default value of the color flag.  If a
    value is passed it's returned unchanged, otherwise it's looked up from
    the current context.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click.globals.resolve_color_default', 'resolve_color_default(color=None)', {'get_current_context': get_current_context, 'color': color, 't': t, 'bool': bool, 't': t, 'bool': bool}, 1)

