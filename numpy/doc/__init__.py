import os
ref_dir = os.path.join(os.path.dirname(__file__))
__all__ = sorted((f[:-3] for f in os.listdir(ref_dir) if (f.endswith('.py') and not f.startswith('__'))))
for f in __all__:
    __import__(__name__ + '.' + f)
del f, ref_dir
__doc__ = 'Topical documentation\n=====================\n\nThe following topics are available:\n%s\n\nYou can view them by\n\n>>> help(np.doc.TOPIC)                                      #doctest: +SKIP\n\n' % '\n- '.join([''] + __all__)
__all__.extend(['__doc__'])

