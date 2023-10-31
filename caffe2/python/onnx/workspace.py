from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import uuid
from caffe2.python import workspace


class _WorkspaceCtx(object):
    
    def __init__(self, workspace_id):
        self.workspace_id = workspace_id
        self.workspace_stack = []
    
    def __enter__(self):
        self.workspace_stack.append(workspace.CurrentWorkspace())
        workspace.SwitchWorkspace(self.workspace_id, create_if_missing=True)
    
    def __exit__(self, exc_type, exc_value, traceback):
        w = self.workspace_stack.pop()
        workspace.SwitchWorkspace(w, create_if_missing=True)



class Workspace(object):
    """
    An object representing a Caffe2 workspace.  It is a context manager,
    so you can say 'with workspace:' to use the represented workspace
    as your global workspace.  It also supports every method supported
    by caffe2.python.workspace, but instead of running these operations
    in the global workspace, it runs them in the workspace represented
    by this object.  When this object goes dead, the workspace (and all
    nets and blobs within it) are freed.

    Why do we need this class?  Caffe2's workspace model is very "global state"
    oriented, in that there is always some ambient global workspace you are
    working in which holds on to all of your networks and blobs.  This class
    makes it possible to work with workspaces more locally, and without
    forgetting to deallocate everything in the end.
    """
    
    def __init__(self):
        self._ctx = _WorkspaceCtx(str(uuid.uuid4()))
    
    def __getattr__(self, attr):
        
        def f(*args, **kwargs):
            with self._ctx:
                return getattr(workspace, attr)(*args, **kwargs)
        return f
    
    def __del__(self):
        self.ResetWorkspace()


