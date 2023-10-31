import torch
from functools import reduce
from .optimizer import Optimizer

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.optim.lbfgs._cubic_interpolate', '_cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None)', {'x1': x1, 'f1': f1, 'g1': g1, 'x2': x2, 'f2': f2, 'g2': g2, 'bounds': bounds}, 1)

def _strong_wolfe(obj_func, x, t, d, f, g, gtd, c1=0.0001, c2=0.9, tolerance_change=1e-09, max_ls=25):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.optim.lbfgs._strong_wolfe', '_strong_wolfe(obj_func, x, t, d, f, g, gtd, c1=0.0001, c2=0.9, tolerance_change=1e-09, max_ls=25)', {'torch': torch, '_cubic_interpolate': _cubic_interpolate, 'obj_func': obj_func, 'x': x, 't': t, 'd': d, 'f': f, 'g': g, 'gtd': gtd, 'c1': c1, 'c2': c2, 'tolerance_change': tolerance_change, 'max_ls': max_ls}, 4)


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """
    
    def __init__(self, params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size, line_search_fn=line_search_fn)
        super(LBFGS, self).__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
    
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache
    
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()
    
    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
    
    def _set_param(self, params_data):
        for (p, pdata) in zip(self._params, params_data):
            p.copy_(pdata)
    
    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return (loss, flat_grad)
    
    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1
        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad
        if opt_cond:
            return orig_loss
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            state['n_iter'] += 1
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)
                if ys > 1e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    H_diag = ys / y.dot(y)
                num_old = len(old_dirs)
                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)
            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss
            if state['n_iter'] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr
            gtd = flat_grad.dot(d)
            if gtd > -tolerance_change:
                break
            ls_func_evals = 0
            if line_search_fn is not None:
                if line_search_fn != 'strong_wolfe':
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()
                    
                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)
                    (loss, flat_grad, t, ls_func_evals) = _strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                self._add_grad(t, d)
                if n_iter != max_iter:
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals
            if n_iter == max_iter:
                break
            if current_evals >= max_eval:
                break
            if opt_cond:
                break
            if d.mul(t).abs().max() <= tolerance_change:
                break
            if abs(loss - prev_loss) < tolerance_change:
                break
        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss
        return orig_loss


