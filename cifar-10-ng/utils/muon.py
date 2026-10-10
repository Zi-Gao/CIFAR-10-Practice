import torch

@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    This version handles the reshaping internally or expects a 2D input.
    Adapted from airbench94_muon.py.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer that handles 4D/nD parameters by reshaping them to 2D 
    during the orthogonalization step.
    """
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # AirBench style weight normalization
                # Helps maintain scale when using orthogonal updates
                p.data.mul_(len(p.data)**0.5 / p.data.norm()) 
                
                # Reshape to 2D, whiten (orthogonalize), and reshape back
                # This allows it to work with standard 4D Conv weights
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) 
                
                p.data.add_(update, alpha=-lr) 
