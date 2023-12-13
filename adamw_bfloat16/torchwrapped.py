"""
This is an experimental idea, where all the work is offloaded to torch optimizer.
Internally, fp32 replica is kept within an optimizer.
"""
from typing import Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import params_t


def mock_param(param):
    assert isinstance(param, torch.Tensor)
    if param.dtype == torch.float32:
        # no need to mock
        return param
    elif param.dtype == torch.bfloat16:
        result = param.to(torch.bfloat16)
        result.requires_grad = True
        result.grad = torch.zeros_like(result)
        param.grad = result.grad
        return result
    else:
        raise RuntimeError("Dtype of parameter can be only float32 or bfloat16")


# we need to rewrite step and creation.
# during creation, parameter inside the optimizer is replaced with fp32-version,
# and after the step is done, we assign new values of parameters + eventually check dtypes of gradients


class WrappedAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: params_t,
        lr: float | Tensor = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        assert isinstance(params, list), "need to add support for groups"
        # replace each bfloat16 param with a float32 mock, that will participate in optimization
        self.mock2param = {mock_param(p): p for p in params}

        super().__init__(
            list(self.mock2param),
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

    def step(self, closure=None):
        result = torch.optim.AdamW.step(self, closure=closure)
        with torch.no_grad():
            for mock, param in self.mock2param.items():
                param[:] = mock
                assert param.grad is mock.grad
        return result
