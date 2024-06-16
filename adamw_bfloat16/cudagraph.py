"""
This version is cuda-graph compatible.
That is, ALL computations can be moved to GPU.
"""

import dataclasses

import torch
from torch.optim.optimizer import Optimizer


@dataclasses.dataclass(frozen=True)
class LR:
    preheat_steps: int = 3000
    lr: float = 1e-3
    decay_power: float = -0.5

    def __call__(self, step):
        # nb input is torch tensor, and all operations should be torch!
        # no if/else, no built-in min/max, etc.
        x = (step + 1) / self.preheat_steps
        return torch.minimum(x, x**self.decay_power) * self.lr


LR_default = LR()


class AdamW_BF16(Optimizer):
    decay_threshold = 5e-3

    def __init__(self, params, *, lr_function: LR = LR_default, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Implements AdamW optimization specifically for bfloat16 models.
        No other dtype is supported.
        Compatible with cuda graphs.
        Uses delayed accumulation for decays and compensated summation for Adam steps.
        Uses only one additional bfloat16 weight for keeping correction.
        Do not use schedulers - those can't affect cuda graphs.
        :param lr_function: a callable that maps torch scalar (step) to torch scalar (learning rate)
        """
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr_function=lr_function, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        assert p.dtype == torch.bfloat16, "only bfloat 16 is supported"
                        state["step"] = torch.zeros([], dtype=torch.int32)
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # accumulated shift that should be added to p, but wasn't because of truncation
                        # true value is p + shift
                        state["shift"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # using decay at each step will work only for float32, so we just remember how much owe to decay
                        # and decay once in n iterations
                        # Each weight has its own starting point to avoid simultaneous updates in all weights
                        state["accumulated_decay"] = torch.rand([], dtype=torch.bfloat16) * self.decay_threshold

                    grad = p.grad
                    state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    shift = state["shift"]

                    # update the steps for each param group update
                    state["step"] += 1
                    denom_correction = (1 - beta2 ** state["step"]) ** 0.5

                    lr = group["lr_function"](state["step"])
                    shift.addcdiv_(
                        state["exp_avg"],
                        state["exp_avg_sq"].sqrt().add_(group["eps"], alpha=1),
                        value=-lr * denom_correction,
                    )

                    # compensated summation, better this be built-in pytorch operation
                    buffer = p.clone()
                    p.add_(shift)
                    shift.add_(buffer.sub_(p))

                    accum_decay = state["accumulated_decay"]
                    accum_decay += group["weight_decay"] * lr
                    decay_this_iteration = (accum_decay > self.decay_threshold) * accum_decay
                    state["shift"].add_(p, alpha=-decay_this_iteration)
                    accum_decay -= decay_this_iteration
