import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


class GradientReversal(nn.Module):
    """
    Implementation of the gradient reversal layer described in
    [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818),
    which 'leaves the input unchanged during forward propagation
    and reverses the gradient by multiplying it
    by a negative scalar during backpropagation.'
    """

    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)
        self.register_buffer("alpha", torch.tensor([alpha], requires_grad=False))

    def update_alpha(self, alpha: float) -> None:
        self.alpha[0] = alpha

    def forward(self, input_):
        return _GradientReversal.apply(input_, self.alpha)

    def extra_repr(self):
        return f"alpha={self.alpha.item()}"
