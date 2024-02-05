import torch
import torch.nn as nn
import torchvision


class FlattenAndMLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: list[int] | None = None, **kwargs
    ):
        super().__init__()
        if hidden_dim is not None:
            hidden_channels = hidden_dim + [output_dim]
        else:
            hidden_channels = [output_dim]

        self.mlp = torchvision.ops.MLP(
            in_channels=input_dim, hidden_channels=hidden_channels, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input.
        x = x.view(x.size(0), -1).contiguous()

        # Forward pass through the MLP
        x = self.mlp(x)

        return x

    def __repr__(self):
        return repr(self.mlp)
