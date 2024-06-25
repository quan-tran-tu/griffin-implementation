import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, input_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_size))
        self.variance_epsilon = eps

    def forward(self, input):
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            input = input.to(self.weight.dtype)

        return self.weight * input