import torch
import torch.nn.functional as F
from torch import Tensor, nn

class Gated_MLP(nn.Module):

    def __init__(self, input_size, expansion_factor):
        super(Gated_MLP, self).__init__()
        output_size = input_size * expansion_factor
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.linear3 = nn.Linear(output_size, input_size)

    def forward(self, x):
        # x_gelu = F.gelu(self.linear1(x))
        x_gelu = F.silu(self.linear1(x))
        x = x_gelu * self.linear2(x)
        x = self.linear3(x)
        return x
