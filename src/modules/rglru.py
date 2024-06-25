import torch
import torch.nn.functional as F
from torch import Tensor, nn

class Cell(nn.Module):
    def __init__(self, rnn_width): 
        super(Cell, self).__init__()
        self.c = 8
        # Initialize weights and biases for the recurrence gate and input gate
        self.Wa = nn.Parameter(Tensor(rnn_width, rnn_width))
        self.Wx = nn.Parameter(Tensor(rnn_width, rnn_width))
        self.ba = nn.Parameter(Tensor(1, rnn_width))
        self.bx = nn.Parameter(Tensor(1, rnn_width))
        # Initialize the learnable parameter Λ for parameterizing 'a'
        self.Lambda = nn.Parameter(Tensor(1, rnn_width))
        self.reset_parameter()

    def reset_parameter(self):
        # LeCun init
        nn.init.kaiming_normal_(self.Wa, nonlinearity="sigmoid")
        nn.init.kaiming_normal_(self.Wx, nonlinearity="sigmoid")

        nn.init.constant_(self.ba, 0)
        nn.init.constant_(self.bx, 0)
        # Initialize Λ such that a powers c is between 0.9 and 0.999
        self.Lambda.data.uniform_(
            torch.logit(torch.tensor(0.9 ** (1 / self.c))),
            torch.logit(torch.tensor(0.999 ** (1 / self.c))),
        )

    def forward(self, xt, ht_minus_1):
        rt = torch.sigmoid(F.linear(xt, self.Wa, self.ba))
        it = torch.sigmoid(F.linear(xt, self.Wx, self.bx))

        log_at = -(self.c * F.softplus(self.Lambda)) * rt
        at = torch.exp(log_at)

        ht = at * ht_minus_1 + (1 - at**2).sqrt() * (it * xt)

        return ht

class RG_LRU(nn.Module):

    def __init__(self, rnn_width):
        super(RG_LRU, self).__init__()
        self.cell = Cell(rnn_width)
        self.rnn_width = rnn_width

    def forward(self, x, h=None):
        B, T, C = x.size()
        if h is None:
            h = torch.zeros(
                B, self.rnn_width,
                dtype=x.dtype, device=x.device
            )
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs
