import torch

import torch.nn as nn

from torch.cuda.amp import autocast
from torch.nn import functional as F

from einops import rearrange, einsum

from local_attention import LocalAttention
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

class LocalMQA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head=128,
        num_queries=1,
        dropout=0.,
        causal=False,
        prenorm=False,
        qk_rmsnorm=False,
        qk_scale=8,
        use_xpos=False,
        xpos_scale_base=None,
        exact_windowsize=None,
        gate_values_per_head=False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * num_queries

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.num_queries = num_queries
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attn_fn = LocalAttention(
            dim=dim_head,
            window_size=window_size,
            causal=causal,
            autopad=True,
            scale=(qk_scale if qk_rmsnorm else None),
            exact_windowsize=default(exact_windowsize, True),
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, num_queries)
            )

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, attn_bias=None):
        if exists(self.norm):
            x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = rearrange(q, 'b n (q d) -> b q n d', q=self.num_queries)
        k = rearrange(k, 'b n (q d) -> b q n d', q=self.num_queries)
        v = rearrange(v, 'b n (q d) -> b q n d', q=self.num_queries)

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        out = []
        for i in range(self.num_queries):
            q_i = q[:, i, :, :]
            k_i = k[:, i, :, :]
            v_i = v[:, i, :, :]
            out_i = self.attn_fn(q_i, k_i, v_i, mask=mask, attn_bias=attn_bias)
            out.append(out_i)

        out = torch.cat(out, dim=-1)

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n q -> b q n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b n (q d) -> b n (q d)', q=self.num_queries)
        return self.to_out(out)

class GlobalMQA(nn.Module):
    def __init__(
        self,
        dim,
        heads=6,
        dim_head=128,
        scale_base=None,
        use_xpos=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, dim_head, bias=False)
        self.to_v = nn.Linear(dim, dim_head, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.rotary_emb = SinusoidalEmbeddings(
            dim_head,
            scale_base=scale_base,
            use_xpos=use_xpos
        )

    @autocast(enabled=False)
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n d -> b 1 n d')
        v = rearrange(v, 'b n d -> b 1 n d')

        freqs, scale = self.rotary_emb(q)

        q, k = apply_rotary_pos_emb(q, k, freqs, scale)

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)