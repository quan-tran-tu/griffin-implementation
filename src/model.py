import torch

import torch.nn as nn
from torch.nn import functional as F

from src.modules import RG_LRU, RMSNorm, Gated_MLP, LocalMQA, GlobalMQA
import sys

class OutputHead(nn.Module):
    def __init__(self, input_size: int):
        super(OutputHead, self).__init__()
        self.layer_norm = RMSNorm(input_size)
        self.linear_layer = nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear_layer(x)
        return x

class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.num_tokens, config.input_size)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.input_size)
        self.lm_head = nn.Linear(config.input_size, config.num_tokens)
        self.out = OutputHead(config.input_size)
        self.layers = nn.ModuleList()
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        assert seq_len <= self.config.max_seq_len, "Input sequence length exceeds max sequence length."
    
        position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        token_embeddings = self.token_emb(x)
        
        position_embeddings = self.pos_emb(position_ids)
        
        x = token_embeddings + position_embeddings
        
        for layer in self.layers:
            x = layer(x)
        logits = self.out(x) # note: using list [-1] to preserve the time dim
        return logits

class RecurrentBlock(nn.Module):
    def __init__(self, input_size, rnn_width):
        super(RecurrentBlock, self).__init__()
        self.rg_lru = RG_LRU(rnn_width)
        self.temporal_conv1d = nn.Conv1d(in_channels=rnn_width, out_channels=rnn_width, kernel_size=4, dilation=2, padding=3)
        self.linear1 = nn.Linear(input_size, rnn_width)
        self.linear2 = nn.Linear(rnn_width, input_size)

    def forward(self, x):
        # x = F.gelu(self.linear1(x))
        x = F.silu(self.linear1(x))
        x = x.transpose(1, 2)
        x = self.temporal_conv1d(x).transpose(1, 2)
        x = self.rg_lru(x)
        x = self.linear2(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, temporal, input_size, rnn_width, expansion_factor):
        super(ResidualBlock, self).__init__()
        self.norm1 = RMSNorm(input_size)
        self.norm2 = RMSNorm(input_size)
        if temporal == 'recurrent':
            self.temporal = RecurrentBlock(input_size, rnn_width)
        elif temporal == 'local':
            self.temporal = LocalMQA(dim=input_size , window_size=1024)
        elif temporal == 'global':
            self.temporal = GlobalMQA(dim=input_size)
        self.mlp = Gated_MLP(input_size, expansion_factor)

    def forward(self, x):
        x = x + self.temporal(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Hawk(LanguageModel):
    def __init__(self, config):
        super(Hawk, self).__init__(config=config)
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    temporal='recurrent',
                    input_size=config.input_size,
                    rnn_width=config.rnn_width,
                    expansion_factor=config.expansion_factor
                ) 
                for _ in range(config.depth)
            ]
        )

    
class Griffin(LanguageModel):
    def __init__(self, config):
        super(Griffin, self).__init__(config=config)
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    temporal="local" if i % 3 == 2 else "recurrent",
                    input_size=config.input_size,
                    rnn_width=config.rnn_width,
                    expansion_factor=config.expansion_factor
                )
                for i in range(config.depth)
            ]
        )
    
class Transformer(LanguageModel):
    def __init__(self, config):
        super(Transformer, self).__init__(config=config)
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    temporal="global",
                    input_size=config.input_size,
                    rnn_width=config.rnn_width,
                    expansion_factor=config.expansion_factor
                )
                for _ in range(config.depth)
            ]
        )