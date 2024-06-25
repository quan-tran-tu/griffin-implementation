from dataclasses import dataclass

# rnn_width ~ 4/3 * input_size

@dataclass
class HawkConfig:
    input_size = 768
    rnn_width = 1024
    depth = 12
    expansion_factor = 3
    max_seq_len = 1024
    num_tokens = 10000 # vocab_size

@dataclass
class GriffinConfig:
    input_size = 768
    rnn_width = 1024
    depth = 12
    expansion_factor = 3
    max_seq_len = 1024
    num_tokens = 10000 # vocab_size

@dataclass
class TransformerConfig:
    input_size = 768
    rnn_width = 1024
    depth = 12
    expansion_factor = 3
    max_seq_len = 1024
    num_tokens = 10000 # vocab_size