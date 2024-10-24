# Stick-breaking Attention Implementation
Triton-based implementation of Stick-breaking Attention on GPUs.
This implementation is for variable length .
You can find the paper [here](https://arxiv.org/abs/2403.08245)

## Installation
```sh
# Install editable. This will allow you to modify stickbreaking in this directory.
pip install -e .
# Check all is working well.
pytest tests
```
### Usage
#### Variable Length Attention
Each mini-batch consists of concatenated sequences of different lengths.

`sb_attn_varlen` implements the counterpart to Flash Attention's 
[`flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L1334).
Assuming we have an input batch that concatenates all documents/sequences into a long array, and the corresponding
sequence lengths in the batch in an array `lengths`. 
Then we can compute the cu_seqlens and pass that to `sb_attn_varlen`:
```python
import torch
from stickbreaking_attention.sb_varlen import sb_attn_varlen
# lengths: batch_size,
total_length = torch.sum(lengths)
# q, k, v: num_heads, total_length, head_dima
cu_seqlens = torch.cumsum(lengths) 
o, rem = sb_attn_varlen(q, k, v, cu_seqlens, zero_start=False)
```
For better performance, we can prepare the additional metadata for
stick-breaking at the start of the forward pass, and using the same
metadata for each attention layer:
```python
import torch

from stickbreaking_attention.sb_varlen import (
    row_block_counts_and_sequence_ids,
    sb_attn_varlen_,
    BLOCK_M, BLOCK_N
)

# Prepare metadata at the start...
with torch.no_grad():
    # lengths: batch_size,
    cu_seqlens = torch.cumsum(lengths) 
    cu_row_blocks, first_row_block, sequence_ids = row_block_counts_and_sequence_ids(
        cu_seqlens, BLOCK_M, BLOCK_N
    )

# Call this every attention layer...
# q, k, v: num_heads, total_length, head_dima
o, rem = sb_attn_varlen_(
    q, k, v,
    cu_seqlens=cu_seqlens,
    first_row_block=first_row_block,
    cu_row_blocks=cu_row_blocks,
    sequence_ids=sequence_ids,
)
```


#### More Helper functions, coming soon...



## Loading trained 3B model

For loading and testing or fine-tuning the 3B model, install [Dolomite Engine](https://github.com/IBM/dolomite-engine):
```sh
pip install git+https://github.com/IBM/dolomite-engine
```
Dolomite Engine implements the `sb_dolomite` class that loads the `ibm/stickbreaking` model from HuggingFace.
```python
from transformers import AutoModelForCausalLM
from dolomite_engine import hf_models # registers dolomite model classes with transformers
# :
sb_model = AutoModelForCausalLM.from_pretrained(
    'shawntan/stickbreaking-3b',
)
# Do stuff with sb_model
```


Enjoy!
