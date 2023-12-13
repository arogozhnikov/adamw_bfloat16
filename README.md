![catchy image of optimization](https://miro.medium.com/max/1400/1*emPDLzTy0oW5BWLuxDSbKQ.png)

[<sub>Image source</sub>](https://medium.com/analytics-vidhya/optimization-acb996a4623c)

## AdamW optimizer for bfloat16 models in pytorch.

- Bfloat16 is currently an optimal tradeoff between range and relative error for deep networks.
- Bfloat16 can be used quite efficiently on Nvidia GPUs with Ampere architecture (H100, A100, A10, A30, RTX3090...)

However, usage of bfloat16 in torch ecosystem is ... awkward 
(torch AMP is very non-transparent, and was initially developed with a focus on fp16, which is **totally** different from bf16).


### Problem of stale weights

If you just convert all weights and inputs to bfloat16, you're likely to run into an **issue of stale weights**:
updates are too small to modify bfloat16 weight 
(see [gopher paper](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf), section C2 for a large-scale example).

There are two possible remedies: 

- keep weights in float32 (precise) and bfloat16 (approximate)
- keep weights in bfloat16, and keep correction term in bfloat16  

As [recent study](https://arxiv.org/abs/2010.06192) has shown, 
both options are completely competitive in quality to float32 training.
That's what we implement here in a convenient wrapper.


## Installation

```bash
pip install git+https://github.com/arogozhnikov/adamw_bfloat16.git
```

## Usage

Use as a drop-in replacement for pytorch's AdamW:
```python
import torch
from adamw_bfloat16 import LR, AdamW_BF16
model = model.to(torch.bfloat16)

# default preheat and decay
optimizer = AdamW_BF16(model.parameters())

# configure LR schedule. Use built-in scheduling opportunity
optimizer = AdamW_BF16(model.parameters(), lr_function=LR(lr=1e-4, preheat_steps=5000, decay_power=-0.25))

# in the loop:
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Or you can even replace last two lines with one:

```python
optimizer.step(zero_grad=True)
```

This optimizer simplifies the code by removing:

- grad scaler
- amp/autocast
- you can just forget about float32 computations
- lr scheduler (also no need to `.step()` scheduler)

Uses ~25% less memory per parameter compared to built-in AdamW.
