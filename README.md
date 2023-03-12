# CBAM-CrossAttention

# Description

This repo is a implement of Cross-Attention module about Neural Network,  
which use attention module CBAM.

It is tested with pytorch-1.12.0.
  
    
# How to use

## Define
```python
from Cross_Attention import CA
CA_object = CA(gate_channels, reduction_ratio, pool_types, no_spatial, KQ, V)
                {(required), (default=16), (default=avg&max), (default=false), (required), (required)}
```

## Sample code
```python
import torch
from Cross_Attention import CA

CA_object = CA(gate_channels=128, KQ=20, V=24)

tensor_KQ = torch.ones(32, 128, 1024, 20)
tensor_V  = torch.ones(32, 128, 1024, 24)

Refined_tensor = CA_object(tensor_KQ, tensor_V)
```

# About the project

Here reserve origin Channel design of CBAM, but add MLP in Spatial Attention,  
because i want resize tensor size and also keep information.

I also tried use single layer linear to replace MLP,
but it mismatch with my another task,  
i presume single layer linear is too simple to work in complicated task.


## Overall Architecture
![architecture](img\architecture.jpg)

## Channel Attention
![channel](img\channel.jpg)

## Spatial Attention
![spatial](img\spatial.jpg)

# Reference


### CBAM: Convolutional Block Attention Module
https://github.com/Jongchan/attention-module

# Contact

Further information please contact me.

wuyiulin@gmail.com