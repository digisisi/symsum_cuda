# Intro
- This repo contains a CUDA kernel implementation of the SymSum activation function as a [PyTorch extention](https://pytorch.org/tutorials/advanced/cpp_extension.html).
- The implementation leverages the optimization described in the paper "Say Goodbye to Gradient Vanishing".
- It achieves similar speed performance as Pytorch's built-in ReLU function.
- See `symsum.py` for the Python wrappers.

# Installation
Simply run:
```python setup_cuda.py install```

# Usage
```
from symsum import SymSum as CudaOptimizedSymSum  
opt_ss = CudaOptimizedSymSum()
x = torch.randn(128, 256, 8, 8).to('cuda')
output = opt_ss(x)
```
