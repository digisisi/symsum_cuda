"""This script compared the GPU forward speed of the Cuda-optimized SymSum with those of
the default SumSum and ReLU.
The extension should be installed for this to work.
"""

import torch
from symsum import SymSum as CudaOptimizedSymSum
import time


class DefaultSymSum(torch.nn.Module):
    """The default implementation of SymSum"""
    def __init__(self):
        super(DefaultSymSum, self).__init__()

    @staticmethod
    def forward(x):
        zero = torch.tensor(0.).to(x.device)
        shift = x.shape[1] // 2  # This is so that R+(a) + R-(b) and R-(a) + R+(b)
        relu = torch.maximum(zero, x)
        inv_relu = torch.minimum(zero, x)
        out = torch.roll(inv_relu, shift, dims=1) + relu
        return out


ss = DefaultSymSum()
opt_ss = CudaOptimizedSymSum()

a_cuda = torch.randn(128, 256, 8, 8).to('cuda').requires_grad_()
a_cuda2 = a_cuda.clone().detach().requires_grad_()

# Testing the output of Cuda-optimized vs default SymSum
with torch.no_grad():
    print((a_cuda == a_cuda2).all())
    o1 = opt_ss(a_cuda)
    o2 = ss(a_cuda)
    same = (o1 == o2).all()
    print(same)

# Timing their forward pass
t0 = time.time()
_ = ss(a_cuda)
t1 = time.time()
default_time = t1-t0

t0 = time.time()
_ = opt_ss(a_cuda)
t1 = time.time()
optimized_time = t1-t0

t0 = time.time()
_ = torch.relu(a_cuda)
t1 = time.time()
relu_time = t1-t0

print(f'Default: {default_time*1000}ms')
print(f'Optimized: {optimized_time*1000}ms')
print(f'Relu: {relu_time*1000}ms')
