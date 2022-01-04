import torch
try:
    import symsum_cuda
    no_cuda_optimized = False
    print('Using Cuda-optimized SymSum')
except:
    no_cuda_optimized = True
    print('Cuda-optimized SymSum was not found. Will use the default implementation for Cuda tensors.')
    

class SymSumCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return symsum_cuda.forward(x)

    @staticmethod
    def backward(ctx, output_grad):
        x = ctx.saved_tensors[0]
        return symsum_cuda.backward(x, output_grad)
 
    
class SymSum(torch.nn.Module):
    def __init__(self):
        super(SymSum, self).__init__()

    def forward(self, x):
        if x.device == 'cpu' or no_cuda_optimized:
            """
            Note 1: There is an optimized AVX version of the CPU implementation available that is ~4x faster (on par with ReLU). 
             But it is not added due to added complexity that is incurred to ensure portability.
            Note 2: At exact zero, the behavior of the non-optimized version is slightly different than the optimized version.
             This is due to the imlementation of `maximum` and `minimum` in PyTorch, where they return 0.5 gradient at the threshold.
             However, this should be of little importance in most practical applications. 
            """
            zero = torch.tensor(0.).to(x.device)
            shift = x.shape[1] // 2  # This is so that R+(a) + R-(b) and R-(a) + R+(b)
            relu = torch.maximum(zero, x)
            inv_relu = torch.minimum(zero, x)
            out = torch.roll(inv_relu, shift, dims=1) + relu
        else:
            out = SymSumCudaFunction.apply(x)                
        return out
