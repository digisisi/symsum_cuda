#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define LESSTHANZERO(x) x < 0


template <typename scalar_t>
__global__ void symsum_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    const int half_batch_stride) {
    // batch index
    const int n = blockIdx.y;
    // complementary element pair indices
    const int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i2 = i1 + half_batch_stride;

    if (i2 < input.size(1)){
        scalar_t v1 = input[n][i1];
        scalar_t v2 = input[n][i2];       
        
        auto b1 = LESSTHANZERO(v1), b2 = LESSTHANZERO(v2);
        output[n][i1] = b1 ? (b2 ? v2 : 0.) : (b2 ? v1+v2 : v1);
        output[n][i2] = b1 ? (b2 ? v1 : v1+v2) : (b2 ? 0. : v2);         

       /* 
       // Logically equialent to the lines above. It is kept for reference.
        if (LESSTHANZERO(v1)) {
            if (LESSTHANZERO(v2)){
                output[n][i1] = v2;
                output[n][i2] = v1;                    
            }
            else{
                output[n][i1] = 0.;
                output[n][i2] = v1 + v2;
            }
        }
        else{
            if (LESSTHANZERO(v2)){
                output[n][i1] = v1 + v2;
                output[n][i2] = 0.;
            }
            else{
                output[n][i1] = v1;
                output[n][i2] = v2;                    
            }                
        }        
        */
    }
}

torch::Tensor symsum_cuda_forward(torch::Tensor input){
    const int batch_size = input.size(0);
    const int batch_stride = input.strides()[0];      
    const int half_batch_stride = batch_stride / 2;
    auto output = torch::empty_like(input);
    auto input_flat = input.flatten(1);
    auto output_flat = output.flatten(1);
    
    const int threads = 1024;
    const dim3 blocks((half_batch_stride + threads - 1) / threads, batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "symsum_forward_cuda", ([&] {
    symsum_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input_flat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        output_flat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        half_batch_stride
    );
    }));
    return output;    
}

template <typename scalar_t>
__global__ void symsum_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_grad,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_grad,
    const int half_batch_stride) {
    // batch index
    const int n = blockIdx.y;
    // complementary element pair indices
    const int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i2 = i1 + half_batch_stride;
    if (i2 < input.size(1)){
        scalar_t v1 = input[n][i1];
        scalar_t v2 = input[n][i2];        
        input_grad[n][i1] = LESSTHANZERO(v1) ? output_grad[n][i2] : output_grad[n][i1];
        input_grad[n][i2] = LESSTHANZERO(v2) ? output_grad[n][i1] : output_grad[n][i2];      
    }
}

torch::Tensor symsum_cuda_backward(torch::Tensor input, torch::Tensor output_grad){
    /* Computes `input_grad` */
    const int batch_size = input.size(0);
    const int batch_stride = input.strides()[0];      
    const int half_batch_stride = batch_stride / 2;
    auto input_flat = input.flatten(1);
    auto output_grad_flat = output_grad.flatten(1);
    auto input_grad = torch::empty_like(input);
    auto input_grad_flat = input_grad.flatten(1);
    
    const int threads = 1024;
    const dim3 blocks((half_batch_stride + threads - 1) / threads, batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "symsum_forward_cuda", ([&] {
    symsum_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        input_flat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        output_grad_flat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        input_grad_flat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        half_batch_stride
    );
    }));
    
    return input_grad;    
}