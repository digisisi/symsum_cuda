from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from shutil import copyfile

copyfile('symsum_cuda_kernel.cpp', 'symsum_cuda_kernel.cu')

setup(
    name='symsum',
    ext_modules=[
        CUDAExtension('symsum_cuda', [
            'symsum_cuda.cpp',
            'symsum_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
