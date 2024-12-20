from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "8"
    common_nvcc_args = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=" + nvcc_threads
    ]
    return nvcc_extra_args + common_nvcc_args

ext_modules = [
    CUDAExtension(
        name='cuda_qjl_score',
        sources=['qjl_kernel/csrc/qjl_score_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_qjl_quant',
        sources=['qjl_kernel/csrc/qjl_quant_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_qjl_gqa_score',
        sources=['qjl_kernel/csrc/qjl_gqa_score_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='quantization',
        sources=['qjl_kernel/csrc/quantization.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
            "nvcc": append_nvcc_threads([
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__"
            ])
        }
    )
]

setup(
    name='combined_cuda_extensions',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=["torch"]
)
