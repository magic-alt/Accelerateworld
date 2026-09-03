from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="accelerateworld_weight_only_gemm_cuda",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_weight_only_gemm_cuda",
            sources=["weight_only_extension.cpp", "weight_only_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
