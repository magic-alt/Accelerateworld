from __future__ import annotations

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="accelerateworld-swiglu-cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_swiglu_cuda",
            sources=["swiglu_extension.cpp", "swiglu_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
