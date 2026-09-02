from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="accelerateworld_cuda",
    py_modules=["accelerateworld_ops"],
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_cuda",
            sources=["extension.cpp", "silu_mul_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
