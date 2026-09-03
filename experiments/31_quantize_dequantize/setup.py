from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="accelerateworld_quant_cuda",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_quant_cuda",
            sources=["quant_extension.cpp", "quant_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
