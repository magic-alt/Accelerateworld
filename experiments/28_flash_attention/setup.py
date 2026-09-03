from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="accelerateworld_flash_attention_cuda",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_flash_attention_cuda",
            sources=["attention_extension.cpp", "attention_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
