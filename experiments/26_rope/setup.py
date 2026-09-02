from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="accelerateworld_rope_cuda",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_rope_cuda",
            sources=["rope_extension.cpp", "rope_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
