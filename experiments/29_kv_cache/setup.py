from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="accelerateworld_kv_cache_cuda",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_kv_cache_cuda",
            sources=["kv_cache_extension.cpp", "kv_cache_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
