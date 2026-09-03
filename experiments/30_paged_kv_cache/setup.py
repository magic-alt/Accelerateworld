from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="accelerateworld_paged_kv_cache_cuda",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_paged_kv_cache_cuda",
            sources=["paged_kv_extension.cpp", "paged_kv_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
