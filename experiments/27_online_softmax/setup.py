from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="accelerateworld_online_softmax_cuda",
    ext_modules=[
        CUDAExtension(
            name="accelerateworld_online_softmax_cuda",
            sources=["softmax_extension.cpp", "softmax_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
