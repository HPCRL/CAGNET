from setuptools import setup, Extension
from torch.utils import cpp_extension
include_dirs = ['/uufs/chpc.utah.edu/sys/installdir/cuda/11.0.2/include']
setup(name='sparse_coo_tensor_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_coo_tensor_cpp', ['sparse_coo_tensor.cpp'],
                                                include_dirs=include_dirs,
                                                    extra_compile_args=["-lcusparse"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
