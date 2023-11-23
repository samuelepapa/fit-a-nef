# build extensions by running: 'python setup.py build_ext --inplace'
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    "libmcubes.mcubes",
    sources=["libmcubes/mcubes.pyx", "libmcubes/pywrapper.cpp", "libmcubes/marchingcubes.cpp"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    include_dirs=[numpy.get_include()],
)

# mise (efficient mesh extraction)
mise_module = Extension(
    "libmise.mise", sources=["libmise/mise.pyx"], include_dirs=[numpy.get_include()]
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    "libsimplify.simplify_mesh",
    sources=["libsimplify/simplify_mesh.pyx"],
    include_dirs=[numpy.get_include()],
)

# Gather all extension modules
ext_modules = [
    mcubes_module,
    mise_module,
    simplify_mesh_module,
]

setup(ext_modules=cythonize(ext_modules), cmdclass={"build_ext": BuildExtension})
