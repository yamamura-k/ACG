from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

source = ["_cluster_coef.c", "./cluster_coef.pyx"]

setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[Extension("cluster_coef", source, language="c")],
    include_dirs=[np.get_include()],
)

source = ["_cluster_coef_2.cpp", "./cluster_coef_2.pyx"]

setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[
        Extension(
            "cluster_coef_2", source, language="c++", extra_compile_args=["-std=c++11"]
        )
    ],
    include_dirs=[np.get_include()],
)

source = [
    "_cluster_coef_for_weighted_graphs.c",
    "./cluster_coef_for_weighted_graphs.pyx",
]

setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[Extension("cluster_coef_for_weighted_graphs", source, language="c")],
    include_dirs=[np.get_include()],
)
