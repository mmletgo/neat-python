"""
neat-python 构建配置 - 支持 Cython 优化模块

使用方法：
    # 开发模式安装（推荐）
    pip install -e ".[cython]"

    # 构建 Cython 扩展
    python setup.py build_ext --inplace
"""

import os
import sys
from setuptools import setup, Extension

# 尝试导入 Cython
USE_CYTHON = True
try:
    from Cython.Build import cythonize
    import numpy
except ImportError:
    USE_CYTHON = False
    cythonize = None
    print("Warning: Cython or NumPy not found. Building without Cython optimization.")

# Cython 扩展模块定义
def get_extensions():
    """获取 Cython 扩展模块列表"""
    if not USE_CYTHON:
        return []

    # 基础目录
    cython_dir = os.path.join("neat", "_cython")

    # 检查 .pyx 文件是否存在
    pyx_files = [
        "random_utils.pyx",
        "fast_attributes.pyx",
        "fast_genes.pyx",
        "fast_genome.pyx",
        "fast_graphs.pyx",
        "fast_network.pyx",
        "fast_gene_factory.pyx",
        "fast_innovation.pyx",
    ]

    extensions = []
    for pyx_file in pyx_files:
        pyx_path = os.path.join(cython_dir, pyx_file)
        if os.path.exists(pyx_path):
            module_name = f"neat._cython.{pyx_file[:-4]}"  # 去掉 .pyx
            extensions.append(
                Extension(
                    module_name,
                    [pyx_path],
                    include_dirs=[numpy.get_include()],
                    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                )
            )

    return extensions


# 编译器指令
COMPILER_DIRECTIVES = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
}


def build_extensions():
    """构建 Cython 扩展"""
    extensions = get_extensions()
    if extensions and USE_CYTHON:
        return cythonize(
            extensions,
            compiler_directives=COMPILER_DIRECTIVES,
            annotate=False,  # 设为 True 可生成 HTML 分析文件
        )
    return []


# 仅在直接运行 setup.py 时构建扩展
if __name__ == "__main__":
    setup(
        ext_modules=build_extensions(),
    )
