#!/usr/bin/env python3
import numpy
from setuptools import setup, Extension

setup(
    name="bertype",
    version="0.1",
    packages=[
        "bertype",
        "bertype.extensions",
        "bertype.extensions._col2sent"
    ],
    py_modules=[
        "bertype.__init__",
        "bertype.classifiers",
        "bertype.column_info",
        "bertype.embedders",
        "bertype.engines",
        "bertype.utils",
    ],
    ext_modules=[
        Extension(
            "bertype.extensions._col2sent",
            sources=[
                "bertype/extensions/_col2sent/bind.c",
                "bertype/extensions/_col2sent/col2sent.c",
            ],
            include_dirs=[numpy.get_include()]
        ),
    ],
    package_dir={"bertype": "bertype"},
    install_requires=[
        'numpy',
        # 'sentence-transfomers'
    ],
    author="Pedro Fluxa",
    author_email="pedro@mindsdb.com",
    description="Package to infer column data types using BERT.",
    url="https://github.com/mindsdb/bertype",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
