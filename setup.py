#!/usr/bin/env python
# -*- coding: utf-8 -*-


from os.path import join
import numpy

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

include_dirs = [numpy.get_include()]

hmm_src = join("mlpy", "libs", "hmmc")
classifier_src = join("mlpy", "libs", "classifier")

ext_modules = []
ext_modules += [Extension("classifier",
                          sources=[join(classifier_src, "classifier_module.cc"),
                                   join(classifier_src, "classifier.cc"),
                                   join(classifier_src, "c45tree.cc"),
                                   join(classifier_src, "random.cc"),
                                   join(classifier_src, "coord.cc"),
                                   join(classifier_src, "array_helper.cc")],
                          include_dirs=include_dirs), ]

ext_modules += [Extension("hmmc",
                          sources=[join(hmm_src, "hmmc_module.c"),
                                   join(hmm_src, "hmm.c")],
                          include_dirs=include_dirs), ]


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy>=1.6.2',
    'scipy>=0.11',
    'matplotlib',
    'scikit-learn',
    'six>=1.9.0',
]

test_requirements = [
    'pytest',
]

import mlpy

setup(
    name='mlpy',
    version=mlpy.__version__,
    description="A machine learning library for Python",
    long_description=readme + '\n\n' + history,
    author="Astrid Jackson",
    author_email='ajackson@eecs.ucf.edu',
    url='https://readthedocs.org/builds/mlpy/',
    download_url='https://github.com/evenmarbles/mlpy',
    ext_package="mlpy.libs",
    ext_modules=ext_modules,
    packages=[
        'mlpy',
    ],
    package_dir={'mlpy':
                 'mlpy'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='machine learning,intelligent agents',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
