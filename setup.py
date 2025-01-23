#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: setup.py
Description: Used to set up Sweet2Plus as a package
Author: David Estrin
Version: 1.0
Date: 11-14-2024
"""

from setuptools import setup, find_packages

setup(
    name='Sweet2PPlus',
    version='0.1',
    packages=find_packages(),  # Automatically find subfolder1 and subfolder2 as packages.
    install_requires=['numpy',
                      'matplotlib',
                      'pandas',
                      'ipdb',
                      'suite2p',
                      'seaborn',
                      'scikit-learn',
                      'tqdm',
                      'opencv-python',
                      'pillow',
                      'tiffile',
                      'optuna',
                      'watchdog',
                      'statsmodels'],
    author='David Estrin',
    author_email='',
    description='A short description of your project',
    url='https://github.com/DJESTRIN/Sweet2PPlus',  # Replace with your repository URL.
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
