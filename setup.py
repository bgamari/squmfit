#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "squmfit",
    version = "0.1",
    author = 'Ben Gamari',
    author_email = 'ben@smart-cactus.org',
    url = 'http://github.com/bgamari/squmfit',
    license = 'BSD',
    packages = find_packages(),
    install_requires=[
        'numpy',
    ],
)
