"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyEvalData',
    version='1.0',
    packages=['pyEvalData'],
    url='https://github.com/dschick/pyEvalData',  # Optional
    install_requires=['numpy', 'matplotlib', 'xrayutilities', 'scipy', 'uncertainties'],  # Optional
    license='',
    author='Daniel Schick',
    author_email='schick.daniel@gmail.com',
    description='Python Modul to evaluate SPEC data and Dectris Pilatus reciprocal space maps',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
)