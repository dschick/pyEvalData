from setuptools import setup, find_packages

setup(
    name='pyEvalData',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/dschick/pyEvalData',
    install_requires=['numpy',
                      'matplotlib',
                      'lmfit',
                      'scipy',
                      'uncertainties'],
    extras_require={
        'xrayutilities':  ['xrayutilities'],
        'testing': ['flake8', 'pytest'],
        'documentation': ['sphinx', 'nbsphinx', 'sphinxcontrib-napoleon'],
    },
    license='MIT',
    author='Daniel Schick',
    author_email='schick.daniel@gmail.com',
    description='Python Modul to evaluate experimental data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.5',
    keywords='data evaluation',
)
