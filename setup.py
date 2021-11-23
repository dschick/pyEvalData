from setuptools import setup, find_packages

setup(
    name='pyEvalData',
    version='1.3.7',
    packages=find_packages(),
    url='https://github.com/dschick/pyEvalData',
    install_requires=['numpy',
                      'matplotlib',
                      'lmfit',
                      'scipy',
                      'uncertainties',
                      'xrayutilities',
                      'h5py>=3.0',
                      'nexusformat'],
    extras_require={
        'testing': ['flake8', 'pytest'],
        'documentation': ['sphinx', 'nbsphinx', 'sphinxcontrib-napoleon'],
    },
    license='MIT',
    author='Daniel Schick',
    author_email='schick.daniel@gmail.com',
    description='Python Modul to evaluate experimental data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    package_data={
        'pyEvalData': ['*.conf']
    },
    python_requires='>=3.5',
    keywords='data evaluation',
)
