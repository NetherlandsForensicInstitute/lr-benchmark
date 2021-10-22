from setuptools import find_packages, setup


dependencies = (
    'confidence',
    'pytest',
    'matplotlib',
    'sklearn',
    'lir',
    'pandas',
    'xgboost',
    'more-itertools'
)


setup(
    name='LRbenchmark',
    version='0.0',
    author='Netherlands Forensics Institute',
    packages=find_packages(),
    install_requires=dependencies,
)
