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
    name='lrbenchmark',
    version='0.1',
    author='Netherlands Forensics Institute',
    packages=find_packages(),
    install_requires=dependencies,
)
