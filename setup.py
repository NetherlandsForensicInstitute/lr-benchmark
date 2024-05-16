from pathlib import Path

from setuptools import find_packages, setup


long_description = (Path(__file__).parent / 'README.md').read_text()


dependencies = (
    'confidence',
    'pytest',
    'matplotlib',
    'scikit-learn',
    'lir',
    'pandas',
    'xgboost',
    'more-itertools',
    'streamlit',
)


setup(
    name='lrbenchmark',
    version='0.2',
    author='Netherlands Forensics Institute',
    description='Benchmarking Likelihood Ratio systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=dependencies,
)
