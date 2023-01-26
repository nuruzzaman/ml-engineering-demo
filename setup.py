from os import path as op

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

here = op.abspath(op.dirname(__file__))

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt', 'rt') as fh:
    requirements = fh.read().splitlines()

setup(
    name="ml-engineering",
    version="0.0.1",
    author="Nur",
    author_email="mohammad.nuruzzaman@ausloans.com.au",
    description="Automated Machine Learning in production",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nuruzzaman/ml-engineering-demo",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: Nur License",
        'Development Status :: 1 - Staging/Stable',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements
)
