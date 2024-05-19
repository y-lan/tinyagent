"""
This module sets up the tinyagent package for distribution.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tinyagent",
    version="0.1.4",
    author="y-lan",
    author_email="lanyuyang@gmail.com",
    description="A minimalistic agent framework",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/y-lan/tinyagent",
    packages=find_packages(),
    install_requires=["pydantic>=2.7.1", "requests>=2.31.0"],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
