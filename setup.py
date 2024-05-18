from setuptools import setup, find_packages

setup(
    name="tinyagent",
    version="0.1.0",
    author="y-lan",
    author_email="lanyuyang@gmail.com",
    description="A minimalistic agent framework",
    long_description=open("README.md").read(),
    url="https://github.com/y-lan/tinyagent",
    packages=find_packages(),
    install_requires=["pydantic>=2.7.1", "requests>=2.31.0"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
