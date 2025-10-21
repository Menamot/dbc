from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="dbc-menamot",
    version="2.0.1",
    description="A Python package for Discrete Bayesian and Minmax Classifiers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Wenlong",
    author_email="menamot.chen@gmail.com",
    url="https://github.com/Menamot/dbc",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    license="MIT",
)
