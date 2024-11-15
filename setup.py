from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="dbc-menamot",                       # 包名称
    version="0.1.0",                                   # 初始版本号
    description="A Python package for Discrete Bayesian and Minimax Classifiers",  # 简短描述
    long_description=open("README.md").read(),         # 详细描述
    long_description_content_type="text/markdown",
    author="Wenlong",
    author_email="menamot.chen@gmail.com",
    url="https://github.com/Menamot/dbc",  # GitHub 仓库地址
    packages=find_packages(),                          # 自动查找包
    install_requires=install_requires,
    classifiers=[                                      # 分类器
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',                           # Python 版本要求
    license="MIT",
)