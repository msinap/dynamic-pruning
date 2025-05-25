"""
Setup script for the prefix_checking package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prefix_checking",
    version="1.0.0",
    author="Dynamic Pruning Team",
    description="A package for validating function call prefixes in LLM outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "datasets>=2.0.0",
        "tqdm>=4.60.0",
        "transformers>=4.0.0",
        "torch>=1.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "filter-dataset=prefix_checking.filter_dataset:main",
            "verify-dataset=prefix_checking.verify_filtered_dataset:main",
            "test-prefixes=prefix_checking.test_dataset_prefixes:main",
        ],
    },
) 