"""
Setup script for DPO Implementation
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dpo-implementation",
    version="1.0.0",
    author="Atlas - LegendEvent AI",
    description="Direct Preference Optimization (DPO) implementation for LLM alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["main"],
    entry_points={
        "console_scripts": [
            "dpo-train=main:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
