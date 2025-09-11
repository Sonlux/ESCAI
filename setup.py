#!/usr/bin/env python3
"""Setup script for ESCAI Framework."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    """Read README.md file."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "ESCAI Framework - Epistemic State Cognitive AI Framework"

# Read requirements
def read_requirements():
    """Read requirements.txt file."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "pydantic>=2.0.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ]

setup(
    name="escai-framework",
    version="0.1.0",
    author="ESCAI Team",
    author_email="team@escai.dev",
    description="ESCAI Framework for epistemic state management and cognitive AI monitoring",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/escai-team/ESCAI",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-html>=3.1.0",
            "pytest-xdist>=3.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
            "pre-commit>=3.3.0",
        ],
        "full": [
            "scikit-learn>=1.3.0",
            "scipy>=1.10.0",
            "networkx>=3.0",
            "matplotlib>=3.7.0",
            "plotly>=5.15.0",
            "streamlit>=1.25.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "escai=escai_framework.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)