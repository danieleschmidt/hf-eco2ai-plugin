#!/usr/bin/env python3
"""
HF Eco2AI Plugin - Production Setup
TERRAGON Labs Enterprise Carbon Tracking for ML Training
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hf-eco2ai-plugin",
    version="1.0.0",
    author="TERRAGON Labs",
    author_email="enterprise@terragonlabs.com",
    description="Enterprise-grade Hugging Face CO₂ tracking with advanced carbon intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/hf-eco2ai-plugin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
)
