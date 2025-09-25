"""
Setup script for multimodal ASR package
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-asr",
    version="0.1.0",
    author="NgDMau",
    author_email="author@example.com",
    description="A multimodal approach to Automatic Speech Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NgDMau/multimodal_asr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-asr-train=multimodal_asr.scripts.train:main",
            "multimodal-asr-eval=multimodal_asr.scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "multimodal_asr": ["data/*.json", "configs/*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/NgDMau/multimodal_asr/issues",
        "Source": "https://github.com/NgDMau/multimodal_asr",
        "Documentation": "https://github.com/NgDMau/multimodal_asr#readme",
    },
)