"""setup.py for MindSync package."""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mindsync",
    version="1.0.0",
    author="MindSync Research Team",
    description=(
        "MindSync: A Multimodal AI Framework for Mental Health Monitoring "
        "via Fine-Grained Emotion Recognition and Cross-Modal Incongruence Detection"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[author]/mindsync-mental-health",
    packages=find_packages(exclude=["tests*", "scripts*", "notebooks*", "app*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.1",
        "torchaudio>=2.0.2",
        "transformers>=4.30.2",
        "datasets>=2.13.1",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy>=1.24.3",
        "scipy>=1.11.1",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "tqdm>=4.65.0",
        "PyYAML>=6.0.1",
    ],
    extras_require={
        "demo": ["gradio>=3.39.0", "streamlit>=1.25.0"],
        "logging": ["wandb>=0.15.5", "tensorboard>=2.13.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.7.0", "flake8>=6.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
