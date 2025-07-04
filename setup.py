from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tennis-analysis-toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your-email@domain.com",
    description="A comprehensive Python toolkit for analyzing tennis serves using 3D keypoint tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tennis-analysis-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    package_data={
        "tennis_analysis": ["*.yaml", "*.json", "*.txt"],
    },
    entry_points={
        "console_scripts": [
            "tennis-analysis=tennis_analysis.cli:main",
        ],
    },
    keywords="tennis, sports analytics, 3D keypoints, biomechanics, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/tennis-analysis-toolkit/issues",
        "Source": "https://github.com/yourusername/tennis-analysis-toolkit",
        "Documentation": "https://tennis-analysis-toolkit.readthedocs.io/",
    },
) 