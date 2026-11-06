from setuptools import find_packages, setup

setup(
    name="arsl-overlay",
    version="1.0.0",
    description="Real-Time Arabic Sign Language Translation Overlay System",
    author="ArSL Project Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "PySide6>=6.5.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "mss>=9.0.0",
        "pywin32>=306; sys_platform == 'win32'",
        "Pillow>=10.0.0",
        "mediapipe>=0.10.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "ml": [
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
        ],
        "dl": [
            "torch>=2.0.0",
            "onnxruntime>=1.15.0",
            "ultralytics>=8.0.0",
        ],
        "arabic": [
            "arabic-reshaper>=2.1.0",
            "python-bidi>=0.4.0",
        ],
        "monitoring": [
            "psutil>=5.9.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
        "all": [
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "torch>=2.0.0",
            "onnxruntime>=1.15.0",
            "arabic-reshaper>=2.1.0",
            "python-bidi>=0.4.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "arsl-overlay=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Display",
    ],
)
