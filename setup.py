from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eovot-benchmark",
    version="0.1.0",
    description="Edge-Optimized Visual Object Tracking Benchmark Suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EOVOT Contributors",
    license="MIT",
    packages=find_packages(exclude=["tests*", "scripts*", "configs*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "opencv-python>=4.5",
        "psutil>=5.9",
        "pandas>=1.3",
        "pyyaml>=6.0",
        "tqdm>=4.62",
    ],
    extras_require={
        "torch": ["torch>=1.13"],
        "onnx": ["onnxruntime>=1.14"],
    },
    entry_points={
        "console_scripts": [
            "eovot=scripts.run_benchmark:main",
        ],
    },
)
