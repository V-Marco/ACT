from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ACT",
    version="0.0.1",
    author="Vladimir Omelyusik",
    author_email="vovwm@umsystem.edu",
    description="Automatic Cell Tuner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/V-Marco/ACT",
    download_url="https://github.com/V-Marco/ACT/archive/refs/heads/main.zip",
    license="MIT",
    install_requires=["numpy", "matplotlib", "scipy", "torch"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests"]),
)
