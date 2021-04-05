from setuptools import setup, find_packages


setup(

    name="libhip",
    version="0.0.1",
    url="https://github.com/slacknate/libhip",
    description="A library for extracting HIP file images.",
    packages=find_packages(include=["libhip", "libhip.*"]),
    install_requires=["Pillow==8.2.0"]
)
