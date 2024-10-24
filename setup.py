import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "stickbreaking_attention",
    version = "0.0.0",
    author = "Shawn Tan",
    author_email = "shawntan@ibm.com",
    description = "Triton implementation of Stick-breaking attention",
    license = "Apache License",
    keywords = "triton pytorch llm stickbreaking attention",
    url = "https://github.com/shawntan/scattermoe",
    packages=find_packages(),
    long_description=read('README.md'),
    python_requires='>=3.10.10',
    install_requires=[
        'torch',
        'triton',
    ],
    tests_require=['pytest'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
    ],
)

