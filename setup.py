import os
from setuptools import setup, find_packages

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "snakenet",
    version = "0.1",
    author = "Riley McDowell",
    author_email = "mcdori02@luther.edu",
    description = "A playable implementation of the 'Snake' game, complete" \
                  "complete with neuralnet player",
    license = "MIT",
    packages=find_packages(),
    long_description=read('README.md'),
    entry_points={'console_scripts': ['snake=snakenet.main:main']}
)
