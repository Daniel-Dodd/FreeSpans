from setuptools import find_packages, setup
import os
import io
import re


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [line.strip() for line in fid.readlines() if line]
    return requires


# Optional Packages
EXTRAS = {
    "dev": [
        "black",
        "isort",
        "pylint",
        "flake8",
    ],
    "tests": [
        "pytest",
    ],
    "docs": [
        "furo",
    ],
}


# Get version number (function from GPyTorch)
def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("freespans", "__init__.py")
readme = open("README.md").read()


setup(
    name="FreeSpans",
    version=version,
    author="Daniel Dodd",
    author_email="d.dodd1@lancaster.ac.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Probabilistic modelling and optimal design for free-spans in Jax.",
    long_description=readme,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://http://freespans.readthedocs.io/",
        "Source": "https://github.com/daniel-dodd/FreeSpans",
    },
    install_requires=parse_requirements_file("requirements.txt"),
    extras_require=EXTRAS,
    keywords=["free-spans, gaussian-processes jax machine-learning bayesian"],
)
