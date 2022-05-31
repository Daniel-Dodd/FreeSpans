from setuptools import find_packages, setup


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires

setup(
    name="Spans",
    version="0.1",
    author="Daniel Dodd",
    author_email="d.dodd1@lancaster.ac.uk",
    packages=find_packages("."),
    license="LICENSE",
    description="Span package.",
    long_description=(
        "Package for spans."
    ),
    install_requires=parse_requirements_file("requirements.txt"),
    keywords=["spans jax"],
)
