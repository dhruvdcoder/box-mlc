from setuptools import setup, find_packages
import os

# References:
#   1. AllenNLP: https://github.com/allenai/allennlp/blob/main/setup.py
#   2. audreyfeldro/cookiecutter-pypackage

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import package whilst setting up.

VERSION = {}  # type: ignore
with open("src/box_mlc/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


PATH_ROOT = os.path.dirname(__file__)

with open("README.md", "r") as fh:
    long_description = fh.read()


def load_requirements(path_dir=PATH_ROOT, comment_char="#"):
    with open(os.path.join(path_dir, "core_requirements.txt"), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []

    for ln in lines:
        # filer all comments

        if comment_char in ln:
            ln = ln[: ln.index(comment_char)]

        if ln:  # if requirement is not empty
            reqs.append(ln)

    return reqs


install_requires = load_requirements()

setup(
    name="box_mlc",
    version=VERSION["VERSION"],
    author="Dhruvesh Patel",
    author_email="1793dnp@gmail.com",
    description="Multilabel classification using box embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="box-mlc.readthedocs.io",
    project_urls={
        "Documentation": "box-mlc.readthedocs.io",
        "Source Code": "https://github.com/dhruvdcoder/box-mlc",
    },
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
        ],
    ),
    package_dir={"": "src"},
    install_requires=install_requires,
    keywords=['ML', 'AI', 'Box Embeddings'],
    entry_points={
        "console_scripts": [
            "box_mlc=box_mlc.__main__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",

        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9","Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],

  python_requires=">=3.7, <=3.9"

)
