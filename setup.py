import codecs
import os.path as op

from setuptools import setup


def read(rel_path):
    here = op.abspath(op.dirname(__file__))
    with codecs.open(op.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "ipython",
    "numpy",
    "scipy",
    "pandas",
    "numba",
    "statsmodels",
    "matplotlib",
    "cmcrameri",
    "scikit-learn",
    "imbalanced-learn",
    "networkx",
    "psutil",
    "joblib>=1.3",
    "tqdm",
    "h5py",
    "seaborn",
    "cachecache"
]

c4_requirements = [
    "torch==1.13.1",
    "torchvision==0.14.1",
    "torchaudio==0.13.1",
    "asdfghjkl==0.1a2",
    "laplace-torch==0.1a2",
    "backpack-for-pytorch==1.6.0",
    "requests",
    "dill",
    "tabulate",
    "scipy==1.10.0", # Constrain scipy version for compatibility with pytorch pinned version
    "numpy<2.0.0" # Constrain numpy version for compatibility with pytorch pinned version
]
dependency_links = ["https://download.pytorch.org/whl/cpu"]

entry_points = {"console_scripts": ["predict_cell_types = npyx.c4.predict_cell_types:main", "c4 = npyx.c4.predict_cell_types:run_c4"]}

setup(
    name="npyx",
    version=get_version("npyx/__init__.py"),
    author="Maxime Beau",
    author_email="maximebeaujeanroch047@gmail.com",
    description="Utilities to load, process, and plot Neuropixels data in Python.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Npix-routines/NeuroPyxels",
    packages=["npyx", "npyx.c4"],
    install_requires=requirements,
    extras_require={"c4": c4_requirements},
    python_requires="<3.12",
    dependency_links=dependency_links,
    entry_points=entry_points,
    keywords="neuropixels,kilosort,phy,data analysis,electrophysiology,neuroscience",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
