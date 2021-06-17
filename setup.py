from setuptools import setup
import codecs
import os.path as op

def read(rel_path):
    here = op.abspath(op.dirname(__file__))
    with codecs.open(op.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements=['ipython', 'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
      'six', 'vispy', 'statsmodels', 'progressbar2', 'cmcrameri',
      'scikit-learn', 'umap-learn', 'networkx', 'psutil', 'imutils', 'python_utils',
      'h5py', 'numba', 'tk', 'urllib3', 'certifi', 'idna', 'elephant', 'neo',
      'opencv-python']

setup(name='npyx',
      version=get_version("npyx/__init__.py"),
      author='Maxime Beau',
      author_email='maximebeaujeanroch047@gmail.com',
      description='Python routines dealing with Neuropixels data.',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='https://github.com/Npix-routines/NeuroPyxels',
      packages=['npyx'],
      #package_dir={'rtn': 'rtn'},
      install_requires=requirements,
      keywords='neuropixels,kilosort,phy,data analysis,electrophysiology,neuroscience',
      classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"]
    )
