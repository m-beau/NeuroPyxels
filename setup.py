
from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements=['ipython', 'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
      'six', 'vispy', 'statsmodels', 'progressbar2', 'cmcrameri',
      'scikit-learn', 'umap-learn', 'networkx', 'psutil', 'imutils', 'python_utils',
      'h5py', 'numba', 'tk', 'urllib3', 'certifi', 'idna', 'elephant', 'neo',
      'opencv-python']

setup(name='npyx',
      version='1.6.2',
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
