
from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
      'six', 'elephant', 'vispy', 'statsmodels', 'progressbar2',
      'scikit-learn', 'umap-learn', 'networkx', 'psutil', 'pyqtgraph', 'imutils', 'opencv-contrib-python',
      'h5py', 'holoviews', 'hvplot', 'twine==1.13.0']

setup(name='npix',
      version='1.0',
      author='Maxime Beau',
      author_email='maximebeaujeanroch047@gmail.com',
      description='Python routines dealing with Neuropixels data.',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='https://github.com/Npix-routines/NeuroPyxels',
      packages=['npix'],
      #package_dir={'rtn': 'rtn'},
      install_requires=requirements,
      keywords='neuropixels,kilosort,phy,data analysis,electrophysiology,neuroscience',
      classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"]
    )
