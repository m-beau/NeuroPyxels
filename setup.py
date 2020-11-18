
from setuptools import setup

setup(name='npix',
      version='1.0',
      description='Python routines dealing with Neuropixels data.',
      url='https://github.com/Npix-routines/NeuroPyxels',
      author='Maxime Beau',
      author_email='maximebeaujeanroch047@gmail.com',
      license='MIT',
      packages=['npix'],
      #package_dir={'rtn': 'rtn'},
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
      'six', 'elephant', 'vispy', 'statsmodels', 'progressbar2',
      'scikit-learn', 'umap-learn', 'networkx', 'psutil', 'pyqtgraph', 'imutils', 'opencv-contrib-python',
      'h5py', 'holoviews', 'hvplot'],
      keywords='neuropixels,kilosort,phy,data analysis,electrophysiology,neuroscience')
