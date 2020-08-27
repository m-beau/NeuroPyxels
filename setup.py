
from setuptools import setup

setup(name='rtn',
      version='2.0',
      description='Maxime Beau personal python routines.',
      url='https://github.com/m-beau/routines',
      author='Maxime Beau',
      author_email='maximebeaujeanroch047@gmail.com',
      license='MIT',
      packages=['rtn', 'rtn.npix'],
      #package_dir={'rtn': 'rtn'},
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
      'six', 'elephant', 'vispy', 'statsmodels', 'progressbar2',
      'scikit-learn', 'umap-learn', 'networkx', 'psutil', 'pyqtgraph', 'imutils', 'opencv-contrib-python',
      'h5py', 'holoviews[recommended]'],
      keywords='phy,data analysis,electrophysiology,neuroscience')
