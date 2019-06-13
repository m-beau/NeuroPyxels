
from setuptools import setup

setup(name='rtn',
      version='1.0',
      description='Maxime Beau personal python routines.',
      url='https://github.com/m-beau/routines',
      author='Maxime Beau',
      author_email='maximebeaujeanroch047@gmail.com',
      license='MIT',
      packages=['rtn', 'rtn.npix'],
      #package_dir={'rtn': 'rtn'},
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
      'six', 'elephant', 'vispy', 'statsmodels', 'progressbar',
      'scikit-learn', 'umap-learn', 'networkx'],
      dependency_links=['https://github.com/llerussell/paq2py.git'],
      keywords='phy,data analysis,electrophysiology,neuroscience')
