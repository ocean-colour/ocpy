
# Standard imports
import glob, os
from setuptools import setup, find_packages


# Begin setup
setup_keywords = dict()
setup_keywords['name'] = 'ocpy'
setup_keywords['description'] = 'Ocean Color Analysis -- Here, there, and everywhere'
setup_keywords['author'] = 'J. Xavier Prochaska et al.'
setup_keywords['author_email'] = 'jxp@ucsc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/ocean-colour/ocpy'
setup_keywords['version'] = '0.1.dev0'
# Use README.rst as long_description.
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>3.8.0)']
setup_keywords['install_requires'] = [
    'seaborn', 'pyarrow', 'healpy', 'cftime', 'bokeh',
    'xarray', 'h5netcdf', 'importlib-metadata', 'scikit-learn',
    'openpyxl', 'pyproj', 'cartopy', 'netcdf4', 'geopy',
    'geopandas', 'ipython']
setup_keywords['zip_safe'] = False
setup_keywords['packages'] = find_packages()
# Ship the bundled data files (ocpy/data/**) in both the sdist and the wheel
# so importlib.resources.files('ocpy') resolves them on a fresh pip install,
# not just from a source checkout. include_package_data + MANIFEST.in covers
# the sdist; package_data forces inclusion in the wheel even though ocpy/data
# is not itself a package (it has no __init__.py).
setup_keywords['include_package_data'] = True
setup_keywords['package_data'] = {'ocpy': ['data/**/*']}
setup_keywords['tests_require'] = ['pytest']

if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if not os.path.basename(fname).endswith('.rst')]

setup(**setup_keywords)
