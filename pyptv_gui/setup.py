from setuptools import setup, find_packages
from codecs import open
from os import path
from glob import glob

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyptv',
    version='0.1',

    description='Python GUI for OpenPTV',
    long_description=long_description,

    author='Denis Lepchev and Alex Liberzon',
    author_email='particle.tracking@gmail.com',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords=('3DPTV', 'PTV', 'OpenPTV'),

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    install_requires=['numpy', 'scipy', 'chaco', 'traits', 'pyface', 'enable'],

    # Will list examples once I add them in.
    # package_data={
    #    'sample': ['package_data.dat'],
    # },
    # data_files=[('my_data', ['data/data_file'])],

    # This can be done but creates a mess. Let's run from the tarball until
    # there's a better idea.
    # scripts=glob('ptv/*') + glob('evolution/*')
)