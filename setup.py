#!/usr/bin/env python3

import os
import glob
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

scripts = glob.glob('bin/*')

with open(os.path.join('groco', '__version__.py')) as version_file:
    version = {}
    exec(version_file.read(), version)
    project_version = version['__version__']

setup(name='groco',
      version=project_version,
      description='Keras implementation of Group Convolutions',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='http://github.com/apjansen/groco',
      author='Aron Jansen',
      author_email='aronpjansen@gmail.com',
      license='MIT',
      zip_safe=False,
      include_package_data=True,
      packages=find_packages(),
      install_requires=[
        'tensorflow',
        'pytest'
        ],
    #   extras_require={'irods': ['python-irodsclient']},
    #   entry_points={'console_scripts':
    #                 ['arts_fix_fits_from_before_20200408=arts_tools.fits.fix_file_from_before_20200408:main',
    #                  'arts_find_pulsars_in_field=arts_tools.pulsars.find_pulsars:main',
    #                  'arts_download_from_alta=arts_tools.archive.download_from_alta:main',
    #                  'arts_psrdada_iquv_to_fits=arts_tools.iquv.psrdada_to_fits:main']},
    #   scripts=scripts,
    #   classifiers=['License :: OSI Approved :: Apache Software License',
    #                'Programming Language :: Python :: 3',
    #                'Operating System :: OS Independent'],
    #   python_requires='>=3.6'
      )