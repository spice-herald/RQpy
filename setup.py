import os
import glob
import shutil
from setuptools import find_packages, Command

from numpy.distutils.core import Extension, setup


upperlim_files = ['upperlim.pyf',
                  'UpperLim.f',
                  'y_vs_CLf.f',
                  'CMaxinf.f',
                  'ConfLev.f',
                  'Cinf.f',
                  'CERN_Stuff.f',
                 ]

f77_paths = []
for fname in upperlim_files:
    f77_paths.append(f"rqpy/limit/_upperlim/{fname}")

ext1 = Extension(name='rqpy.limit._upperlim.upperlim',
                 sources=f77_paths,
                )

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        here = os.path.dirname(os.path.abspath(__file__))

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)

setup(
    name="RQpy", 
    version="0.1.0", 
    description="DM Search Analsis and Processing Tools", 
    author="Samuel Watkins, Caleb Fink", 
    author_email="samwatkins@berkeley.edu, cwfink@berkeley.edu", 
    url="https://github.com/berkeleycdms/RQpy", 
    packages=find_packages(), 
    zip_safe=False,
    cmdclass={
            'clean': CleanCommand,
            },
    package_data = {'': ['*.f', '*.pyf']}, 
    include_package_data = True,
    ext_modules=[ext1],
)
