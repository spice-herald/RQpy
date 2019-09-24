import os
import glob
import shutil
from setuptools import find_packages, Command

from numpy.distutils.core import Extension, setup


upperlim_files = [
    'upperlim.pyf',
    'UpperLim.f',
    'y_vs_CLf.f',
    'CMaxinf.f',
    'ConfLev.f',
    'Cinf.f',
    'CERN_Stuff.f',
]

upperlim_f77_paths = []
for fname in upperlim_files:
    upperlim_f77_paths.append(f"rqpy/limit/_upperlim/{fname}")

ext1 = Extension(
    name='rqpy.limit._upperlim.upperlim',
    sources=upperlim_f77_paths,
)

upperlim_data_files = [
    'CMaxf.in',
    'CMaxfLow.in',
    'ymintable.in',
    'y_vs_CLf.in',
    'CLtable.txt',
]

upperlim_data_paths = []
for fname in upperlim_data_files:
    upperlim_data_paths.append(f"rqpy/limit/_upperlim/{fname}")


upper_files = [
    'upper.pyf',
    'Upper.f',
    'UpperLim.f',
    'C0.f',
    'CnMax.f',
    'CombConf.f',
    'y_vs_CLf.f',
    'CMaxinf.f',
    'ConfLev.f',
    'Cinf.f',
    'CERN_Stuff.f',
]

upper_f77_paths = []
for fname in upper_files:
    upper_f77_paths.append(f"rqpy/limit/_upper/{fname}")

ext2 = Extension(
    name='rqpy.limit._upper.upper',
    sources=upper_f77_paths,
)

upper_data_files = [
    'CMaxf.txt',
    'CMax.txt',
    'Cm.txt',
    'ymintable.txt',
    'y_vs_CLf.txt',
]

upper_data_paths = []
for fname in upper_data_files:
    upper_data_paths.append(f"rqpy/limit/_upper/{fname}")

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
    description="DM Search Analysis and Processing Tools", 
    author="Samuel Watkins, Caleb Fink", 
    author_email="samwatkins@berkeley.edu, cwfink@berkeley.edu", 
    url="https://github.com/ucbpylegroup/RQpy", 
    packages=find_packages(), 
    zip_safe=False,
    cmdclass={
        'clean': CleanCommand,
    },
    data_files=[
        ('rqpy/limit/_upperlim/', upperlim_data_paths),
        ('rqpy/limit/_upper/', upper_data_paths),
    ],
    ext_modules=[
        ext1,
        ext2,
    ],
)
