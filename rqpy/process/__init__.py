from importlib.util import find_spec
import sys

package_req = 'scdmsPyTools'

spec = find_spec(package_req)

if spec is None:
    print("scdmsPyTools is not installed, cannot import the process submodule")
else:
    from . import rq
    from .rq import *

del find_spec
del sys
del package_req
del spec
