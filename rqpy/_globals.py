from importlib.util import find_spec
import sys

package_req = 'scdmsPyTools'

spec = find_spec(package_req)

if spec is None:
    HAS_SCDMSPYTOOLS = False
else:
    HAS_SCDMSPYTOOLS = True

del find_spec
del sys
del package_req
del spec
