from importlib.util import find_spec
import sys

package_req = 'rawio'
spec = find_spec(package_req)

if spec is None:
    HAS_RAWIO = False
else:
    HAS_RAWIO = True

package_req = 'trigsim'
spec = find_spec(package_req)

if spec is None:
    HAS_TRIGSIM = False
else:
    HAS_TRIGSIM = True

del find_spec
del sys
del package_req
del spec
