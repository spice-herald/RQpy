from importlib.util import find_spec
import sys

package_req = 'rawio'
spec = find_spec(package_req)

if spec is None:
    HAS_RAWIO = False
else:
    HAS_RAWIO = True

package_req = 'scdmsPyTools'
spec = find_spec(package_req)

if not HAS_RAWIO and spec is not None:
    try:
        from scdmsPyTools.BatTools.IO import getRawEvents
        HAS_SCDMSPYTOOLS = True
        import warnings
        warnings.warn(
            "Support for using scdmsPyTools to access CDMS IO functions will "
            "be deprecated in a future release. Consider switching to the "
            "CDMS package `rawio`."
        )
        del warnings
        del getRawEvents
    except:
        HAS_SCDMSPYTOOLS = False
else:
    HAS_SCDMSPYTOOLS = False

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
