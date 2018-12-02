import numpy as np



def _bindata(arr,xrange = None, bins = 'sqrt'):
    """
    Helper function to convert 1d array into binned (x,y) data
    
    Parameters
    ----------
    arr: array
        Input array
    xrange: tuple, optional
        Range over which to bin data
    bins: int or str, optional
        Number of bins, or type of automatic binning scheme 
        (see numpy.histogram())
    
    Returns
    -------
    x: array
        Array of x data
    y: array
        Array of y data
    bins: array
        Array of bins returned by numpy.histogram()
    
    """
    
    if xrange is not None:
        y, bins = np.histogram(arr,bins = bins, range = xrange)
    else:
        y, bins = np.histogram(arr,bins = bins)
    x = (bins[1:]+bins[:-1])/2
    return x, y, bins