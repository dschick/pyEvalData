#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
# Copyright (c) 2015-2020 Daniel Schick
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from scipy.stats import binned_statistic


def edges4grid(grid):
    """Creates a vector of the corresponding edges for a grid vector. """
    binwidth = np.diff(grid)
    edges = np.hstack([grid[0]-binwidth[0]/2, grid[0:-1] +
                       binwidth/2, grid[-1]+binwidth[-1]/2])

    return edges, binwidth


def bin_data(y, x, X, statistic='mean'):
    """
    Bin data y(x) on new grid X using a statistic type. 

    Args:
        y         (ndarray) : old y values
        x         (ndarray) : old x values
        X         (ndarray) : new grid
        statistic (str)     : statitic type, used for
                              scipy's binned_statistic


    Returns:
        Y     (ndarray)  : binned Y data without NaNs
        X     (ndarray)  : new X grid, same shape as Y
        Yerr  (ndarray)  : Error for Y, according to statistic, same shape as X and Y
        Xerr  (ndarray)  : Error for Y, according to statistic, same shape as X and Y
        Ystd  (ndarray)  : Std for Y, according to statistic, same shape as X and Y
        Xstd  (ndarray)  : Std for X, according to statistic, same shape as X and Y
        edges (ndarray)  : Edges of binned data, same shape as X and Y
        bins  (ndarray)  : Indices of the bins, same shape as X and Y
        n     (ndarray)  : Number of values per given bin, same shape as X and Y

    wrapper around scipy's binned_statistic
    for more info refer to:
    docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
    """

    y = y.flatten('F')
    x = x.flatten('F')
    X = np.sort(X.flatten('F'))

    # set non-finite values to 0
    idx = ~np.isfinite(y)
    y[idx] = 0
    # create bins for the grid
    edges, _ = edges4grid(X)

    if np.array_equal(x, X):
        # no binning since the new grid is the same as the old one
        Y = y
        bins = np.ones_like(Y)
        n = np.ones_like(Y)
    else:
        # do the binning and get the Y results
        Y, _, bins = binned_statistic(x, y, statistic, edges)
        bins = bins.astype(np.int_)

        n = np.bincount(bins[bins > 0], minlength=len(X)+1)
        n = n[1:len(X)+1]

    if np.array_equal(x, X) and statistic != 'sum':
        Ystd = np.zeros_like(Y)
        Xstd = np.zeros_like(X)
        Yerr = np.zeros_like(Y)
        Xerr = np.zeros_like(X)
    else:
        # calculate the std of X and Y
        if statistic == 'sum':
            Ystd = np.sqrt(Y)
            Yerr = Ystd
        else:
            Ystd, _, _ = binned_statistic(x, y, 'std', edges)
            Yerr = Ystd/np.sqrt(n)

        Xstd, _, _ = binned_statistic(x, x, 'std', edges)
        Xerr = Xstd/np.sqrt(n)

    # remove NaNs
    Y = Y[n > 0]
    X = X[n > 0]
    Yerr = Yerr[n > 0]
    Xerr = Xerr[n > 0]
    Ystd = Ystd[n > 0]
    Xstd = Xstd[n > 0]

    return Y, X, Yerr, Xerr, Ystd, Xstd, edges, bins, n
