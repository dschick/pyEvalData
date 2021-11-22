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
    """edges4grid

    Returns the edges for a given grid vector as well as the
    corresponding width of these bins.

    The ``grid`` is NOT altered - on purpose!
    So even if the ``grid`` is not *unique* there will be bins of width 0.

    Be also aware of the hanling of the first and last bin, as they will
    contain values which will lay outside of the original ``grid``.

    grid       x     x     x     x     x     x     x     x

    edges   |     |     |     |     |     |     |     |     |

    binwidth <---> <---> <---> <---> <---> <---> <---> <--->

    Attributes:
        grid (ndarray[float]): array of grid points.

    Returns:
        (tuple):
        - *edges (ndarray[float])* - array of edges.
        - *binwidth (ndarray[float])* - array of bin widths.

    """
    diff = np.diff(grid)
    edges = np.hstack([grid[0]-diff[0]/2, grid[0:-1] +
                       diff/2, grid[-1]+diff[-1]/2])
    binwidth = np.diff(edges)

    return edges, binwidth


def bin_data(y, x, X, statistic='mean'):
    """bin_data

    Bin data y(x) on new grid X using a statistic type.

    """

    # get only unmasked data
    idx = ~np.ma.getmask(x)
    idy = ~np.ma.getmask(y)
    y = y[idx & idy].flatten('F')
    x = x[idx & idy].flatten('F')
    idX = ~np.ma.getmask(X)
    X = np.unique(np.sort(X[idX].flatten('F')))
    # set non-finite values to 0
    y[~np.isfinite(y)] = 0
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
