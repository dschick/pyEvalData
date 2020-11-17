# This file is part of the evalData module.
#
# eval Data is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2015 Daniel Schick <schick.daniel@gmail.com>

import numpy as np
from scipy.stats import binned_statistic


def edges4grid(grid):
    """Creates a vector of the corresponding edges for a grid vector. """
    binwidth = np.diff(grid)
    edges = np.hstack([grid[0]-binwidth[0]/2, grid[0:-1] +
                       binwidth/2, grid[-1]+binwidth[-1]/2])

    return edges, binwidth


def binData(y, x, X, statistic='mean'):
    """Bin data y(x) on new grid X using a statistic type. """

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