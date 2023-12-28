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
import re

__all__ = ['edges4grid', 'bin_data', 'traverse_counters', 'resolve_counter_name',
           'col_string_to_eval_string']

__docformat__ = 'restructuredtext'


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
    edges = np.hstack([grid[0]-diff[0]/2,
                       grid[0:-1] + diff/2,
                       grid[-1]+diff[-1]/2])
    binwidth = np.diff(edges)

    return edges, binwidth


def bin_data(y, x, X, statistic='mean'):
    """bin_data

    This is a wrapper around `scipy's binned_statistic
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_.
    In the first step possbile masked elements from the input arrays `x` and
    `y` are removed. The same applies for the new grid array `X` which is also
    sorted and made unique.

    In a second step the edges for the new grid are calculated by
    ``edges4grid`` and used to calculate the new binned values `Y` by using
    ``scipy.stats.binned_statistsic``.

    The type of `statistic` can be chosen. In case of `sum` Poisson statistics
    are applied to calculate the standard derivation of the binned values `Y`.
    Also errors due to the horizontal binning are calculated and returned.
    All return values contain only elements with according non-zero bins.

    Arguments:
        y (ndarray[float]): input y array.
        x (ndarray[float]): input x array.
        X (ndarray[float]): new grid array.
        statistic (str, optional): type of statistics used for scipy's
          ``binned_statistic`` - default is ``mean``.

    Returns:
        (tuple):
        - *Y (ndarray[float])* - binned Y data without zero-bins.
        - *X (ndarray[float])* - new X grid array.
        - *Yerr (ndarray[float])* - Error for Y, according to statistic.
        - *Xerr (ndarray[float])* - Error for Y, according to statistic.
        - *Ystd (ndarray[float])* - Std for Y, according to statistic.
        - *Xstd (ndarray[float])* - Std for X, according to statistic.
        - *edges (ndarray[float])* - Edges of binned data.
        - *bins (ndarray[float])* - Indices of the bins.
        - *n (ndarray[float])* - Number of values per given bin.

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
        # if no binning is applied and no Poisson statistics is applied, all
        # errors and stds are set to zero
        Ystd = np.zeros_like(Y)
        Xstd = np.zeros_like(X)
        Yerr = np.zeros_like(Y)
        Xerr = np.zeros_like(X)
    else:
        # calculate the std of X and Y
        if statistic == 'sum':
            # the std and error are calculated as 1/sqrt(N) for each bin
            Ystd = np.sqrt(Y)
            Yerr = Ystd
        else:
            Ystd, _, _ = binned_statistic(x, y, 'std', edges)
            Yerr = Ystd/np.sqrt(n)

        # calculate the std and error for the horizontal x grid
        Xstd, _, _ = binned_statistic(x, x, 'std', edges)
        Xerr = Xstd/np.sqrt(n)

    # remove zero-bins
    Y = Y[n > 0]
    X = X[n > 0]
    Yerr = Yerr[n > 0]
    Xerr = Xerr[n > 0]
    Ystd = Ystd[n > 0]
    Xstd = Xstd[n > 0]

    return Y, X, Yerr, Xerr, Ystd, Xstd, edges, bins, n


def traverse_counters(clist, cdef, source_cols=''):
    """traverse_counters

    Traverse all counters and replace all predefined counter definitions.
    Returns also a list of the included source counters for error propagation.

    Args:
        clist (list[str]): list of counter names to evaluate.
        cdef (dict{str:str}): dict of predefined counter names and
            definitions.
        source_cols (list[str], optional): counters in the raw source data.

    Returns:
        (tuple):
        - *resolved_counters (list[str])* - resolved counters.
        - *source_counters (list[str])* - all source counters in the resolved counters.

    """
    resolved_counters = []
    source_counters = []

    for counter_name in clist:
        # resolve each counter in the clist
        counter_string, res_source_counters = \
            resolve_counter_name(cdef, counter_name, source_cols)

        resolved_counters.append(counter_string)
        source_counters.extend(res_source_counters)

    return resolved_counters, list(set(source_counters))


def resolve_counter_name(cdef, col_name, source_cols=''):
    """resolve_counter_name

    Replace all predefined counter definitions in a given counter name.
    The function works recursively.

    Args:
        cdef (dict{str:str}): dict of predefined counter names and
            definitions.
        col_name (str): initial counter string.
        source_cols (list[str], optional): columns in the source data.

    Returns:
        (tuple):
        - *col_string (str)* - resolved counter string.
        - *source_counters (list[str])* - source counters in the col_string

    """
    recall = False  # boolean to stop recursive calls
    source_counters = []
    col_string = col_name

    for find_cdef in cdef.keys():
        # check for all predefined counters
        search_pattern = r'\b' + find_cdef + r'\b'
        if re.search(search_pattern, col_string) is not None:
            if cdef[find_cdef] in source_cols:
                # this counter definition is a base source counter
                source_counters.append(cdef[find_cdef])
            # found a predefined counter
            # recursive call if predefined counter must be resolved again
            recall = True
            # replace the counter definition in the string
            (col_string, _) = re.subn(search_pattern,
                                        '(' + cdef[find_cdef] + ')', col_string)

    if recall:
        # do the recursive call
        col_string, rec_source_counters = resolve_counter_name(cdef, col_string, source_cols)
        source_counters.extend(rec_source_counters)

    for find_cdef in source_cols:
        # check for all base source counters
        search_pattern = r'\b' + find_cdef + r'\b'
        if re.search(search_pattern, col_string) is not None:
            source_counters.append(find_cdef)

    return col_string, source_counters


def col_string_to_eval_string(col_string, math_keys, ignore_keys, array_name='source_data'):
    """col_string_to_eval_string

    Use regular expressions in order to generate an evaluateable string
    from the counter string in order to append the new counter to the
    source data.

    Args:
        col_string (str) : Definition of the counter.
         math_keys (list[str]): list of keywords which are evaluated as numpy
            functions.
        ignore_keys (list[str]): list of keywords which should not be
            evaluated.
        array_name (str) : name of the data array. 

    Returns:
        eval_string (str): Evaluateable string to add the new counter
            to the source data.

    """

    # search for alphanumeric counter names in col_string
    iterator = re.finditer(
        '([0-9]*[a-zA-Z\_]+[0-9]*[a-zA-Z]*)*', col_string)
    # these are keys which should not be replaced but evaluated
    keys = list(math_keys).copy()

    for key in iterator:
        # traverse all found counter names
        if len(key.group()) > 0:
            # the match is > 0
            if not key.group() in keys:
                # the counter name is not in the keys list

                # remember this counter name in the key list in order
                # not to replace it again
                keys.append(key.group())
                # the actual replacement
                (col_string, _) = re.subn(r'\b'+key.group()+r'\b',
                                            array_name + '[\'' + key.group() + '\']', col_string)

    # add 'np.' prefix to numpy functions/math keys
    for mk in math_keys:
        if mk not in ignore_keys:
            (col_string, _) = re.subn(r'\b' + mk + r'\b', 'np.' + mk, col_string)
    return col_string
