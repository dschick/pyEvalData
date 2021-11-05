#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
# Copyright (c) 2015-2021 Daniel Schick
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

__all__ = ['Scan']

__docformat__ = 'restructuredtext'

import numpy as np


class Scan(object):
    """Scan

    Class for scan which hold the relevant (meta) data.

    Args:
        number (uint): number of the scan.
        meta (dict): meta data dictionary.
        data (ndarray[float]): all data recarray

    Keyword Args:

    Attributes:
        number (uint): number of the scan.
        meta (dict): meta data dictionary.
        data (ndarray[float]): all data recarray

    """

    def __init__(self, number, data, **kwargs):
        self.number = np.uint64(number)

        if isinstance(data, np.recarray):
            self.data = data
        else:
            raise TypeError('data must be numpy recarray')

        self.scalar_data_names = []
        self.oned_data_names = []
        self.twod_data_names = []
        self.index_data()
        self.meta = {}
        self.meta['cmd'] = kwargs.get('cmd', '')
        self.meta['user'] = kwargs.get('cmd', '')
        self.meta['date'] = kwargs.get('cmd', '')
        self.meta['time'] = kwargs.get('cmd', '')
        self.meta['init_mopo'] = kwargs.get('init_mopo', {})
        self.meta['header'] = kwargs.get('header', '')

    def __getattr__(self, attr):
        """__getattr__

        return scanX objects where X stands for the scan number in the SPECFile
        which for this purpose is assumed to be unique. (otherwise the first
        instance of scan number X is returned)

        Returns:


        """
        # check data recarray
        try:
            return self.data[attr]
        except ValueError:
            pass

        # check meta dict
        try:
            return self.meta[attr]
        except KeyError:
            pass

        # check meta init_mopo dict
        try:
            return self.meta['init_mopo'][attr]
        except KeyError:
            raise AttributeError('Scan has no attribute {:s}'.format(attr))

    def index_data(self):
        """index_data

        Check the dimensions of the data recarray elements and
        remember the names for scaler, 1d, and 2d data columns.

        """
        for descr in self.data.dtype.descr:
            try:
                if len(descr[2]) == 1:
                    self.oned_data_names.append(descr[0])
                elif len(descr[2]) == 2:
                    self.twod_data_names.append(descr[0])
            except Exception:
                self.scalar_data_names.append(descr[0])

    def get_scalar_data(self):
        """get_scalar_data

        Returns only scalar data from the data recarray.

        Returns:
            data (ndarray[float]): scalar data.

        """
        if self.scalar_data_names == []:
            return np.array([])
        else:
            return self.data[self.scalar_data_names]

    def get_oned_data(self):
        """get_oned_data

        Returns only scalar data from the data recarray.

        Returns:
            data (ndarray[float]): 1d data.

        """
        if self.oned_data_names == []:
            return np.array([])
        else:
            return self.data[self.oned_data_names]

    def get_twod_data(self):
        """get_twod_data

        Returns only scalar data from the data recarray.

        Returns:
            data (ndarray[float]): 2d data.

        """
        if self.twod_data_names == []:
            return np.array([])
        else:
            return self.data[self.twod_data_names]