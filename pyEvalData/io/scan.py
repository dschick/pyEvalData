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

from .. import config
import logging

__all__ = ['Scan']

__docformat__ = 'restructuredtext'

import numpy as np


class Scan(object):
    """Scan

    Class for scan which hold the relevant (meta) data.

    Args:
        number (uint): number of the scan.

    Keyword Args:
        cmd (str): scan command.
        user (str): scan user.
        date (str): scan date.
        time (str): scan time.
        int_time (float): integration time.
        init_mopo (dict(float)): initial motor position.
        header (str): full scan header.

    Attributes:
        log (logging.logger): logger instance from logging.
        number (uint): number of the scan.
        meta (dict): meta data dictionary.
        data (ndarray[float]): all data recarray

    """

    def __init__(self, number, **kwargs):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(config.LOG_LEVEL)
        self.log.debug('Creating scan #{:d}'.format(number))
        self.number = np.uint64(number)
        # initialize empty data array and circumvent
        # check for recarray here
        self._data = None
        self.scalar_data_names = []
        self.oned_data_names = []
        self.twod_data_names = []
        self.index_data()
        self.meta = {}
        self.meta['number'] = self.number
        self.meta['cmd'] = kwargs.get('cmd', '')
        self.meta['user'] = kwargs.get('user', '')
        self.meta['date'] = kwargs.get('date', '')
        self.meta['time'] = kwargs.get('time', '')
        self.meta['int_time'] = kwargs.get('int_time', '')
        self.meta['init_mopo'] = kwargs.get('init_mopo', {})
        self.meta['header'] = kwargs.get('header', '')

    def __getattr__(self, attr):
        """__getattr__

        Allows to access the data and meta(init_mopo) keys as scan attributes.

        Returns:
            attr (ndarray[float]|float|str): data/meta values.

        """
        # check data recarray
        try:
            return self.data[attr]
        except (ValueError, IndexError, TypeError):
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
        if self.data is not None:
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
            return None
        else:
            return self.data[self.scalar_data_names]

    def get_oned_data(self):
        """get_oned_data

        Returns only scalar data from the data recarray.

        Returns:
            data (ndarray[float]): 1d data.

        """
        if self.oned_data_names == []:
            return None
        else:
            return self.data[self.oned_data_names]

    def get_twod_data(self):
        """get_twod_data

        Returns only scalar data from the data recarray.

        Returns:
            data (ndarray[float]): 2d data.

        """
        if self.twod_data_names == []:
            return None
        else:
            return self.data[self.twod_data_names]

    def clear_data(self):
        self._data = None
        self.log.debug('Cleared data for scan #{:d}'.format(self.number))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, np.recarray):
            self._data = data
        elif data is None:
            self.log.info('Scan #{:d} contains no data!'.format(self.number))
            self._data = None
        else:
            raise TypeError('data must be numpy recarray')
