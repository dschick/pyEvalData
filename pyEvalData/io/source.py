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

import os.path as path
import h5py
from xrayutilities.io.helper import xu_h5open


class Source(object):
    """Source

    Class of default source implementation.

    Args:
        file_name (str): file name including extension,
          can include regex pattern.
        file_path (str): file path.

    Keyword Args:
        start_scan_number (uint): start of scan numbers to parse.
        stop_scan_number (uint): stop of scan numbers to parse.
          This number is included.
        h5_file_name (str): name for generated h5 file.
        h5_file_name_postfix (str): postfix for h5 file name.
        h5_file_path (str): path for generated h5 file.
        read_all_data (bool): read all data on parsing.
          If false, data will be read only on demand.
        read_and_forget (bool): clear data after read to save memory.
        update_before_read (bool): always update from source
          before reading scan data.
        use_h5 (bool): use h5 file to join/compress raw data.
        force_overwrite (bool): forced re-read of raw source and
          re-generated of h5 file.

    Attributes:
        log (logging.logger): logger instance from logging.
        scan_dict (dict(scan)): dict of scan objects with
          key being the scan number.
        start_scan_number (uint): start of scan numbers to parse.
        stop_scan_number (uint): stop of scan numbers to parse.
          This number is included.
        file_name (str): file name including extension,
          can include regex pattern.
        file_path (str): file path.
        h5_file_name (str): name for generated h5 file.
        h5_file_name_postfix (str): postfix for h5 file name.
        h5_file_path (str): path for generated h5 file.
        h5_file_exists(bool): if h5 file exists.
        read_all_data (bool): read all data on parsing.
        read_and_forget (bool): clear data after read to save memory.
        update_before_read (bool): always update from source
          before reading scan data.
        use_h5 (bool): use h5 file to join/compress raw data.
        force_overwrite (bool): forced re-read of raw source and
          re-generated of h5 file.

    """
    def __init__(self, file_name, file_path, **kwargs):
        self.log = logging.getLogger(__name__)
        self.scan_dict = {}
        self._start_scan_number = 0
        self._stop_scan_number = -1
        self.start_scan_number = kwargs.get('start_scan_number', 0)
        self.stop_scan_number = kwargs.get('stop_scan_number', -1)
        self.file_name = file_name
        self.file_path = file_path
        self.h5_file_name_postfix = kwargs.get('h5_file_name_postfix',
                                               '.pyevaldata')
        self.h5_file_name = kwargs.get('h5_file_name', self.file_name)
        self.h5_file_path = kwargs.get('h5_file_path', self.file_path)
        self.check_h5_file_exists()
        self.read_all_data = kwargs.get('read_all_data', False)
        self.read_and_forget = kwargs.get('read_and_forget', False)
        self.update_before_read = kwargs.get('update_before_read', True)
        self.use_h5 = kwargs.get('use_h5', True)
        self.force_overwrite = kwargs.get('force_overwrite', False)

        # update from the source
        self.update()

    def update(self):
        """update

        update the `scan_dict` either from the raw source file/folder
        or from the h5 file.

        """
        self.log.info('Update source')

        if self.use_h5:
            self.log.debug('Updating from h5')
            # do not combine cases for better flow control
            if not self.h5_file_exists:
                self.log.debug('h5 file does not exist')
                self.parse_raw()
                self.save_all_scans_to_h5()
            elif self.update_before_read:
                self.log.debug('Update before read')
                self.parse_raw()
                self.save_all_scans_to_h5()
            elif self.force_overwrite:
                self.log.debug('Force overwrite')
                self.parse_raw()
                self.save_all_scans_to_h5()
            else:
                self.parse_h5()
        else:
            self.log.debug('Updating from raw source')
            self.parse_raw()

    def parse_raw(self):
        """parse_raw

        Parse the raw source file/folder and populate the `scan_dict`.

        """
        raise NotImplementedError('Needs to be implemented!')

    def parse_h5(self):
        """parse_h5

        Parse the h5 file and populate the `scan_dict`.

        """
        self.log.debug('parse_h5')
        print('parsing h5 file')

    def check_h5_file_exists(self):
        """check_h5_file_exists

        Check if the h5 file is present and set `self.h5_file_exists`.

        """
        if path.exists(path.join(self.h5_file_path, self.h5_file_name)):
            self.h5_file_exists = True
        else:
            self.h5_file_exists = False

    def get_last_scan_number(self):
        """get_last_scan_number

        Return the number of the last scan in the `scan_dict`.
        If the `scan_dict` is empty return 0.

        """
        try:
            return sorted(self.scan_dict.keys())[-1]
        except IndexError:
            return 0

    def get_scan(self, scan_number, read_data=True, dismiss_update=False):
        """get_scan

        Returns a scan object from the scan dict determined by the scan_number.

        Args:
            scan_number (uint): number of the scan.
            read_data (bool, optional): read data from source.
              Defaults to `False`.
            dismiss_update (bool, optional): Dismiss update even if set as
              object attribute. Defaults to `False`.

        Returns:
            scan (Scan): scan object.

        """
        self.log.debug('get_scan')

        if self.update_before_read and not dismiss_update:
            self.update()

        try:
            scan = self.scan_dict[scan_number]
        except KeyError:
            raise KeyError('Scan #{:d} not found in scan dict.'.format(scan_number))
        if read_data:
            self.read_scan_data(scan)
        return scan

    def get_scan_list(self, scan_number_list, read_data=True):
        """get_scan_list

        Returns a list of scan object from the `scan_dict` determined by
        the list of scan_number.

        Args:
            scan_number_list (list(uint)): list of numbers of the scan.
            read_data (bool, optional): read data from source.
              Defaults to `False`.

        Returns:
            scans (list(Scan)): list of scan object.

        """
        self.log.debug('get_scan_list')

        if self.update_before_read:
            self.update()

        scans = []
        for scan_number in scan_number_list:
            scan = self.get_scan(scan_number, read_data, dismiss_update=True)

            scans.append(scan)

        return scans

    def get_scan_data(self, scan_number):
        """get_scan_data

        Returns data and meta information from a scan object from the `scan_dict`
        determined by the scan_number.

        Args:
            scan_number (uint): number of the scan.

        Returns:
            data (numpy.recarray[float]): scan data.
            meta (dict()): scan meta information.

        """
        self.log.debug('get_scan_data')

        scan = self.get_scan(scan_number)
        data = scan.data.copy()
        meta = scan.meta.copy()
        if self.read_and_forget:
            scan.clear_data()
        return data, meta

    def get_scan_list_data(self, scan_number_list, dismiss_update=False):
        """get_scan_list_data

        Returns data and meta information for a list of scan objects from
        the `scan_dict` determined by the scan_numbers.

        Args:
            scan_number_list (list(uint)): list of numbers of the scan.
            dismiss_update (bool, optional): Dismiss update even if set as
              object attribute. Defaults to `False`.

        Returns:
            data (list(numpy.recarray[float])): list of scan data.
            meta (list(dict())): list scan meta information.

        """
        self.log.debug('get_scan_list_data')

        data_list = []
        meta_list = []
        for scan in self.get_scan_list(scan_number_list):
            data_list.append(scan.data.copy())
            meta_list.append(scan.meta.copy())
            if self.read_and_forget:
                scan.clear_data()
        return data_list, meta_list

    def read_scan_data(self, scan):
        """read_scan_data

        Reads the data for a given scan object.

        Args:
            scan (Scan): scan object.

        """
        self.log.debug('read_scan_data for scan #{:d}'.format(scan.number))

        last_scan_number = self.get_last_scan_number()

        if (scan.data is None) or \
                (scan.number >= last_scan_number) or self.force_overwrite:
            if self.use_h5:
                self.read_h5_scan_data(scan)
            else:
                self.read_raw_scan_data(scan)
        else:
            self.log.debug('data not updated for scan #{:d}'.format(scan.number))

    def read_raw_scan_data(self, scan):
        """read_raw_scan_data

        Reads the data for a given scan object from raw source.

        Args:
            scan (Scan): scan object.

        """
        raise NotImplementedError('Needs to be implemented!')

    def read_h5_scan_data(self, scan):
        """read_h5_scan_data

        Reads the data for a given scan object from the h5 file.

        Args:
            scan (Scan): scan object.

        """
        self.log.debug('read_h5_scan_data for scan #{:d}'.format(scan.number))

    def clear_scan_data(self, scan):
        """clear_scan_data

        Clear the data for a given scan object.

        Args:
            scan (Scan): scan object.

        """
        self.log.debug('clear_scan_data')

        scan.clear_data()

    def read_all_scan_data(self):
        """read_all_scan_data

        Reads the data for all scan objects in the `scan_dict` from source.

        """
        self.log.debug('read_all_scan_data')

        for scan_number, scan in self.scan_dict.items():
            self.read_scan_data(scan)

    def clear_all_scan_data(self):
        """clear_all_scan_data

        Clears the data for all scan objects in the `scan_dict`.

        """
        self.log.debug('clear_all_scan_data')

        for scan_number, scan in self.scan_dict.items():
            self.clear_scan_data(scan)

    def save_scan_to_h5(self, scan, compression=True):
        """clear_all_scan_data

        Clears the data for all scan objects in the `scan_dict`.

        """
        self.log.debug('save_scan_to_h5 scan #{:d}'.format(scan.number))

        last_scan_number = self.get_last_scan_number()
        # check if the scan must me saved
        # if scan does not exist in file
        # if scan is last one
        # if force_overwrite

        with xu_h5open(path.join(self.h5_file_path,
                                 self.h5_file_name), 'a') as h5:
            groupname = path.splitext(path.splitext(self.file_name)[0])[0]
            try:
                g = h5.create_group(groupname)
            except ValueError:
                g = h5.get(groupname)

            g.attrs['TITLE'] = "Data of SPEC - File %s" % (self.file_name)
            # for s in self.scan_list:
            #     if (((s.name not in g) or s.ischanged) and
            #             s.scan_status != "NODATA"):
            #         s.ReadData()
            #         if s.data is not None:
            #             s.Save2HDF5(h5, group=g, compression=compression)
            #             s.ClearData()
            #             s.ischanged = False
            h5.flush()

    def save_all_scans_to_h5(self):
        """save_all_scans_to_h5

        Saves all scan objects in the `scan_dict` to the h5 file.

        """
        self.log.debug('save_all_scans_to_h5')

        for scan_number, scan in self.scan_dict.items():
            self.save_scan_to_h5(scan)

    @property
    def h5_file_name(self):
        return self._h5_file_name

    @h5_file_name.setter
    def h5_file_name(self, h5_file_name):
        self._h5_file_name = h5_file_name + self.h5_file_name_postfix + '.h5'

    @property
    def start_scan_number(self):
        return self._start_scan_number

    @start_scan_number.setter
    def start_scan_number(self, start_scan_number):
        if start_scan_number < 0:
            self.log.warning('start_scan_number must not be negative!')
            return
        elif (start_scan_number > self.stop_scan_number) and (self.stop_scan_number > -1):
            self.log.warning('start_scan_number must be <= stop_scan_number!')
            return
        else:
            self._start_scan_number = start_scan_number

    @property
    def stop_scan_number(self):
        return self._stop_scan_number

    @stop_scan_number.setter
    def stop_scan_number(self, stop_scan_number):
        if stop_scan_number < -1:
            self.log.warning('stop_scan_number cannot be smaller than -1!')
            return
        elif (stop_scan_number < self.start_scan_number) and (stop_scan_number > -1):
            self.log.warning('stop_scan_number must be >= start_scan_number!')
            return
        else:
            self._stop_scan_number = stop_scan_number
