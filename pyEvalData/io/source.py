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
        overwrite_h5 (bool): overwrite generated h5 file even
          if already existent.

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
        overwrite_h5 (bool): overwrite generated h5 file even
          if already existent.

    """
    def __init__(self, file_name, file_path, **kwargs):
        self.log = logging.getLogger(__name__)
        # self.log.setLevel(config.LOG_LEVEL)
        self.scan_dict = {}
        self.start_scan_number = kwargs.get('start_scan_number', 0)
        self.stop_scan_number = kwargs.get('stop_scan_number', 0)
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
        self.overwrite_h5 = kwargs.get('overwrite_h5', False)

        # update from the source
        self.update()

    def update(self):
        """update

        update the `scan_dict` either from the raw source file/folder
        or from the h5 file.

        """
        self.log.debug('Update source')
        # """update

        # update the scan_list either from the source file/folder by
        # calling parse() or by reading from the h5 file.

        # """
        # # if self.use_h5 and (not self.h5_file_exists or self.overwrite_h5):
        # #     # save the new or changed spec file content to the hdf5 file
        # #     # if it does not exist
        # #     self.spec_file.Save2HDF5(path.join(self.h5_file_path,
        # #                                        self.h5_file_name))
        # # else:
        # self.parse()

    def parse_raw(self):
        """parse_raw

        Parse the raw source file/folder and populate the `scan_dict`.

        """
        raise NotImplementedError('Needs to be implemented!')

    def parse_h5(self):
        """parse_h5

        Parse the h5 file and populate the `scan_dict`.

        """
        print('parsing h5 file')

    def check_h5_file_exists(self):
        """check_h5_file_exists

        Check if the h5 file is present and set `self.h5_file_exists`.

        """
        if path.exists(path.join(self.h5_file_path, self.h5_file_name)):
            self.h5_file_exists = True
        else:
            self.h5_file_exists = False

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
        if self.update_before_read and not dismiss_update:
            self.update()

        try:
            scan = self.scan_dict[scan_number]
        except KeyError:
            raise KeyError('Scan #{:d} not found in scan dict.'.format(scan_number))
        if read_data:
            self.read_scan_data(scan)
        return scan

    def get_scan_data(self, scan_number, dismiss_update=False):
        """get_scan_data

        Returns data from a scan object from the `scan_dict` determined by the scan_number.

        Args:
            scan_number (uint): number of the scan.
            dismiss_update (bool, optional): Dismiss update even if set as
              object attribute. Defaults to `False`.

        Returns:
            scan (Scan): scan object.

        """
        scan = self.get_scan(scan_number, dismiss_update=dismiss_update)
        data = scan.data.copy()
        if self.read_and_forget:
            scan.clear_data()
        return data

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
        if self.update_before_read:
            self.update()

        scans = []
        for scan_number in scan_number_list:
            scan = self.get_scan(scan_number, read_data, dismiss_update=True)

            scans.append(scan)

        return scans

    def read_scan_data(self, scan):
        """read_scan_data

        Reads the data for a given scan object from source.

        Args:
            scan (Scan): scan object.

        """
        raise NotImplementedError('Needs to be implemented!')

    def clear_scan_data(self, scan):
        """clear_scan_data

        Clear the data for a given scan object.

        Args:
            scan (Scan): scan object.

        """
        scan.clear_data()

    def read_all_scan_data(self):
        """read_all_scan_data

        Reads the data for all scan objects in the `scan_dict` from source.

        """
        for scan_number, scan in self.scan_dict.items():
            self.read_scan_data(scan)

    def clear_all_scan_data(self):
        """clear_all_scan_data

        Clears the data for all scan objects in the `scan_dict`.

        """
        for scan_number, scan in self.scan_dict.items():
            self.clear_scan_data(scan)

    def save_scan_to_h5(self, scan, compression=True):
        """clear_all_scan_data

        Clears the data for all scan objects in the `scan_dict`.

        """
        with xu_h5open(path.join(self.h5_file_path,
                                 self.h5_file_name), 'a') as h5:
            groupname = path.splitext(path.splitext(self.file_name)[0])[0]
            print(groupname)
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

    @property
    def h5_file_name(self):
        return self._h5_file_name

    @h5_file_name.setter
    def h5_file_name(self, h5_file_name):
        self._h5_file_name = h5_file_name + self.h5_file_name_postfix + '.h5'
