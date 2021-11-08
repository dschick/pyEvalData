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

import os.path as path


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
        update_before_read (bool): always update from source
          before reading scan data.
        overwrite_h5 (bool): overwrite generated h5 file even
          if already existent.

    Attributes:
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
        update_before_read (bool): always update from source
          before reading scan data.
        overwrite_h5 (bool): overwrite generated h5 file even
          if already existent.

    """

    def __init__(self, file_name, file_path, **kwargs):
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
        self.update_before_read = kwargs.get('update_before_read', True)
        self.overwrite_h5 = kwargs.get('overwrite_h5', False)

        # parse the source
        self.parse()

    def parse(self):
        """parse

        Parse the source file/folder and populate the scan_list.

        """
        raise NotImplementedError('Needs to be implemented!')

    def update(self):
        """update

        update the scan_list either from the source file/folder by
        calling parse() or by reading from the h5 file.

        """
        raise NotImplementedError('Needs to be implemented!')

    def check_h5_file_exists(self):
        """check_h5_file_exists

        Check if the h5 file is present and set `self.h5_file_exists`.

        """
        if path.exists(path.join(self.h5_file_path, self.h5_file_name)):
            self.h5_file_exists = True
        else:
            self.h5_file_exists = False

    def get_scan(self, scan_number, read_data=True):
        """get_scan

        Returns a scan object from the scan list determined by the scan_number.

        Args:
            scan_number (uint): number of the scan.
            read_data (bool, optional): read data from source.
              Defaults to `False`.

        Returns:
            scan (Scan): scan object.

        """
        try:
            scan = self.scan_dict[scan_number]
        except KeyError:
            raise KeyError('Scan #{:d} not found in scan list.'.format(scan_number))

        if read_data:
            self.read_scan_data(scan)

        return scan

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

        Reads the data for all scan objects in the `scan_list` from source.

        """
        for scan_number, scan in self.scan_dict.items():
            self.read_scan_data(scan)

    def clear_all_scan_data(self):
        """clear_all_scan_data

        Clears the data for all scan objects in the `scan_list`.

        """
        for scan_number, scan in self.scan_dict.items():
            self.clear_scan_data(scan)

    @property
    def h5_file_name(self):
        return self._h5_file_name

    @h5_file_name.setter
    def h5_file_name(self, h5_file_name):
        self._h5_file_name = h5_file_name + self.h5_file_name_postfix + '.h5'
