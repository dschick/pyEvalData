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


from numpy.core.records import fromarrays
import os
import h5py

from .source import Source
from .scan import Scan


class PalH5(Source):
    """PalH5

    Source implementation for PalH5 folder/files.

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
        name (str): name of the source
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
    def __init__(self, name, file_name, file_path, **kwargs):
        super().__init__(file_name, file_path, **kwargs)
        self.name = name

    def parse_raw(self):
        """parse_raw

        Parse the PalH5 folder and populate the `scan_dict`.

        """
        self.log.info('parse_raw')

        if not os.path.exists(self.file_path):
            self.log.error('File path does not exist!')
            return

        for root, subdirectories, files in os.walk(self.file_path):
            for sub_dir in sorted(subdirectories):
                # check for scan number in given range
                try:
                    scan_number = int(sub_dir)
                except ValueError:
                    self.log.exception('{:s} is no scan folder - skipping'.format(sub_dir))
                    continue

                if (scan_number >= self.start_scan_number) and \
                        ((scan_number <= self.stop_scan_number) or
                            (self.stop_scan_number == -1)):
                    last_scan_number = self.get_last_scan_number()
                    # check if Scan needs to be re-created
                    # if scan is not present, its the last one, or force overwrite
                    if (scan_number not in self.scan_dict.keys()) or \
                            (scan_number >= last_scan_number) or \
                            self.force_overwrite:
                        # create scan object

                        h5_file = os.path.join(self.file_path,
                                               self.file_name.format(scan_number),
                                               self.file_name.format(scan_number) + '.h5')

                        try:
                            with h5py.File(h5_file, 'r') as h5:
                                header = h5['R{0:04d}/header'.format(scan_number)]

                                init_motor_pos = {}
                                for key in header['motor_init_pos'].keys():
                                    init_motor_pos[key] = \
                                        header['motor_init_pos/{:s}'.format(key)][()]

                                # create scan object
                                try:
                                    # this is a fixQ fix
                                    int_time = float(header['scan_cmd'].asstr()[()].split(' ')[-1])
                                except ValueError:
                                    int_time = float(header['scan_cmd'].asstr()[()].split(' ')[-2])
                                scan = Scan(int(scan_number),
                                            cmd=header['scan_cmd'].asstr()[()],
                                            date=header['time'].asstr()[()].split(' ')[0],
                                            time=header['time'].asstr()[()].split(' ')[1],
                                            int_time=int_time,
                                            header='',
                                            init_mopo=init_motor_pos)
                                self.scan_dict[scan_number] = scan
                            # check if the data needs to be read as well
                            if self.read_all_data:
                                self.read_scan_data(self.scan_dict[scan_number])
                        except OSError:
                            self.log.warning('Could not open file {:s}'.format(h5_file))
                            continue

    def read_raw_scan_data(self, scan):
        """read_raw_scan_data

        Reads the data for a given scan object from Sardana NeXus file.

        Args:
            scan (Scan): scan object.

        """
        self.log.info('read_raw_scan_data for scan #{:d}'.format(scan.number))
        # try to open the file
        h5_file = os.path.join(self.file_path,
                               self.file_name.format(scan.number),
                               self.file_name.format(scan.number) + '.h5')

        with h5py.File(h5_file, 'r') as h5:
            entry = h5['R{0:04d}'.format(scan.number)]
            # iterate through data fields
            data_list = []
            dtype_list = []
            for key in entry['scan_dat'].keys():
                if '_raw' not in key:
                    data_list.append(entry['scan_dat'][key])
                    dtype_list.append((key,
                                      entry['scan_dat'][key].dtype,
                                      entry['scan_dat'][key].shape))
            if len(data_list) > 0:
                scan.data = fromarrays(data_list, dtype=dtype_list)
            else:
                scan.data = None
