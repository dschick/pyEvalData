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


import xrayutilities as xu

from .source import Source
from .scan import Scan


class Spec(Source):
    """Spec

    Source implementation for SPEC files.

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
    def __init__(self, file_name, file_path, **kwargs):
        super().__init__(file_name, file_path, **kwargs)

    def parse_raw(self):
        """parse_raw

        Parse the raw source file/folder and populate the `scan_dict`.

        """
        self.log.info('parse_raw')

        if ('spec_file' not in dir(self)) or self.force_overwrite:
            self.log.info('Create spec_file from xrayutilities')
            self.spec_file = xu.io.SPECFile(self.file_name,
                                            path=self.file_path)

        # update the xu.spec_file
        self.spec_file.Update()
        # iterate through scan list in xu.spec_file
        for spec_scan in self.spec_file.scan_list:
            # check for scan number in given range
            if (spec_scan.nr >= self.start_scan_number) and \
                    ((spec_scan.nr <= self.stop_scan_number) or
                        (self.stop_scan_number == -1)):
                last_scan_number = self.get_last_scan_number()
                # check if Scan needs to be re-created
                # if scan is not present, its the last one, or force overwrite
                if (spec_scan.nr not in self.scan_dict.keys()) or \
                        (spec_scan.nr >= last_scan_number) or \
                        self.force_overwrite:
                    # rename init_motor_pos keys without prefix
                    init_motor_pos = {}
                    for key, value in spec_scan.init_motor_pos.items():
                        init_motor_pos[key.replace('INIT_MOPO_', '')] = value
                    # catching PR for itime in xu SpecScan missing
                    try:
                        int_time = float(spec_scan.itime)
                    except AttributeError:
                        int_time = 0.0
                    # create scan object
                    scan = Scan(int(spec_scan.nr),
                                cmd=spec_scan.command,
                                date=spec_scan.date,
                                time=spec_scan.time,
                                int_time=int_time,
                                header=spec_scan.header,
                                init_mopo=init_motor_pos)
                    self.scan_dict[spec_scan.nr] = scan
                    # check if the data needs to be read as well
                    if self.read_all_data:
                        self.read_scan_data(self.scan_dict[spec_scan.nr])

    def read_raw_scan_data(self, scan):
        """read_raw_scan_data

        Reads the data for a given scan object from raw source.

        Args:
            scan (Scan): scan object.

        """
        self.log.info('read_raw_scan_data for scan #{:d}'.format(scan.number))

        spec_scan = self.spec_file.__getattr__('scan{:d}'.format(scan.number))
        spec_scan.ReadData()
        scan.data = spec_scan.data
        spec_scan.ClearData()
        scan.meta['header'] = spec_scan.header
