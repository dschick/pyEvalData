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
import nexusformat.nexus as nxs
import os.path as path

from .source import Source
from .scan import Scan


class SardanaNeXus(Source):
    """SardanaNeXus

    Source implementation for Sardana NeXus files.

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

        Parse the Sardana NeXus file and populate the `scan_dict`.

        """
        self.log.info('parse_raw')
        nxs_file_path = path.join(self.file_path, self.file_name)
        try:
            nxs_file = nxs.nxload(nxs_file_path, mode='r')
        except nxs.NeXusError:
            raise nxs.NeXusError('Sardana NeXus file \'{:s}\' does not exist!'.format(
                nxs_file_path))

        with nxs_file.nxfile:
            for entry in nxs_file:
                # check for scan number in given range
                entry_number = int(nxs_file[entry].entry_identifier)
                if (entry_number >= self.start_scan_number) and \
                        ((entry_number <= self.stop_scan_number) or
                            (self.stop_scan_number == -1)):
                    last_scan_number = self.get_last_scan_number()
                    # check if Scan needs to be re-created
                    # if scan is not present, its the last one, or force overwrite
                    if (entry_number not in self.scan_dict.keys()) or \
                            (entry_number >= last_scan_number) or \
                            self.force_overwrite:
                        # create scan object
                        init_mopo = {}
                        for field in nxs_file[entry].measurement.pre_scan_snapshot:
                            init_mopo[field] = \
                                nxs_file[entry]['measurement/pre_scan_snapshot'][field]

                        scan = Scan(int(entry_number),
                                    cmd=nxs_file[entry].title,
                                    date=nxs_file[entry].start_time,
                                    time=nxs_file[entry].start_time,
                                    int_time=float(0),
                                    header='',
                                    init_mopo=init_mopo)
                        self.scan_dict[entry_number] = scan
                        # check if the data needs to be read as well
                        if self.read_all_data:
                            self.read_scan_data(self.scan_dict[entry_number])

    def read_raw_scan_data(self, scan):
        """read_raw_scan_data

        Reads the data for a given scan object from Sardana NeXus file.

        Args:
            scan (Scan): scan object.

        """
        self.log.info('read_raw_scan_data for scan #{:d}'.format(scan.number))
        # try to open the file
        nxs_file_path = path.join(self.file_path, self.file_name)
        try:
            nxs_file = nxs.nxload(nxs_file_path, mode='r')
        except nxs.NeXusError:
            raise nxs.NeXusError('Sardana NeXus file \'{:s}\' does not exist!'.format(
                nxs_file_path))
        entry_name = 'entry{:d}'.format(scan.number)
        # try to enter entry
        try:
            entry = nxs_file[entry_name]
        except nxs.NeXusError:
            self.log.exception('Entry #{:d} not present in NeXus file!'.format(scan.number))
            return
        # iterate through data fields
        data_list = []
        dtype_list = []
        for field in entry.measurement:
            # do not add data which is already in the pre-scan snapshot
            # that is tricky if it is in the snapshot and scanned ...
            if field != 'pre_scan_snapshot':
                data_list.append(entry.measurement[field])
                dtype_list.append((field,
                                   entry.measurement[field].dtype,
                                   entry.measurement[field].shape))
        if len(data_list) > 0:
            scan.data = fromarrays(data_list, dtype=dtype_list)
        else:
            scan.date = None
