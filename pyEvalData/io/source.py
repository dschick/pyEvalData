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

from .scan import Scan

import os.path as path
from numpy.core.records import fromarrays
import nexusformat.nexus as nxs


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
        nexus_file_name (str): name for generated nexus file.
        nexus_file_name_postfix (str): postfix for nexus file name.
        nexus_file_path (str): path for generated nexus file.
        read_all_data (bool): read all data on parsing.
          If false, data will be read only on demand.
        read_and_forget (bool): clear data after read to save memory.
        update_before_read (bool): always update from source
          before reading scan data.
        use_nexus (bool): use nexus file to join/compress raw data.
        force_overwrite (bool): forced re-read of raw source and
          re-generated of nexus file.

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
        nexus_file_name (str): name for generated nexus file.
        nexus_file_name_postfix (str): postfix for nexus file name.
        nexus_file_path (str): path for generated nexus file.
        nexus_file_exists(bool): if nexus file exists.
        read_all_data (bool): read all data on parsing.
        read_and_forget (bool): clear data after read to save memory.
        update_before_read (bool): always update from source
          before reading scan data.
        use_nexus (bool): use nexus file to join/compress raw data.
        force_overwrite (bool): forced re-read of raw source and
          re-generated of nexus file.

    """
    def __init__(self, file_name, file_path, **kwargs):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(config.LOG_LEVEL)
        self.name = file_name
        self.scan_dict = {}
        self._start_scan_number = 0
        self._stop_scan_number = -1
        self.start_scan_number = kwargs.get('start_scan_number', 0)
        self.stop_scan_number = kwargs.get('stop_scan_number', -1)
        self.file_name = file_name
        self.file_path = file_path
        self.nexus_file_name_postfix = kwargs.get('nexus_file_name_postfix',
                                                  '.pyevaldata')
        self.nexus_file_name = kwargs.get('nexus_file_name', self.file_name)
        self.nexus_file_path = kwargs.get('nexus_file_path', self.file_path)
        self.check_nexus_file_exists()
        self.read_all_data = kwargs.get('read_all_data', False)
        self.read_and_forget = kwargs.get('read_and_forget', False)
        self.update_before_read = kwargs.get('update_before_read', False)
        self.use_nexus = kwargs.get('use_nexus', True)
        self.force_overwrite = kwargs.get('force_overwrite', False)

        # update from the source
        self.update()

    def __getattr__(self, attr):
        """__getattr__

        Allows to access scans as source attributes.

        Returns:
            scan (Scan): scan object.

        """
        if attr.startswith("scan"):
            index = attr[4:]

            try:
                scan_number = int(index)
            except ValueError:
                raise ValueError('Scan number must be convertable to an integer!')

            return self.get_scan(scan_number)
        else:
            raise AttributeError('\'{:s}\' has no attribute \'{:s}\''.format(__name__, attr))

    def __len__(self):
        """Returns length of ``scan_dict``"""
        return self.scan_dict.__len__()

    def update(self, scan_number_list=[]):
        """update

        update the ``scan_dict`` either from the raw source file/folder
        or from the nexus file.
        The optional ``scan_number_list`` runs the update only if required
        for the included scan.

        Attributes:
            scan_number_list (list[int]): explicit list of scans

        """

        if ~isinstance(scan_number_list, list):
            scan_number_list = list(scan_number_list)

        last_scan_number = self.get_last_scan_number()
        if (len(scan_number_list) == 0) \
                or (last_scan_number in scan_number_list) \
                or any(list(set(scan_number_list) - set(self.scan_dict.keys()))):

            self.log.info('Update source')

            if self.use_nexus:
                self.log.debug('Updating from nexus')
                # do not combine cases for better flow control
                if not self.nexus_file_exists:
                    self.log.debug('nexus file does not exist')
                    self.parse_raw()
                    self.save_all_scans_to_nexus()
                elif self.update_before_read:
                    self.log.debug('Update before read')
                    self.parse_raw()
                    self.save_all_scans_to_nexus()
                elif self.force_overwrite:
                    self.log.debug('Force overwrite')
                    self.parse_raw()
                    self.save_all_scans_to_nexus()
                else:
                    self.parse_nexus()
            else:
                self.log.debug('Updating from raw source')
                self.parse_raw()
        else:
            self.log.debug('Skipping update for scans {:s} '
                           'which are already present in '
                           'scan_dict.'.format(str(scan_number_list)))

    def parse_raw(self):
        """parse_raw

        Parse the raw source file/folder and populate the `scan_dict`.

        """
        raise NotImplementedError('Needs to be implemented!')

    def parse_nexus(self):
        """parse_nexus

        Parse the nexus file and populate the `scan_dict`.

        """
        self.log.info('parse_nexus')
        nxs_file_path = path.join(self.nexus_file_path, self.nexus_file_name)
        try:
            nxs_file = nxs.nxload(nxs_file_path, mode='r')
        except nxs.NeXusError:
            raise nxs.NeXusError('NeXus file \'{:s}\' does not exist!'.format(nxs_file_path))

        with nxs_file.nxfile:
            for entry in nxs_file:
                # check for scan number in given range
                if (nxs_file[entry].number >= self.start_scan_number) and \
                        ((nxs_file[entry].number <= self.stop_scan_number) or
                            (self.stop_scan_number == -1)):
                    last_scan_number = self.get_last_scan_number()
                    # check if Scan needs to be re-created
                    # if scan is not present, its the last one, or force overwrite
                    if (nxs_file[entry].number not in self.scan_dict.keys()) or \
                            (nxs_file[entry].number >= last_scan_number) or \
                            self.force_overwrite:
                        # create scan object
                        init_mopo = {}
                        for field in nxs_file[entry].init_mopo:
                            init_mopo[field] = nxs_file[entry]['init_mopo'][field]

                        scan = Scan(int(nxs_file[entry].number),
                                    cmd=nxs_file[entry].cmd,
                                    date=nxs_file[entry].date,
                                    time=nxs_file[entry].time,
                                    int_time=float(nxs_file[entry].int_time),
                                    header=nxs_file[entry].header,
                                    init_mopo=init_mopo)
                        self.scan_dict[nxs_file[entry].number] = scan
                        # check if the data needs to be read as well
                        if self.read_all_data:
                            self.read_scan_data(self.scan_dict[nxs_file[entry].number])

    def check_nexus_file_exists(self):
        """check_nexus_file_exists

        Check if the nexus file is present and set `self.nexus_file_exists`.

        """
        if path.exists(path.join(self.nexus_file_path, self.nexus_file_name)):
            self.nexus_file_exists = True
        else:
            self.nexus_file_exists = False

    def get_last_scan_number(self):
        """get_last_scan_number

        Return the number of the last scan in the `scan_dict`.
        If the `scan_dict` is empty return 0.

        """
        try:
            return sorted(self.scan_dict.keys())[-1]
        except IndexError:
            return 0

    def get_all_scan_numbers(self):
        """get_all_scan_numbers

        Return the all scan number from the `scan_dict`.

        """
        try:
            return sorted(self.scan_dict.keys())
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
            self.update(scan_number)

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
            self.update(scan_number_list)

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
        if scan.data is not None:
            data = scan.data.copy()
        else:
            data = None
        meta = scan.meta.copy()
        if self.read_and_forget:
            scan.clear_data()
        return data, meta

    def get_scan_list_data(self, scan_number_list):
        """get_scan_list_data

        Returns data and meta information for a list of scan objects from
        the `scan_dict` determined by the scan_numbers.

        Args:
            scan_number_list (list(uint)): list of numbers of the scan.

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
            if self.use_nexus:
                self.read_nexus_scan_data(scan)
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

    def read_nexus_scan_data(self, scan):
        """read_nexus_scan_data

        Reads the data for a given scan object from the nexus file.

        Args:
            scan (Scan): scan object.

        """
        self.log.debug('read_nexus_scan_data for scan #{:d}'.format(scan.number))
        # try to open the file
        nxs_file_path = path.join(self.nexus_file_path, self.nexus_file_name)
        try:
            nxs_file = nxs.nxload(nxs_file_path, mode='r')
        except nxs.NeXusError:
            raise nxs.NeXusError('NeXus file \'{:s}\' does not exist!'.format(nxs_file_path))

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
        for field in entry.data:
            data_list.append(entry.data[field])
            dtype_list.append((field, entry.data[field].dtype, entry.data[field].shape))
        if len(data_list) > 0:
            scan.data = fromarrays(data_list, dtype=dtype_list)
        else:
            scan.data = None

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

    def save_scan_to_nexus(self, scan, nxs_file=''):
        """save_scan_to_nexus

        Saves a scan to the nexus file.

        """
        if nxs_file == '':
            nxs_file = self.get_nexus_file()

        entry_name = 'entry{:d}'.format(scan.number)

        # evaluate if we need to forget the data again
        if scan.data is None:
            clear_data = True
        else:
            clear_data = False
        # read the raw data
        self.read_raw_scan_data(scan)

        self.log.info('save_scan_to_nexus for scan #{:d}'.format(scan.number))
        with nxs_file.nxfile:
            # if the entry already exists, it must be deleted in advance
            try:
                del nxs_file[entry_name]
            except nxs.NeXusError:
                pass
            # (re-)create entry
            entry = nxs_file[entry_name] = nxs.NXentry()
            # iterate meta information
            for key, value in scan.meta.items():
                if key == 'init_mopo':
                    # create dedicated collection for initial motor positions
                    entry['init_mopo'] = nxs.NXcollection()
                    # iterate through initial motor positions
                    for mopo_key, mopo_value in scan.meta['init_mopo'].items():
                        entry.init_mopo[mopo_key] = nxs.NXfield(mopo_value)
                else:
                    # add meta information as attribute to entry
                    entry.attrs[key] = value
            # create dedicated collection for data
            entry['data'] = nxs.NXcollection()
            # check if there is any data present at all
            if scan.data is not None:
                # iterate data
                for col in scan.data.dtype.names:
                    entry.data[col] = nxs.NXfield(scan.data[col])
                # clear data of the scan if it was not present before
                # or read and forget
                if clear_data or self.read_and_forget:
                    scan.clear_data()

    def save_all_scans_to_nexus(self):
        """save_all_scans_to_nexus

        Saves all scan objects in the `scan_dict` to the nexus file.

        """
        self.log.info('save_all_scans_to_nexus')
        nxs_file = self.get_nexus_file()
        try:
            last_scan_in_nexus = sorted(int(num.strip('entry')) for num in nxs_file.keys())[-1]
        except IndexError:
            last_scan_in_nexus = -1

        for scan_number, scan in self.scan_dict.items():
            entry_name = 'entry{:d}'.format(scan.number)
            try:
                _ = nxs_file[entry_name]
                scan_in_nexus = True
            except KeyError:
                scan_in_nexus = False

            if (not scan_in_nexus) or (scan.number >= last_scan_in_nexus) \
                    or self.force_overwrite:
                self.save_scan_to_nexus(scan, nxs_file)

    def get_nexus_file(self, mode='rw'):
        """get_nexus_file

        Return the file handle to the NeXus file in a given ``mode```.

        Args:
            mode (str, optional): file mode. defaults to 'rw'.

        Returns:
            nxs_file (NXFile): file handle to NeXus file.

        """
        self.log.debug('get_nexus_file')
        try:
            nxs_file = nxs.nxload(path.join(self.nexus_file_path, self.nexus_file_name), mode='rw')
        except nxs.NeXusError:
            nxs.NXroot().save(path.join(self.nexus_file_path, self.nexus_file_name))
            nxs_file = nxs.nxload(path.join(self.nexus_file_path, self.nexus_file_name), mode='rw')
        return nxs_file

    @property
    def nexus_file_name(self):
        return self._nexus_file_name

    @nexus_file_name.setter
    def nexus_file_name(self, nexus_file_name):
        self._nexus_file_name = nexus_file_name + self.nexus_file_name_postfix + '.nxs'

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
