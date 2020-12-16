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

import os
import xrayutilities as xu
import numpy as np
from numpy.lib.recfunctions import append_fields


class Spec(object):
    """Spec"""

    def __init__(self, name, file_path, spec_file_ext='', h5_path=''):
        """Initialize the class, set all file names and load the spec file.

        Args:
            name (str)                  : Name of the spec file.
            file_path (str)              : Base path of the spec and HDF5 files.
            spec_file_ext (Optional[str]) : File extension of the spec file,
                                          default is none.

        """
        self.name = name
        self.spec_file_name = self.name + spec_file_ext
        self.h5_file_name = self.name + '_pyEvalData.h5'
        self.file_path = file_path
        self.h5_path = h5_path
        self.spec_file = ''
        self.update_before_read = False
        self.overwrite_h5 = False
        self.motor_names = []
        # load the spec data
        self.update_spec()

    def load_spec(self):
        """Load the spec data either from the hdf5 or from the spec file."""
        # check if the hdf5 file exists
        if not os.path.exists(self.h5_path + self.h5_file_name):
            # no hdf5 file found --> read the spec file
            self.update_spec()

    def update_spec(self):
        """Update the current spec file if already in memory.
        Otherwise read it and write its content to the hdf5 file.

        """
        try:
            # try if spec file object already exist
            self.spec_file.Update()
        except:
            # load the spec file from disc
            self.spec_file = xu.io.SPECFile(self.spec_file_name, path=self.file_path)
            self.spec_file.Update()

        if (not os.path.exists(os.path.join(self.h5_path, self.h5_file_name))
                or self.overwrite_h5):
            # save the new or changed spec file content to the hdf5 file
            # if it does not exist
            print()
            self.spec_file.Save2HDF5(os.path.join(
                self.h5_path, self.h5_file_name))

        self.motor_names = self.spec_file.init_motor_names

    def write_data_to_hdf5(self, scan_num, child_name, data, data_name):
        """Write data for a given scan number to the HDF5 file.

        Args:
            scan_num (int)   : Scan number of the spec scan.
            child_name (str) : Name of the child where to save the data to.
            data (ndarray)  : Data array
            data_name (str)  : Name of the dataset.

        """

        # open the HDF5 file
        with xu.io.helper.xu_h5open(os.path.join(self.h5_path,
                                                 self.h5_file_name), mode='a') as h5:

            h5g = h5.get(list(h5.keys())[0])  # get the root
            scan = h5g.get("scan_{:d}".format(scan_num))  # get the current scan
            try:
                # try to create the new subgroup for the area detector data
                scan.create_group(child_name)
            except Exception as e:
                print(e)
                np.void

            g5 = scan[child_name]  # this is the new group

            try:
                # add the data to the group
                g5.create_dataset(data_name, data=data,
                                  compression="gzip", compression_opts=9)
            except Exception as e:
                print(e)
                np.void

            h5.flush()  # write the data to the file

    def read_data_from_hdf5(self, scan_num, child_name, data_name):
        """Read data for a given scan number from the HDF5 file.

        Args:
            scan_num (int)   : Scan number of the spec scan.
            child_name (str) : Name of the child where to save the data to.
            data_name (str)  : Name of the dataset.

        Returns:
            data (ndarray): Data array from the spec scan.

        """

        # open the HDF5 file
        with xu.io.helper.xu_h5open(os.path.join(self.h5_path,
                                                 self.h5_file_name), mode='a') as h5:
            h5g = h5.get(list(h5.keys())[0])  # get the root

            try:
                scan = h5g.get("scan_{:d}".format(scan_num))  # get the current scan
                # access the child if a childName is given
                if len(child_name) == 0:
                    g5 = scan
                else:
                    g5 = scan[child_name]

                data = g5[data_name][:]  # get the actual dataset
            except Exception as e:
                print(e)
                # if no data is available return False
                data = False

        return data

    def get_scan_data(self, scan_num):
        """Read the spec data for a given scan number from the hdf5 file.

        Args:
            scan_num (int) : Number of the spec scan.

        Returns:
            motors (ndarray): Motors of the spec scan.
            data   (ndarray): Counters of the spec scan.

        """

        # if set, update the spec and hdf5 files before reading the scan data
        if self.update_before_read:
            self.update_spec()

        # read the scan from the hdf5 file
        try:
            self.motor_names = self.spec_file.init_motor_names
            # if no motor_names are given motors are set as empty array
            # read the data providing the motor_names
            motors, data = xu.io.geth5_scan(os.path.join(self.h5_path, self.h5_file_name),
                                            scan_num, *self.motor_names, rettype='numpy')

            # convert the data array to float64 since lmfit works better
            # is there a smarter way to do so?
            dt = data.dtype
            dt = dt.descr
            for i, thisType in enumerate(dt):
                dt[i] = (dt[i][0], 'float64')
            dt = np.dtype(dt)
            data = data.astype(dt)
            data = np.rec.array(data, names=data.dtype.names)

            for name in list(set(list(motors.dtype.names)) - set(list(data.dtype.names))):
                data = append_fields(data, name, data=motors[name],
                                     dtypes=float, asrecarray=True, usemask=False)
        except Exception as e:
            print(e)
            print('Scan #{0:.0f} not present in hdf5 file!'.format(scan_num))
            data = []

        return data
