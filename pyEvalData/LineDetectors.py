# This file is part of the evalData module.
#
# eval Data is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2015 Daniel Schick <schick.daniel@gmail.com>


import numpy as np
import os
from .evalData import spec

class scopeTraces(spec):
    """Inherit from spec and add capabilities to read and plot scope traces.

    Attributes:
        inherited from spec

    """

    # properties
    scopeDataPath = ''

    def __init__(self, name, filePath, specFileExt=''):
        super().__init__(name, filePath, specFileExt)
        # set the path to the scope raw traces
        self.scopeDataPath = self.filePath + '/scope/' + self.specFileName + \
            '_{:04d}/F1_' + self.specFileName + '_{:04d}_{:05d}.txt'

    def readScopeTrace(self, scanNum, scanPoint):
        # add automatic file copying in the future
        # import os#,shutil
        # basePath = 'D:/HZB/Beamtimes/ZPM/2017-02-Schick/scope/'
        # filepath = 'Dy_{:04d}/F1_Dy_{:04d}_{:05d}.txt'.format(ScanNr,ScanNr,ScanPoint)

        #     if not os.path.exists(basePath + filepath):
        #         # copy data if path not exists
        #         shutil.copytree('/mnt/lecroy/2017/2017-02-Schick/Dy_{:04d}'.format(ScanNr),
        #         'D:/HZB/Beamtimes/ZPM/2017-02-Schick/scope/Dy_{:04d}'.format(ScanNr))

        trace = np.genfromtxt(self.scopeDataPath.format(
            scanNum, scanNum, scanPoint), skip_header=5, delimiter=',')
        delays = trace[:, 0]*1e9
        ampl = trace[:, 1]

        return delays, ampl

    def readScopeScan(self, scanNum, delayGrid=[]):

        import evalData

        # check if there is data in the hdf5 file available
        intensities = self.readDataFromHDF5(scanNum, 'ScopeRaw', 'intensities')

        delays, _ = self.readScopeTrace(scanNum, 0)

        if any(intensities):
            # if the data is present in the HDF5 file and we don't want to
            # overwrite, read also the other datasets
            delays = self.readDataFromHDF5(scanNum, 'ScopeRaw', 'delays')
            # print('Scan #{0:.0f} read from HDF5.'.format(scanNum))

        elif any(delays):
            # data is not present in the HDF5 file but there are frames
            # on the disk, so read them and save them

            # print('Scan #{0:.0f} read from files and saved to HDF5.'.format(scanNum))

            folderName = os.path.dirname(
                self.scopeDataPath.format(scanNum, scanNum, 0))

            numFiles = len(os.listdir(folderName))

            if len(delayGrid) > 0:
                delayGrid, _, _, _, _, _, _, _, _ = evalData.binData(
                    delays, delays, delayGrid)
                intensities = np.zeros([numFiles, len(delayGrid)])
            else:
                intensities = np.zeros([numFiles, len(delays)])

            for i in range(numFiles):
                _, ampl = self.readScopeTrace(scanNum, i)

                if len(delayGrid) > 0:
                    ampl, X, Yerr, Xerr, Ystd, Xstd, edges, bins, n = evalData.binData(
                        ampl, delays, delayGrid, statistic='mean')

                intensities[i, :] = ampl

            self.writeData2HDF5(scanNum, 'ScopeRaw',
                                intensities, 'intensities')
            self.writeData2HDF5(scanNum, 'ScopeRaw', delays, 'delays')

        else:
            # no scope data for this scan
            print('Scan #{0:.0f} includes no scope data!'.format(scanNum))
            intensities = []

        scanData = self.plotScans([scanNum], skipPlot=True)

        if len(delayGrid) > 0:
            return scanData, delayGrid, intensities
        else:
            return scanData, delays, intensities
