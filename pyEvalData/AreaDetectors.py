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
import numpy.lib.recfunctions as recfuncs
from xrayutilities.io.imagereader import ImageReader
from .evalData import spec


class Pilatus100k(ImageReader):

    """
    parse Dectris Pilatus 100k frames (*.tiff) to numpy arrays
    Ignore the header since it seems to contain no useful data
    """

    def __init__(self, **keyargs):
        """
        initialize the Piulatus100k reader, which includes setting the dimension of
        the images as well as defining the data used for flat- and darkfield
        correction!

        Parameter
        ---------
         optional keywords arguments keyargs:
          flatfield: filename or data for flatfield correction. supported file
                     types include (*.bin *.bin.xz and *.npy files). otherwise
                     a 2D numpy array should be given
          darkfield: filename or data for darkfield correction. same types as
                     for flat field are supported.
        """

        ImageReader.__init__(self, 195, 487, hdrlen=4096, dtype=np.int32,
                             byte_swap=False, **keyargs)


###########################

class areaDetector(spec):
    """Inherit from spec and add capabilities to read and evaluate area
    detector frames from its specific goniometer setup to reciprocal space.

    Attributes:
        hxrd (HXRD[xrayutilities]) : Instance of the HXRD class of the
                                     xrayutilities.
        gridder 
        (gidder[xrayutilities])    : Instance of the gridder class of the
                                     xrayutilities.
        normalizer 
        (IntensityNormalizer[xrayutilities])
                                   : Instance of the IntensityNormalizerr class
                                     of the xrayutilities.
        UB (ndarray[float])        : Transformation matrix UB for
                                     q-hkl tranformation in xrayutilities
        delta (List[float])        : Offset angles of the goniometer axis:
                                     Theta, Two_Theta
                                     default is [0,0].
        motorNames (List[str])     : List of goniometer motor names - default
                                     is ['Theta', 'Two_Theta']
        customCounters (List[str]) : List of custom counters - default is
                                     ['qx', 'qy', 'qz', 'QxMap', 'QyMap', 'QzMap']
        plotLog (bool)             : Boolean if subplots of RSM are log or lin

    """

    # properties
    rawDataPath = ''
    hxrd = ''
    gridder = ''
    normalizer = ''
    UB = ''
    delta = [0, 0]
    motorNames = ['Theta', 'TwoTheta']
    customCounters = ['qx', 'qy', 'qz', 'QxMap', 'QyMap',
                      'QzMap', 'Hs', 'Ks', 'Ls', 'HMap', 'KMap', 'LMap']
    plotLog = True

    def __init__(self, name, filePath, specFileExt=''):
        super().__init__(name, filePath, specFileExt)

    def addCustomCounters(self, specData, scanNum, baseCounters):
        """Add custom counters to the spec data array.
        Here we add the Qx, Qy, Qz maps and axises which by default have a
        different length than the spec data array. In this case all default
        spec counters are removed and only custom counters are given.

        Args:
            specData (ndarray)     : Data array from the spec scan.
            scanNum (int)          : Scan number of the spec scan.
            baseCounters list(str) : List of the base spec and custom counters
                                     from the cList and xCol.

        Returns:
            specData (ndarray): Updated data array from the spec scan.

        """

        # check if any custom counter is in the baseCounters list
        usedCustomCounters = set(baseCounters) & set(self.customCounters)
        if usedCustomCounters:

            if usedCustomCounters & set(['qx', 'qy', 'qz', 'QxMap', 'QyMap', 'QzMap']):
                # calculate the Q data for the current scan number
                Qmap, qx, qy, qz = self.convAreaScan2Q(scanNum)

                # do the integration along the different axises
                QxMap = np.trapz(np.trapz(Qmap, qy, axis=1), qz, axis=1)
                QyMap = np.trapz(np.trapz(Qmap, qx, axis=0), qz, axis=1)
                QzMap = np.trapz(np.trapz(Qmap, qx, axis=0), qy, axis=0)

            if usedCustomCounters & set(['Hs', 'Ks', 'Ls', 'HMap', 'KMap', 'LMap']):
                # calculate the HKL data for the current scan number
                HKLmap, Hs, Ks, Ls = self.convAreaScan2HKL(scanNum)

                # do the integration along the different axises
                HMap = np.trapz(np.trapz(HKLmap, Ks, axis=1), Ls, axis=1)
                KMap = np.trapz(np.trapz(HKLmap, Hs, axis=0), Ls, axis=1)
                LMap = np.trapz(np.trapz(HKLmap, Hs, axis=0), Ks, axis=0)

            sizeValid = True

            for customCounter in usedCustomCounters:

                if len(eval(customCounter)) != len(specData):
                    # the length of the custom counters is different from the
                    # spec data array, so we cannot append and need to init an
                    # empty spec data array
                    sizeValid = False

            if not sizeValid:
                specData = np.array([])
                print('Custom counter has a different length than the spec scan!')
                print('Cannot use default spec counters anymore!')

            for customCounter in usedCustomCounters:
                if len(specData) == 0:
                    specData = np.array(eval(customCounter), dtype=[
                                        (customCounter, float)])
                else:
                    try:
                        # in case the custom counter has the same name as a
                        # default spec base counter
                        specData[customCounter] = eval(customCounter)
                    except Exception as e:
                        if len(eval(customCounter)) == len(specData):
                            # append the custom counters to data array
                            specData = recfuncs.append_fields(specData, customCounter, data=eval(
                                customCounter), dtypes=float, asrecarray=True, usemask=False)
                        else:
                            print(
                                'Adding a custom counter with a different length does not work!')

        return specData

    def readRawScan(self, scanNum):
        """Read the raw data of an area detector scan including.

        """
        # stub
        return

    def writeAllAreaScans2HDF5(self):
        """Use this function with caution. It might take some time.
        Reads all scans from the spec file and save the area detector frames,
        if present, to the HDF5 file.
        Currently it allways overwrite the whole hdf5 file

        """

        # update the spec file in order to have the specFile object at hand
        self.updateSpec()

        for i, scan in enumerate(self.specFile.scan_list):
            # iterate over all scan in the specFile
            self.readAreaScan(i+1)  # read (and write) the pilatus data

    def readAreaScan(self, scanNum):
        """Read the complete data of an area detector scan including the frames,
        motors, and spec data.

        Args:
            scanNum (int)   : Scan number of the spec scan.

        Returns:
            frames (ndarray): Data array from the area detector data.
            motors (ndarray): Data array from the spec motors.
            data (ndarray)  : Data array from the spec scan.

        """

        # update the spec file
        if self.updateBeforeRead:
            self.updateSpec()

        # check if frames are already stored in hdf5 file
        frames = self.readDataFromHDF5(scanNum, 'AreaRaw', 'frames')

        if any(frames) and not self.overwriteHDF5:
            # if the data is present in the HDF5 file and we don't want to
            # overwrite, read also the other datasets
            motors = self.readDataFromHDF5(scanNum, 'AreaRaw', 'motors')
            data = self.readDataFromHDF5(scanNum, '', 'data')
            # print('Scan #{0:.0f} read from HDF5.'.format(scanNum))
        elif os.path.isfile(self.rawDataPath.format(scanNum, 1)):
            # data is not present in the HDF5 file but there are frames
            # on the disk, so read them and save them

            # print('Scan #{0:.0f} read from files and saved to HDF5.'.format(scanNum))

            # get the motors and data from the spec scan
            motors, data = self.getScanData(scanNum)

            frames = self.readRawScan(scanNum)

            # write the frames and motors to the HDF5 file
            self.writeData2HDF5(scanNum, 'AreaRaw', motors, 'motors')
            self.writeData2HDF5(scanNum, 'AreaRaw', frames, 'frames')

        else:
            # no pilatus imagers for this scna
            print(
                'Scan #{0:.0f} includes no area detector frames!'.format(scanNum))
            frames = []
            motors, data = self.getScanData(scanNum)

        # if a normalizer is set to the normalization here after reading the data
        if self.normalizer and any(frames):
            frames = self.normalizer(data, ccd=frames)

        return frames, motors, data

    def convAreaScan(self, scanNum, hkl=False):
        """Convert the area detector data for a given scan number to q/hkl-space.

        Args:
            scanNum (int)   : Scan number of the spec scan.

        Returns:
            data (ndarray)   : area detector frames in q/hkl space.
            xaxis (ndarray)  : qx qxis.
            yaxis (ndarray)  : qy qxis.
            zaxis (ndarray)  : qz qxis.

        """

        # read the frames, motors and data
        frames, motors, data = self.readAreaScan(scanNum)

        # convert the data to q-space using the HXRD instance
        evalString = 'self.hxrd.Ang2Q.area('

        for motorName in self.motorNames:
            evalString += 'motors[\'' + motorName + '\'], '

        if hkl is True:
            evalString += 'UB=self.UB, '

        evalString += 'delta=self.delta)'

        x, y, z = eval(evalString)

        # convert data to rectangular grid in reciprocal space using the gridder
        self.gridder(x, y, z, frames[:, :, :])

        data = (self.gridder.data)

        return data, self.gridder.xaxis, self.gridder.yaxis, self.gridder.zaxis

    def convAreaScan2Q(self, scanNum):
        """Convert the area detector data for a given scan number to q-space.

        Args:
            scanNum (int)   : Scan number of the spec scan.

        Returns:
            data (ndarray)   : area detector frames in q-space.
            xaxis (ndarray)  : qx qxis.
            yaxis (ndarray)  : qy qxis.
            zaxis (ndarray)  : qz qxis.

        """

        return self.convAreaScan(scanNum, hkl=False)

    def convAreaScan2HKL(self, scanNum):
        """Convert the area detector data for a given scan number to q-space.

        Args:
            scanNum (int)   : Scan number of the spec scan.

        Returns:
            data (ndarray)   : area detector frames in hkl-space.
            xaxis (ndarray)  : h qxis.
            yaxis (ndarray)  : k qxis.
            zaxis (ndarray)  : l qxis.

        """
        return self.convAreaScan(scanNum, hkl=True)

    def plotAreaScan(self, scanNum, hkl=False, levels=100, setGrid=True):
        """Plot the area detector data for a given scan number in q-space.

        Args:
            scanNum (int)   : Scan number of the spec scan.

        """

        if hkl is True:
            xlabelText = 'H'
            ylabelText = 'K'
            zlabelText = 'L'

            data, xaxis, yaxis, zaxis = self.convAreaScan2HKL(scanNum)
        else:
            xlabelText = r'$q_x$'
            ylabelText = r'$q_y$'
            zlabelText = r'$q_z$'

            # get the data to plot
            data, xaxis, yaxis, zaxis = self.convAreaScan2Q(scanNum)

        if self.plotLog:
            scaleType = 'log'

            def scaleFunc(x): return np.log10(x)
        else:
            scaleType = 'linear'

            def scaleFunc(x): return x

        from matplotlib import gridspec

        # do the plotting
        fig = plt.figure()
        # qy qx Map
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[3, 1],
                               height_ratios=[1, 3]
                               )

        plt.subplot(gs[2])

        z = np.trapz(data, zaxis, axis=2)

        x = yaxis
        y = xaxis
        plt.contourf(x, y, scaleFunc(z), levels)
        plt.xlabel(ylabelText, size=18)
        plt.ylabel(xlabelText, size=18)
        plt.xlim([min(x), max(x)])
        plt.ylim([min(y), max(y)])
        plt.grid(setGrid)

        ax = plt.subplot(gs[3])
#        temp = sum(z,axis=1)
        temp = np.trapz(z, yaxis, axis=1)
        plt.plot(temp, y, '-')
        ax.set_xscale(scaleType)

        plt.ylim([min(y), max(y)])
        plt.grid(setGrid)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        plt.ylabel(xlabelText, size=18, )

        ax = plt.subplot(gs[0])
#        temp = sum(z,axis=0)
        temp = np.trapz(z, xaxis, axis=0)
        plt.plot(x, temp, '-')
        ax.set_yscale(scaleType)

        plt.xlim([min(x), max(x)])
        plt.xlabel(ylabelText, size=18)
        plt.grid(setGrid)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        gs.tight_layout(fig)
        plt.show()

        #####################################################################
        fig = plt.figure()
        # qz qx Map
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[3, 1],
                               height_ratios=[1, 3]
                               )

        plt.subplot(gs[2])
#        z = sum(data,axis=1)
        z = np.trapz(data, yaxis, axis=1)

        x = zaxis
        y = xaxis
        plt.contourf(x, y, scaleFunc(z), levels)
        plt.xlabel(zlabelText, size=18)
        plt.ylabel(xlabelText, size=18)
        plt.xlim([min(x), max(x)])
        plt.ylim([min(y), max(y)])
        plt.grid(setGrid)

        ax = plt.subplot(gs[3])
#        temp = sum(z,axis=1)
        temp = np.trapz(z, zaxis, axis=1)
        plt.plot(temp, y)
        ax.set_xscale(scaleType)

        plt.ylim([min(y), max(y)])
        plt.grid(setGrid)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        plt.ylabel(xlabelText, size=18, )

        ax = plt.subplot(gs[0])
#        temp = sum(z,axis=0)
        temp = np.trapz(z, xaxis, axis=0)
        plt.plot(x, temp)
        ax.set_yscale(scaleType)

        plt.xlim([min(x), max(x)])
        plt.xlabel(zlabelText, size=18)
        plt.grid(setGrid)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        gs.tight_layout(fig)
        plt.show()

        #####################################################################
        fig = plt.figure()
        # qz qy Map
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[3, 1],
                               height_ratios=[1, 3]
                               )

        plt.subplot(gs[2])
#        z = sum(data,axis=0)
        z = np.trapz(data, xaxis, axis=0)
        x = zaxis
        y = yaxis
        plt.contourf(x, y, scaleFunc(z), levels)
        plt.xlabel(zlabelText, size=18)
        plt.ylabel(ylabelText, size=18)
        plt.xlim([min(x), max(x)])
        plt.ylim([min(y), max(y)])
        plt.grid(setGrid)

        ax = plt.subplot(gs[3])
#        temp = sum(z,axis=1)
        temp = np.trapz(z, zaxis, axis=1)
        plt.plot(temp, y)
        ax.set_xscale(scaleType)

        plt.ylim([min(y), max(y)])
        plt.grid(setGrid)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        plt.ylabel(ylabelText, size=18, )

        ax = plt.subplot(gs[0])
#        temp = sum(z,axis=0)
        temp = np.trapz(z, yaxis, axis=0)
        plt.plot(x, temp)
        ax.set_yscale(scaleType)

        plt.xlim([min(x), max(x)])
        plt.xlabel(zlabelText, size=18)
        plt.grid(setGrid)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        gs.tight_layout(fig)
        plt.show()

    def plotAreaScanQ(self, scanNum, levels=100, setGrid=True):
        """Plot the area detector data for a given scan number in q-space.

        Args:
            scanNum (int)   : Scan number of the spec scan.

        """

        self.plotAreaScan(scanNum, hkl=False, levels=levels, setGrid=setGrid)

    def plotAreaScanHKL(self, scanNum, levels=100, setGrid=True):
        """Plot the area detector data for a given scan number in hkl-space.

        Args:
            scanNum (int)   : Scan number of the spec scan.

        """

        self.plotAreaScan(scanNum, hkl=True, levels=levels, setGrid=setGrid)


###########################

class princtonPM3(areaDetector):
    """Inherit from areaDetector and add specfic routins for reading data files
    of princton instruments SPE files and goniometer setup at BESSY II PM3

    Attributes:
        inherited from area detector

    """

    # properties
    delta = [0, 0]
    motorNames = ['Theta', 'TwoTheta']

    def __init__(self, name, filePath, specFileExt=''):
        super().__init__(name, filePath, specFileExt)
        # set the path to the raw data frame files
        self.rawDataPath = self.filePath + '/ccd/' + \
            self.specFileName + '_{0:0>4d}.SPE'

    def readRawScan(self, scanNum):
        """Read the raw data of a Princton Instrument CCD scan.

        """

        from winspec import SpeFile
        frames = SpeFile(self.rawDataPath.format(scanNum)).data

        return frames


########################################

class pilatusXPP(areaDetector):
    """Inherit from spec and add capabilities to read Pilatus images from the
    BESSY II XPP beamline with its specific goniometer setup.

    Attributes:
        inherited from areaDetector
        pilatus (ImageReader(xrayutilities)): Pilatus100k-object for reading frames

    """

    # properties
    delta = [0, 0, 0, 0]
    motorNames = ['Theta', 'Chi', 'Phi', 'Two Theta']
    pilatus = ''

    def __init__(self, name, filePath, specFileExt=''):
        super().__init__(name, filePath, specFileExt)
        # set the path to the raw data frame files
        self.rawDataPath = self.filePath + \
            '/pilatus/S{0:0>5d}/' + self.specFileName + '_{0:.0f}_{2:.0f}.tif'

    def readRawScan(self, scanNum):
        """Read the raw data of a Pilatsu 100k scan.

        """

        motors, data = self.getScanData(scanNum)

        numPoints = len(data)  # number of points in the scan

        # initilize the frames array
        frames = np.zeros([numPoints, self.pilatus.nop1,
                           self.pilatus.nop2], dtype=np.int32)

        for i in range(1, numPoints, 1):
            # traverse all points in the scan
            # format the pilatus image path
            pfile = self.filePath.format(scanNum, i)
            img = self.pilatus.readImage(pfile)  # read the image
            frames[i, :, :] = img  # save the image in the return array

        return frames
