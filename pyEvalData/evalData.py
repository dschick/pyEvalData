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
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import xrayutilities as xu
import re
from uncertainties import unumpy
from .helpers import binData

"""
evalData module provide class definitions to read, average, plot, and fit data
from spec files. The base class "spec" supports only spec files. The spec class
allows for user-defined counters as predefinitions or even at run-time.
The spec file is converted into a hdf5 file.
All file I/O are handled by the xrayutility package.
The child-class pilatusXPP inherits from spec and has additional methods to read
Pilatus ((C) Dectris Ltd.) .tiff files and further convert them to q-space
using the xrayutility package.
"""

class spec(object):
    """Read, average, plot, and fit spec data.

    Attributes:
        name (str)              : Name of the spec file.
        specFileName (str)      : Full file name of the spec file.
        h5FileName (str)        : Full file name of the HDF5 file.
        filePath (str)          : Base path of the spec file.
        hdf5Path (str)          : Base path of the HDF5 file.
        specFile (object)       : specFile object of xrayutility package.
        updateBeforeRead (bool) : Boolean to force an update of the spec file
                                  prior reading each scan.
        overwriteHDF5 (bool)    : Boolean to force overwriting the HDF5 file.
        cList (List[str])       : List of counter names to evaluate.
        cDef (Dict{str:str})    : Dict of predefined counter names and
                                  definitions.
        xCol (str)              : spec counter or motor to plot as x-axis.
        t0 (float)              : approx. time zero for delay scans to
                                  determine the unpumped region of the data
                                  for normalization.
        motorNames (List[str])  : default axis of the goniometer for reading
                                  spec files by xrayutilities.
        customCounters (List[str]): List of custom counters - default is []
        mathKeys (List[str])    : List of keywords which are not replaced in
                                  counter names
        statisticType  (str)    : 'gauss' for normal averaging,
                                  'poisson' for counting statistics
        propagateErrors  (bool) : whether to propagate errors or not

    """

    # properties
    name = ''
    specFileName = ''
    h5FileName = ''
    filePath = './'
    hdf5Path = './'
    specFile = ''
    updateBeforeRead = False
    overwriteHDF5 = False
    cList = []
    cDef = {}
    xCol = ''
    t0 = 0
    # must be the same order as for xu experiment configuration
    # (first sample axis, last detector axis)
    motorNames = ['Theta', 'TwoTheta']
    customCounters = []
    mathKeys = ['mean', 'sum', 'diff', 'max', 'min', 'round', 'abs',
                        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                        'pi', 'exp', 'log', 'log10', 'sqrt']
    statisticType = 'gauss'
    propagateErrors = True
    removeOutliners = False
    sigmaOutliners = 0.1

    def __init__(self, name, filePath, specFileExt=''):
        """Initialize the class, set all file names and load the spec file.

        Args:
            name (str)                  : Name of the spec file.
            filePath (str)              : Base path of the spec and HDF5 files.
            specFileExt (Optional[str]) : File extension of the spec file,
                                          default is none.

        """
        self.name = name
        self.specFileName = self.name + specFileExt
        self.h5FileName = self.name + '.h5'
        self.filePath = filePath
        # load the spec data
        self.loadSpec()

    def loadSpec(self):
        """Load the spec data either from the hdf5 or from the spec file."""
        # check if the hdf5 file exists
        if not os.path.exists(self.hdf5Path + self.h5FileName):
            # no hdf5 file found --> read the spec file
            self.updateSpec()

    def updateSpec(self):
        """Update the current spec file if already in memory.
        Otherwise read it and write its content to the hdf5 file.

        """
        try:
            # try if spec file object already exist
            self.specFile.Update()
        except Exception as e:
            # load the spec file from disc
            self.specFile = xu.io.SPECFile(
                self.specFileName, path=self.filePath)
            self.specFile.Update()

        if not os.path.exists(os.path.join(self.hdf5Path, self.h5FileName)) or self.overwriteHDF5:
            # save the new or changed spec file content to the hdf5 file
            # if it does not exist
            self.specFile.Save2HDF5(os.path.join(
                self.hdf5Path, self.h5FileName))

    def getScanData(self, scanNum):
        """Read the spec data for a given scan number from the hdf5 file.

        Args:
            scanNum (int) : Number of the spec scan.

        Returns:
            motors (ndarray): Motors of the spec scan.
            data   (ndarray): Counters of the spec scan.

        """

        # if set, update the spec and hdf5 files before reading the scan data
        if self.updateBeforeRead:
            self.updateSpec()

        # read the scan from the hdf5 file
        try:
            # if no motorNames are given motors are set as empty array
            if len(self.motorNames) == 0:
                # read the data
                data = xu.io.geth5_scan(os.path.join(
                    self.hdf5Path, self.h5FileName), scanNum)
                motors = []
            else:
                # read the data providing the motorNames
                motors, data = xu.io.geth5_scan(os.path.join(self.hdf5Path, self.h5FileName),
                                                scanNum, *self.motorNames)

            # convert the data array to float64 since lmfit works better
            # is there a smarter way to do so?
            dt = data.dtype
            dt = dt.descr
            for i, thisType in enumerate(dt):
                dt[i] = (dt[i][0], 'float64')
            dt = np.dtype(dt)
            data = data.astype(dt)

            # convert list of motors to recarray
            motors = np.rec.array(motors, names=self.motorNames)
        except Exception as e:
            print(e)
            print('Scan #{0:.0f} not present in hdf5 file!'.format(scanNum))
            motors = []
            data = []

        return motors, data

    def getClist(self):
        """Return the list of counters to evaluate as list even if they are
        provided as Dict by the user.
        This method is only for backward compatibility to older versions.

        Returns:
            cList (List[str]): List of counter names to evaluate.

        """

        if isinstance(self.cList, dict):
            # the cList property is a dict, so retrun its keys as list
            cList = list(self.cList.keys())
        else:
            cList = list(self.cList)

        return cList

    def getLastFigNumber(self):
        """Return the last figure number of all opened figures for plotting
        data in the same figure during for-loops.

        Returns:
            figNumber (int): Last figure number of all opened figures.

        """

        try:
            # get the number of all opened figures
            figNumber = mpl._pylab_helpers.Gcf.get_active().num
        except Exception as e:
            # there are no figures open
            figNumber = 1

        return figNumber

    def getNextFigNumber(self):
        """Return the number of the next figure for plotting data in the
        same figure during for-loops.

        Returns:
            nextFigureNum (int): Next figure number of all opened figures.

        """

        return self.getLastFigNumber() + 1

    def traverseCounters(self, cList, specCols=''):
        """Traverse all counters and replace all predefined counter definitions.
        Returns also the included spec base counters for error propagation.

        Args:
            cList    (list) : Initial counter list.
            specCols (list) : Counters in spec file.

        Returns:
            resolvedCounters (list): Resolved counters.
            baseCounters (list)    : Base counters.

        """

        resolvedCounters = []
        baseCounters = []

        for colName in cList:
            colString, resBaseCounters = self.resolveCounterName(
                colName, specCols)

            resolvedCounters.append(colString)
            baseCounters.extend(resBaseCounters)

        # remove duplicates using list(set())
        return resolvedCounters, list(set(baseCounters))

    def resolveCounterName(self, colName, specCols=''):
        """Replace all predefined counter definitions in a counter name.
        The function works recursively.

        Args:
            colName (str) : Initial counter string.

        Returns:l
            colString (str): Resolved counter string.

        """

        reCall = False  # boolean to stop recursive calls

        baseCounters = []

        colString = colName

        for findcDef in self.cDef.keys():
            # check for all predefined counters
            searchPattern = r'\b' + findcDef + r'\b'
            if re.search(searchPattern, colString) is not None:
                if self.cDef[findcDef] in specCols:
                    # this counter definition is a base spec counter
                    baseCounters.append(self.cDef[findcDef])
                # found a predefined counter
                # recursive call if predefined counter must be resolved again
                reCall = True
                # replace the counter definition in the string
                (colString, _) = re.subn(searchPattern,
                                         '(' + self.cDef[findcDef] + ')', colString)

        if reCall:
            # do the recursive call
            colString, recBaseCounters = self.resolveCounterName(
                colString, specCols)
            baseCounters.extend(recBaseCounters)

        for findcDef in specCols:
            # check for all base spec counters
            searchPattern = r'\b' + findcDef + r'\b'
            if re.search(searchPattern, colString) is not None:
                baseCounters.append(findcDef)

        return colString, baseCounters

    def colString2evalString(self, colString, arrayName='specData'):
        """Use regular expressions in order to generate an evaluateable string
        from the counter string in order to append the new counter to the
        spec data.

        Args:
            colString (str) : Definition of the counter.
            mode (int)      : Flag for different modes

        Returns:
            evalString (str): Evaluateable string to add the new counter
                              to the spec data.

        """

        # search for alphanumeric counter names in colString
        iterator = re.finditer(
            '([0-9]*[a-zA-Z\_]+[0-9]*[a-zA-Z]*)*', colString)
        # these are keys which should not be replaced but evaluated
        math_keys = list(self.mathKeys)
        keys = math_keys.copy()

        for key in iterator:
            # traverse all found counter names
            if len(key.group()) > 0:
                # the match is > 0
                if not key.group() in keys:
                    # the counter name is not in the keys list

                    # remember this counter name in the key list in order
                    # not to replace it again
                    keys.append(key.group())
                    # the actual replacement
                    (colString, _) = re.subn(r'\b'+key.group()+r'\b',
                                             arrayName + '[\'' + key.group() + '\']', colString)

        # add 'np.' prefix to numpy functions/math keys
        for mk in math_keys:
            (colString, _) = re.subn(r'\b' + mk + r'\b', 'np.' + mk, colString)
        return colString

    def addCustomCounters(self, specData, scanNum, baseCounters):
        """Add custom counters to the spec data array.
        This is a stub for child classes.

        Args:
            specData (ndarray) : Data array from the spec scan.
            scanNum (int)  : Scan number of the spec scan.
            baseCounters list(str) : List of the base spec and custom counters
                                     from the cList and xCol.

        Returns:
            specData (ndarray): Updated data array from the spec scan.

        """

        return specData

    def avgNbinScans(self, scanList, xGrid=np.array([]), binning=True):
        """Averages data defined by the counter list, cList, onto an optional
        xGrid. If no xGrid is given the x-axis data of the first scan in the
        list is used instead.

        Args:
            scanList (List[int])      : List of scan numbers.
            xGrid (Optional[ndarray]) : Grid to bin the data to -
                                        default in empty so use the
                                        x-axis of the first scan.

        Returns:
            avgData (ndarray) : Averaged data for the scan list.
            stdData (ndarray) : Standart derivation of the data for the scan list.
            errData (ndarray) : Error of the data for the scan list.
            name (str)        : Name of the data set.

        """

        # generate the name of the data set from the spec file name and scanList
        name = self.specFileName + "_{0:03d}".format(scanList[0])

        # get the counters which should be evaluated
        cList = self.getClist()
        if not cList:
            raise Exception('No cList is defined. Do not know what to plot!')
            return
        # process also the xCol as counter in order to allow for newly defined xCols
        if not self.xCol:
            raise Exception('No xCol is defined. Do not know what to plot!')
            return
        if self.xCol not in cList:
            cList.append(self.xCol)

        specCols = []
        concatData = np.array([])
        for i, scanNum in enumerate(scanList):
            # traverse the scan list and read data
            try:
                # try to read the motors and data of this scan
                motors, specData = self.getScanData(scanNum)
            except Exception as e:
                raise
                print('Scan #' + scanNum + ' not found, skipping')

            if i == 0 or len(specCols) == 0:  # we need to evaluate this only once
                # these are the base spec counters which are present in the data
                # file plus custom counters
                specCols = list(
                    set(list(specData.dtype.names) + self.customCounters))

                # resolve the cList and retrieve the resolves counters and the
                # necessary base spec counters for error propagation
                resolvedCounters, baseCounters = self.traverseCounters(
                    cList, specCols)

                # add custom counters if defined
                specData = self.addCustomCounters(
                    specData, scanNum, baseCounters)

                # counter names and resolved strings for further calculations
                if self.statisticType == 'poisson' or self.propagateErrors:
                    # for error propagation we just need the base spec counters
                    # and the xCol
                    colNames = baseCounters[:]
                    colStrings = baseCounters[:]
                    # add the xCol to both lists
                    colNames.append(self.xCol)
                    colStrings.append(resolvedCounters[cList.index(self.xCol)])
                else:
                    # we need to average the resolved counters
                    colNames = cList[:]
                    colStrings = resolvedCounters[:]

                # create the dtype of the return array
                dtypes = []
                for colName in cList:
                    dtypes.append((colName, '<f8'))

            data = np.array([])
            # read data into data array
            for colString, colName in zip(colStrings, colNames):
                # traverse the counters in the cList and append to data if not
                # already present
                evalString = self.colString2evalString(
                    colString, arrayName='specData')

                if len(data) == 0:
                    data = np.array(eval(evalString), dtype=[(colName, float)])
                elif colName not in data.dtype.names:
                    data = eval('recfuncs.append_fields(data,\'' + colName + '\',data=(' +
                                evalString + '), dtypes=float, asrecarray=True, usemask=False)')

            if i > 0:
                # this is not the first scan in the list so append the data to
                # the concatenated data array
                concatData = np.concatenate((concatData, data), axis=0)
            else:
                concatData = data

                if len(xGrid) == 0:
                    # if no xGrid is given we use the xData of the first scan instead
                    xGrid = concatData[self.xCol]

        # remove xCol from cList and resolved counters for further treatment
        del resolvedCounters[cList.index(self.xCol)]
        cList.remove(self.xCol)

        try:
            # bin the concatenated data to the xGrid
            # if a custom counter was calculated it might have a different length
            # than the spec counters which will result in an error while binning data
            # from a default spec counter and a custom counter.
            if binning:
                xGridReduced, _, _, _, _, _, _, _, _ = binData(
                    concatData[self.xCol], concatData[self.xCol], xGrid)
            else:
                xGridReduced = xGrid
            # create empty arrays for averages, std and errors
            avgData = np.recarray(np.shape(xGridReduced)[0], dtype=dtypes)
            stdData = np.recarray(np.shape(xGridReduced)[0], dtype=dtypes)
            errData = np.recarray(np.shape(xGridReduced)[0], dtype=dtypes)

            if self.statisticType == 'poisson':
                binStat = 'sum'
            else:  # gauss
                binStat = 'mean'

            if binning:
                if self.statisticType == 'poisson' or self.propagateErrors:
                    # propagate errors using the uncertainties package

                    # create empty dict for uncertainties data arrays
                    uncDataErr = {}
                    uncDataStd = {}

                    for col in baseCounters:
                        # for all cols in the cList bin the data to the xGrid an calculate
                        # the averages, stds and errors
                        y, avgData[self.xCol], yErr, errData[self.xCol], yStd,\
                            stdData[self.xCol], _, _, _ =\
                                binData(concatData[col], concatData[self.xCol],
                                    xGridReduced, statistic=binStat)
                        # add spec base counters to uncData arrays
                        uncDataStd[col] = unumpy.uarray(y, yStd)
                        uncDataErr[col] = unumpy.uarray(y, yErr)

                    for colName, colString in zip(cList, resolvedCounters):

                        evalString = self.colString2evalString(
                            colString, arrayName='uncDataErr')
                        temp = eval(evalString)

                        avgData[colName] = unumpy.nominal_values(temp)
                        errData[colName] = unumpy.std_devs(temp)

                        evalString = self.colString2evalString(
                            colString, arrayName='uncDataStd')
                        temp = eval(evalString)
                        stdData[colName] = unumpy.std_devs(temp)

                else:
                    # no error propagation but averaging of individual scans
                    for col in cList:
                        # for all cols in the cList bin the data to the xGrid an calculate
                        # the averages, stds and errors
                        avgData[col], avgData[self.xCol], errData[col], errData[self.xCol],\
                            stdData[col], stdData[self.xCol], _, _, _ =\
                                binData(concatData[col], concatData[self.xCol],
                                        xGridReduced, statistic=binStat)
            else:
                for col in cList:
                    avgData[col] = concatData[col]
                    avgData[self.xCol] = concatData[self.xCol]
                    errData[col] = 0
                    errData[self.xCol] = 0
                    stdData[col] = 0
                    stdData[self.xCol] = 0

        except Exception as e:
            raise
            print('xCol and yCol must have the same length --> probably you try plotting a custom'
                  ' counter together with a spec counter')

        return avgData, stdData, errData, name

    def writeData2HDF5(self, scanNum, childName, data, dataName):
        """Write data for a given scan number to the HDF5 file.

        Args:
            scanNum (int)   : Scan number of the spec scan.
            childName (str) : Name of the child where to save the data to.
            data (ndarray)  : Data array
            dataName (str)  : Name of the dataset.

        """

        # open the HDF5 file
        with xu.io.helper.xu_h5open(os.path.join(self.hdf5Path, self.h5FileName), mode='a') as h5:

            h5g = h5.get(list(h5.keys())[0])  # get the root
            scan = h5g.get("scan_%d" % scanNum)  # get the current scan
            try:
                # try to create the new subgroup for the area detector data
                scan.create_group(childName)
            except Exception as e:
                np.void

            g5 = scan[childName]  # this is the new group

            try:
                # add the data to the group
                g5.create_dataset(dataName, data=data,
                                  compression="gzip", compression_opts=9)
            except Exception as e:
                np.void

            h5.flush()  # write the data to the file

    def readDataFromHDF5(self, scanNum, childName, dataName):
        """Read data for a given scan number from the HDF5 file.

        Args:
            scanNum (int)   : Scan number of the spec scan.
            childName (str) : Name of the child where to save the data to.
            dataName (str)  : Name of the dataset.

        Returns:
            data (ndarray): Data array from the spec scan.

        """

        # open the HDF5 file
        with xu.io.helper.xu_h5open(os.path.join(self.hdf5Path, self.h5FileName), mode='a') as h5:
            h5g = h5.get(list(h5.keys())[0])  # get the root

            try:
                scan = h5g.get("scan_%d" % scanNum)  # get the current scan
                # access the child if a childName is given
                if len(childName) == 0:
                    g5 = scan
                else:
                    g5 = scan[childName]

                data = g5[dataName][:]  # get the actual dataset
            except Exception as e:
                # if no data is available return False
                data = False

        return data

    def plotScans(self, scanList, ylims=[], xlims=[], figSize=[], xGrid=[],
                  yErr='std', xErr='std', norm2one=False, binning=True,
                  labelText='', titleText='', skipPlot=False, gridOn=True,
                  yText='', xText='', fmt='-o'):
        """Plot a list of scans from the spec file.
        Various plot parameters are provided.
        The plotted data are returned.

        Args:
            scanList (List[int])        : List of scan numbers.
            ylims (Optional[ndarray])   : ylim for the plot.
            xlims (Optional[ndarray])   : xlim for the plot.
            figSize (Optional[ndarray]) : Figure size of the figure.
            xGrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            yErr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xErr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.
            labelText (Optional[str])   : Label of the plot - default is none.
            titleText (Optional[str])   : Title of the figure - default is none.
            skipPlot (Optional[bool])   : Skip plotting, just return data
                                          default is False.
            gridOn (Optional[bool])     : Add grid to plot - default is True.
            yText (Optional[str])       : y-Label of the plot - defaults is none.
            xText (Optional[str])       : x-Label of the plot - defaults is none.
            fmt (Optional[str])         : format string of the plot - defaults is -o.

        Returns:
            y2plot (OrderedDict)    : y-data which was plotted.
            x2plot (ndarray)        : x-data which was plotted.
            yerr2plot (OrderedDict) : y-error which was plotted.
            xerr2plot (ndarray)     : x-error which was plotted.
            name (str)              : Name of the data set.

        """

        # initialize the y-data as ordered dict in order to allow for multiple
        # counters at the same time
        y2plot = collections.OrderedDict()
        yerr2plot = collections.OrderedDict()

        # get the averaged data, stds and errors for the scan list and the xGrid
        avgData, stdData, errData, name = self.avgNbinScans(
            scanList, xGrid=xGrid, binning=binning)

        # set the error data
        if xErr == 'std':
            xErrData = stdData
        elif xErr == 'err':
            xErrData = errData
        else:
            xErrData = np.zeros_like(stdData)

        if yErr == 'std':
            yErrData = stdData
        elif yErr == 'err':
            yErrData = errData
        else:
            yErrData = np.zeros_like(stdData)

        # set x-data and errors
        x2plot = avgData[self.xCol]
        xerr2plot = xErrData[self.xCol]

        # plot all keys in the clist
        cList = self.getClist()
        for col in cList:
            # traverse the counter list

            # save the counter data and errors in the ordered dictionary
            y2plot[col] = avgData[col]
            yerr2plot[col] = yErrData[col]

            if norm2one:
                # normalize the y-data to 1 for t < t0
                # just makes sense for delay scans
                beforeZero = y2plot[col][x2plot <= self.t0]
                y2plot[col] = y2plot[col]/np.mean(beforeZero)
                yerr2plot[col] = yerr2plot[col]/np.mean(beforeZero)

            if len(labelText) == 0:
                # if no labelText is given use the counter name
                lt = col
            else:
                if len(cList) > 1:
                    # for multiple counters add the counter name to the label
                    lt = labelText + ' | ' + col
                else:
                    # for a single counter just use the labelText
                    lt = labelText

            if not skipPlot:
                # plot the errorbar for each counter
                if (xErr == 'none') & (yErr == 'none'):
                    plt.plot(x2plot, y2plot[col], fmt, label=lt)
                else:
                    plt.errorbar(
                        x2plot, y2plot[col], fmt=fmt, label=lt,
                        xerr=xerr2plot, yerr=yerr2plot[col])

        if not skipPlot:
            # add a legend, labels, title and set the limits and grid
            plt.legend(frameon=True, loc=0, numpoints=1)
            plt.xlabel(self.xCol)
            if xlims:
                plt.xlim(xlims)
            if ylims:
                plt.ylim(ylims)
            if len(titleText) > 0:
                plt.title(titleText)
            else:
                plt.title(name)
            if len(xText) > 0:
                plt.xlabel(xText)

            if len(yText) > 0:
                plt.ylabel(yText)

            if gridOn:
                plt.grid(True)

        return y2plot, x2plot, yerr2plot, xerr2plot, name

    def plotMeshScan(self, scanNum, skipPlot=False, gridOn=False, yText='', xText='',
                     levels=20, cBar=True):
        """Plot a single mesh scan from the spec file.
        Various plot parameters are provided.
        The plotted data are returned.

        Args:
            scanNum (int)               : Scan number of the spec scan.
            skipPlot (Optional[bool])   : Skip plotting, just return data
                                          default is False.
            gridOn (Optional[bool])     : Add grid to plot - default is False.
            yText (Optional[str])       : y-Label of the plot - defaults is none.
            xText (Optional[str])       : x-Label of the plot - defaults is none.
            levels (Optional[int])      : levels of contour plot - defaults is 20.
            cBar (Optional[bool])       : Add colorbar to plot - default is True.

        Returns:
            xx, yy, zz              : x,y,z data which was plotted

        """

        from matplotlib.mlab import griddata
        from matplotlib import gridspec

        # read data from spec file
        try:
            # try to read the motors and data of this scan
            motors, specData = self.getScanData(scanNum)
        except Exception as e:
            print('Scan #' + scanNum + ' not found, skipping')

        dt = specData.dtype
        dt = dt.descr

        xMotor = dt[0][0]
        yMotor = dt[1][0]

        X = specData[xMotor]
        Y = specData[yMotor]

        xx = np.sort(np.unique(X))
        yy = np.sort(np.unique(Y))

        cList = self.getClist()

        if len(cList) > 1:
            print('WARNING: Only the first counter of the cList is plotted.')

        Z = specData[cList[0]]

        zz = griddata(X, Y, Z, xx, yy, interp='linear')

        if not skipPlot:

            if cBar:
                gs = gridspec.GridSpec(4, 2,
                                       width_ratios=[3, 1],
                                       height_ratios=[0.2, 0.1, 1, 3]
                                       )
                k = 4
            else:
                gs = gridspec.GridSpec(2, 2,
                                       width_ratios=[3, 1],
                                       height_ratios=[1, 3]
                                       )
                k = 0

            ax1 = plt.subplot(gs[0+k])

            plt.plot(xx, np.mean(zz, 0), label='mean')

            plt.plot(xx, zz[np.argmax(np.mean(zz, 1)), :], label='peak')

            plt.xlim([min(xx), max(xx)])
            plt.legend(loc=0)
            ax1.xaxis.tick_top()
            if gridOn:
                plt.grid(True)

            plt.subplot(gs[2+k])

            plt.contourf(xx, yy, zz, levels, cmap='viridis')

            plt.xlabel(xMotor)
            plt.ylabel(yMotor)

            if len(xText) > 0:
                plt.xlabel(xText)

            if len(yText) > 0:
                plt.ylabel(yText)

            if gridOn:
                plt.grid(True)

            if cBar:
                cb = plt.colorbar(cax=plt.subplot(
                    gs[0]), orientation='horizontal')
                cb.ax.xaxis.set_ticks_position('top')
                cb.ax.xaxis.set_label_position('top')

            ax4 = plt.subplot(gs[3+k])

            plt.plot(np.mean(zz, 1), yy)
            plt.plot(zz[:, np.argmax(np.mean(zz, 0))], yy)
            plt.ylim([np.min(yy), np.max(yy)])

            ax4.yaxis.tick_right()
            if gridOn:
                plt.grid(True)

        return xx, yy, zz

    def plotScanSequence(self, scanSequence, ylims=[], xlims=[], figSize=[],
                         xGrid=[], yErr='std', xErr='std', norm2one=False,
                         binning=True, sequenceType='', labelText='',
                         titleText='', skipPlot=False, gridOn=True, yText='',
                         xText='', fmt='-o'):
        """Plot a list of scans from the spec file.
        Various plot parameters are provided.
        The plotted data are returned.

        Args:
            scanSequence (ndarray[List[int]
                          , int/str])   : Sequence of scan lists and parameters.
            ylims (Optional[ndarray])   : ylim for the plot.
            xlims (Optional[ndarray])   : xlim for the plot.
            figSize (Optional[ndarray]) : Figure size of the figure.
            xGrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            yErr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xErr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.
            sequenceType (Optional[str]): Type of the sequence: [fluence, delay,
                                          energy, theta, position, voltage, none,
                                          text] - default is enumeration.
            labelText (Optional[str])   : Label of the plot - default is none.
            titleText (Optional[str])   : Title of the figure - default is none.
            skipPlot (Optional[bool])   : Skip plotting, just return data
                                          default is False.
            gridOn (Optional[bool])     : Add grid to plot - default is True.
            yText (Optional[str])       : y-Label of the plot - defaults is none.
            xText (Optional[str])       : x-Label of the plot - defaults is none.
            fmt (Optional[str])         : format string of the plot - defaults is -o.

        Returns:
            sequenceData (OrderedDict) : Dictionary of the averaged scan data.
            parameters (ndarray)       : Parameters of the sequence.
            names (List[str])          : List of names of each data set.
            labelTexts (List[str])     : List of labels for each data set.

        """

        # initialize the return data
        sequenceData = collections.OrderedDict()
        names = []
        labelTexts = []
        parameters = []

#        pb = ProgressBar(len(scanSequence), title='Read Data', key='scanSequence')
#        for i in pb:
#            scanList = scanSequence[i,0]
#            parameter = scanSequence[i,1]
        for i, (scanList, parameter) in enumerate(scanSequence):
            # traverse the scan sequence

            parameters.append(parameter)
            # format the parameter as label text of this plot if no label text
            # is given
            if len(labelText) == 0:
                if sequenceType == 'fluence':
                    lt = str.format('{:.2f}  mJ/cm²', parameter)
                elif sequenceType == 'delay':
                    lt = str.format('{:.2f}  ps', parameter)
                elif sequenceType == 'energy':
                    lt = str.format('{:.2f}  eV', parameter)
                elif sequenceType == 'theta':
                    lt = str.format('{:.2f}  deg', parameter)
                elif sequenceType == 'temperature':
                    lt = str.format('{:.2f}  K', parameter)
                elif sequenceType == 'position':
                    lt = str.format('{:.2f}  mm', parameter)
                elif sequenceType == 'voltage':
                    lt = str.format('{:.2f}  V', parameter)
                elif sequenceType == 'current':
                    lt = str.format('{:.2f}  A', parameter)
                elif sequenceType == 'scans':
                    lt = str(scanList)
                elif sequenceType == 'none':
                    # no parameter for single scans
                    lt = ''
                elif sequenceType == 'text':
                    # parameter is a string
                    lt = parameter
                else:
                    # no sequence type is given --> enumerate
                    lt = str.format('#{}', i+1)

            # get the plot data for the scan list
            y2plot, x2plot, yerr2plot, xerr2plot, name = self.plotScans(
                scanList,
                ylims=ylims,
                xlims=xlims,
                figSize=figSize,
                xGrid=xGrid,
                yErr=yErr,
                xErr=xErr,
                norm2one=norm2one,
                binning=binning,
                labelText=lt,
                titleText=titleText,
                skipPlot=skipPlot,
                gridOn=gridOn,
                yText=yText,
                xText=xText,
                fmt=fmt
            )

            if self.xCol not in sequenceData.keys():
                # if the xCol is not in the return data dict - add the key
                sequenceData[self.xCol] = []
                sequenceData[self.xCol + 'Err'] = []

            # add the x-axis data to the return data dict
            sequenceData[self.xCol].append(x2plot)
            sequenceData[self.xCol + 'Err'].append(xerr2plot)

            for counter in y2plot:
                # traverse all counters in the data set
                if counter not in sequenceData.keys():
                    # if the counter is not in the return data dict - add the key
                    sequenceData[counter] = []
                    sequenceData[counter + 'Err'] = []

                # add the counter data to the return data dict
                sequenceData[counter].append(y2plot[counter])
                sequenceData[counter + 'Err'].append(yerr2plot[counter])

            # append names and labels to their lists
            names.append(name)
            labelTexts.append(lt)

        return sequenceData, parameters, names, labelTexts

    def exportScanSequence(self, scanSequence, path, fileName, yErr='std',
                           xErr='std', xGrid=[], norm2one=False, binning=True):
        """Exports spec data for each scan list in the sequence as individual file.

        Args:
            scanSequence (ndarray[List[int]
                          , int/str])   : Sequence of scan lists and parameters.
            path (str)                  : Path of the file to export to.
            fileName (str)              : Name of the file to export to.
            yErr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xErr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            xGrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.

        """
        # get scanSequence data without plotting
        sequenceData, parameters, names, labelTexts = self.plotScanSequence(
            scanSequence,
            xGrid=xGrid,
            yErr=yErr,
            xErr=xErr,
            norm2one=norm2one,
            binning=binning,
            skipPlot=True)

        for i, labelText in enumerate(labelTexts):
            # travserse the sequence

            header = ''
            saveData = []
            for counter in sequenceData:
                # travserse all counters in the data

                # build the file header
                header = header + counter + '\t '
                # build the data matrix
                saveData.append(sequenceData[counter][i])

            # save data with header to text file
            np.savetxt('%s/%s_%s.dat' % (path,
                                         fileName, "".join(x for x in labelText if x.isalnum())),
                                           np.r_[saveData].T, delimiter='\t', header=header)

    def fitScans(self, scans, mod, pars, ylims=[], xlims=[], figSize=[], xGrid=[],
                 yErr='std', xErr='std', norm2one=False, binning=True,
                 sequenceType='text', labelText='', titleText='', yText='',
                 xText='', select='', fitReport=0, showSingle=False,
                 weights=False, fitMethod='leastsq', offsetT0=False,
                 plotSeparate=False, gridOn=True, fmt='o'):
        """Fit, plot, and return the data of scans.

            This is just a wrapper for the fitScanSequence method
        """
        scanSequence = [[scans, '']]
        return self.fitScanSequence(scanSequence, mod, pars, ylims, xlims, figSize,
                                    xGrid, yErr, xErr, norm2one, binning,
                                    'none', labelText, titleText, yText,
                                    xText, select, fitReport, showSingle,
                                    weights, fitMethod, offsetT0, plotSeparate,
                                    gridOn, fmt=fmt)

    def fitScanSequence(self, scanSequence, mod, pars, ylims=[], xlims=[], figSize=[],
                        xGrid=[], yErr='std', xErr='std', norm2one=False,
                        binning=True, sequenceType='', labelText='',
                        titleText='', yText='', xText='', select='',
                        fitReport=0, showSingle=False, weights=False,
                        fitMethod='leastsq', offsetT0=False,
                        plotSeparate=False, gridOn=True,
                        lastResAsPar=False, sequenceData=[], fmt='o'):
        """Fit, plot, and return the data of a scan sequence.

        Args:
            scanSequence (ndarray[List[int]
                          , int/str])   : Sequence of scan lists and parameters.
            mod (Model[lmfit])          : lmfit model for fitting the data.
            pars (Parameters[lmfit])    : lmfit parameters for fitting the data.
            ylims (Optional[ndarray])   : ylim for the plot.
            xlims (Optional[ndarray])   : xlim for the plot.
            figSize (Optional[ndarray]) : Figure size of the figure.
            xGrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            yErr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xErr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.
            sequenceType (Optional[str]): Type of the sequence: [fluence, delay,
                                          energy, theta] - default is fluence.
            labelText (Optional[str])   : Label of the plot - default is none.
            titleText (Optional[str])   : Title of the figure - default is none.
            yText (Optional[str])       : y-Label of the plot - defaults is none.
            xText (Optional[str])       : x-Label of the plot - defaults is none.
            select (Optional[str])      : String to evaluate as select statement
                                          for the fit region - default is none
            fitReport (Optional[int])   : Set the fit reporting level:
                                          [0: none, 1: basic, 2: full]
                                          default 0.
            showSingle (Optional[bool]) : Plot each fit seperately - default False.
            weights (Optional[bool])    : Use weights for fitting - default False.
            fitMethod (Optional[str])   : Method to use for fitting; refer to
                                          lmfit - default is 'leastsq'.
            offsetT0 (Optional[bool])   : Offset time scans by the fitted
                                          t0 parameter - default False.
            plotSeparate (Optional[bool]):A single plot for each counter
                                          default False.
            gridOn (Optional[bool])     : Add grid to plot - default is True.
            lastResAsPar (Optional[bool]): Use the last fit result as start
                                           values for next fit - default is False.
            sequenceData (Optional[ndarray]): actual exp. data are externally given.
                                              default is empty
            fmt (Optional[str])         : format string of the plot - defaults is -o.


        Returns:
            res (Dict[ndarray])        : Fit results.
            parameters (ndarray)       : Parameters of the sequence.
            sequenceData (OrderedDict) : Dictionary of the averaged scan data.equenceData

        """

        # get the last open figure number
        mainFigNum = self.getLastFigNumber()

        if not figSize:
            # use default figure size if none is given
            figSize = mpl.rcParams['figure.figsize']

        # initialization of returns
        res = {}  # initialize the results dict

        for i, counter in enumerate(self.getClist()):
            # traverse all counters in the counter list to initialize the returns

            # results for this counter is again a Dict
            res[counter] = {}

            if isinstance(pars, (list, tuple)):
                # the fit paramters might individual for each counter
                _pars = pars[i]
            else:
                _pars = pars

            for pname, par in _pars.items():
                # add a dict key for each fit parameter in the result dict
                res[counter][pname] = []
                res[counter][pname + 'Err'] = []

            # add some more results
            res[counter]['chisqr'] = []
            res[counter]['redchi'] = []
            res[counter]['CoM'] = []
            res[counter]['int'] = []
            res[counter]['fit'] = []

        if len(sequenceData) > 0:
            # get only the parameters
            _, parameters, names, labelTexts = self.plotScanSequence(
                scanSequence,
                ylims=ylims,
                xlims=xlims,
                figSize=figSize,
                xGrid=xGrid,
                yErr=yErr,
                xErr=xErr,
                norm2one=norm2one,
                binning=True,
                sequenceType=sequenceType,
                labelText=labelText,
                titleText=titleText,
                skipPlot=True)
        else:
            # get the sequence data and parameters
            sequenceData, parameters, names, labelTexts = self.plotScanSequence(
                scanSequence,
                ylims=ylims,
                xlims=xlims,
                figSize=figSize,
                xGrid=xGrid,
                yErr=yErr,
                xErr=xErr,
                norm2one=norm2one,
                binning=True,
                sequenceType=sequenceType,
                labelText=labelText,
                titleText=titleText,
                skipPlot=True)

        # this is the number of different counters
        numSubplots = len(self.getClist())

        # fitting and plotting the data
        l = 1  # counter for singlePlots

        for i, parameter in enumerate(parameters):
            # traverse all parameters of the sequence
            lt = labelTexts[i]
            name = names[i]

            x2plot = sequenceData[self.xCol][i]
            xerr2plot = sequenceData[self.xCol + 'Err'][i]

            if fitReport > 0:
                # plot for basics and full fit reporting
                print('')
                print('='*10 + ' Parameter: ' + lt + ' ' + '='*15)

            j = 0  # counter for counters ;)
            k = 1  # counter for subplots
            for counter in sequenceData:
                # traverse all counters in the sequence

                # plot only y counters - next is the coresp. error
                if j >= 2 and j % 2 == 0:

                    # add the counter name to the label for not seperate plots
                    if sequenceType == 'none':
                        _lt = counter
                    else:
                        if plotSeparate or numSubplots == 1:
                            _lt = lt
                        else:
                            _lt = lt + ' | ' + counter

                    # get the fit models and fit parameters if they are lists/tupels
                    if isinstance(mod, (list, tuple)):
                        _mod = mod[k-1]
                    else:
                        _mod = mod

                    if lastResAsPar and i > 0:
                        # use last results as start values for pars
                        _pars = pars
                        for pname, par in pars.items():
                            _pars[pname].value = res[counter][pname][i-1]
                    else:
                        if isinstance(pars, (list, tuple)):
                            _pars = pars[k-1]
                        else:
                            _pars = pars

                    # get the actual y-data and -errors for plotting and fitting
                    y2plot = sequenceData[counter][i]
                    yerr2plot = sequenceData[counter + 'Err'][i]

                    # evaluate the select statement
                    if select == '':
                        # select all
                        sel = np.ones_like(y2plot, dtype=bool)
                    else:
                        sel = eval(select)

                    # execute the select statement
                    y2plot = y2plot[sel]
                    x2plot = x2plot[sel]
                    yerr2plot = yerr2plot[sel]
                    xerr2plot = xerr2plot[sel]

                    # remove nans
                    y2plot = y2plot[~np.isnan(y2plot)]
                    x2plot = x2plot[~np.isnan(y2plot)]
                    yerr2plot = yerr2plot[~np.isnan(y2plot)]
                    xerr2plot = xerr2plot[~np.isnan(y2plot)]

                    # do the fitting with or without weighting the data
                    if weights:
                        out = _mod.fit(y2plot, _pars, x=x2plot,
                                       weights=1/yerr2plot, method=fitMethod)
                    else:
                        out = _mod.fit(y2plot, _pars, x=x2plot,
                                       method=fitMethod)

                    if fitReport > 0:
                        # for basic and full fit reporting
                        print('')
                        print('-'*10 + ' ' + counter + ': ' + '-'*15)
                        for key in out.best_values:
                            print('{:>12}:  {:>10.4f} '.format(
                                key, out.best_values[key]))

                    # set the x-offset for delay scans - offset parameter in
                    # the fit must be called 't0'
                    if offsetT0:
                        offsetX = out.best_values['t0']
                    else:
                        offsetX = 0

                    plt.figure(mainFigNum)  # select the main figure

                    if plotSeparate:
                        # use subplot for separate plotting
                        plt.subplot((numSubplots+numSubplots % 2)/2, 2, k)

                    # plot the fit and the data as errorbars
                    x2plotFit = np.linspace(
                        np.min(x2plot), np.max(x2plot), 10000)
                    plot = plt.plot(x2plotFit-offsetX,
                                    out.eval(x=x2plotFit), '-', lw=2, alpha=1)
                    plt.errorbar(x2plot-offsetX, y2plot, fmt=fmt, xerr=xerr2plot,
                                 yerr=yerr2plot, label=_lt, alpha=0.25, color=plot[0].get_color())

                    if len(parameters) > 5:
                        # move the legend outside the plot for more than
                        # 5 sequence parameters
                        plt.legend(bbox_to_anchor=(0., 1.08, 1, .102), frameon=True,
                                   loc=3, numpoints=1, ncol=3, mode="expand",
                                   borderaxespad=0.)
                    else:
                        plt.legend(frameon=True, loc=0, numpoints=1)

                    # set the axis limits, title, labels and gird
                    if xlims:
                        plt.xlim(xlims)
                    if ylims:
                        plt.ylim(ylims)
                    if len(titleText) > 0:
                        if isinstance(titleText, (list, tuple)):
                            plt.title(titleText[k-1])
                        else:
                            plt.title(titleText)
                    else:
                        plt.title(name)

                    if len(xText) > 0:
                        plt.xlabel(xText)

                    if len(yText) > 0:
                        if isinstance(yText, (list, tuple)):
                            plt.ylabel(yText[k-1])
                        else:
                            plt.ylabel(yText)

                    if gridOn:
                        plt.grid(True)

                    # show the single fits and residuals
                    if showSingle:
                        plt.figure(mainFigNum+l, figsize=figSize)
                        gs = mpl.gridspec.GridSpec(
                            2, 1, height_ratios=[1, 3], hspace=0.1)
                        ax1 = plt.subplot(gs[0])
                        markerline, stemlines, baseline = plt.stem(
                            x2plot-offsetX, out.residual, markerfmt=' ')
                        plt.setp(stemlines, 'color',
                                 plot[0].get_color(), 'linewidth', 2, alpha=0.5)
                        plt.setp(baseline, 'color', 'k', 'linewidth', 0)

                        ax1.xaxis.tick_top()
                        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
                        plt.ylabel('Residuals')
                        if xlims:
                            plt.xlim(xlims)
                        if ylims:
                            plt.ylim(ylims)

                        if len(xText) > 0:
                            plt.xlabel(xText)

                        if gridOn:
                            plt.grid(True)

                        if len(titleText) > 0:
                            if isinstance(titleText, (list, tuple)):
                                plt.title(titleText[k-1])
                            else:
                                plt.title(titleText)
                        else:
                            plt.title(name)
                        ax2 = plt.subplot(gs[1])
                        x2plotFit = np.linspace(
                            np.min(x2plot), np.max(x2plot), 1000)
                        ax2.plot(x2plotFit-offsetX, out.eval(x=x2plotFit),
                                 '-', lw=2, alpha=1, color=plot[0].get_color())
                        ax2.errorbar(x2plot-offsetX, y2plot, fmt=fmt, xerr=xerr2plot,
                                     yerr=yerr2plot, label=_lt, alpha=0.25,
                                     color=plot[0].get_color())
                        plt.legend(frameon=True, loc=0, numpoints=1)

                        if xlims:
                            plt.xlim(xlims)
                        if ylims:
                            plt.ylim(ylims)

                        if len(xText) > 0:
                            plt.xlabel(xText)

                        if len(yText) > 0:
                            if isinstance(yText, (list, tuple)):
                                plt.ylabel(yText[k-1])
                            else:
                                plt.ylabel(yText)

                        if gridOn:
                            plt.grid(True)
#                        show()

                        l += 1
                    if fitReport > 1:
                        # for full fit reporting
                        print('_'*40)
                        print(out.fit_report())

                    # add the fit results to the returns
                    for pname, par in _pars.items():
                        res[counter][pname] = np.append(
                            res[counter][pname], out.best_values[pname])
                        res[counter][pname + 'Err'] = np.append(
                            res[counter][pname + 'Err'], out.params[pname].stderr)

                    res[counter]['chisqr'] = np.append(
                        res[counter]['chisqr'], out.chisqr)
                    res[counter]['redchi'] = np.append(
                        res[counter]['redchi'], out.redchi)
                    res[counter]['CoM'] = np.append(
                        res[counter]['CoM'], sum(y2plot*x2plot)/sum(y2plot))
                    res[counter]['int'] = np.append(
                        res[counter]['int'], sum(y2plot))
                    res[counter]['fit'] = np.append(res[counter]['fit'], out)

                    k += 1

                j += 1
                # end if
            # end for

        plt.figure(mainFigNum)  # set as active figure

        return res, parameters, sequenceData


