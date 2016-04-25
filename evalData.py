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

from numpy import *
from numpy.lib.recfunctions import *
from numpy.core import records
import collections
from matplotlib.pyplot import *
import matplotlib as mpl
import os
import xrayutilities as xu
from scipy.stats import binned_statistic
#from ipy_progressbar import ProgressBar
import re

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
        filePath (str)          : Base path of the spec and HDF5 files.
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
    
    """
    
    # properties
    name             = ''
    specFileName     = ''
    h5FileName       = ''
    filePath         = './'
    specFile         = '' 
    updateBeforeRead = False
    overwriteHDF5    = False
    cList            = []
    cDef             = {}
    xCol             = ''
    t0               = 0
    motorNames       = ['Theta', 'TwoTheta'] # must be the same order as for xu experiment configuration (first sample axis, last detector axis)
    customCounters   = []
    mathKeys         = ['mean', 'sum', 'diff', 'max', 'min', 'round', 'abs', 
                        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 
                        'pi', 'exp', 'log', 'log10']
    statisticType    = 'gauss'
    
    def __init__(self, name, filePath, specFileExt=''):
        """Initialize the class, set all file names and load the spec file. 
        
        Args:
            name (str)                  : Name of the spec file.
            filePath (str)              : Base path of the spec and HDF5 files.
            specFileExt (Optional[str]) : File extension of the spec file, 
                                          default is none.        
        
        """
        self.name         = name
        self.specFileName = self.name + specFileExt
        self.h5FileName   = self.name + '.h5'
        self.filePath     = filePath
        # load the spec data        
        self.loadSpec()        
        
    
    def loadSpec(self):
        """Load the spec data either from the hdf5 or from the spec file."""
        # check if the hdf5 file exists        
        if not os.path.exists(self.filePath + self.h5FileName):
            # no hdf5 file found --> read the spec file            
            self.updateSpec()
            
    
    def updateSpec(self):
        """Update the current spec file if already in memory.
        Otherwise read it and write its content to the hdf5 file.
        
        """
        try:
            # try if spec file object already exist
            self.specFile.Update()  
        except:
            # load the spec file from disc
            self.specFile = xu.io.SPECFile(self.specFileName, path=self.filePath)
            self.specFile.Update() 
            
        if not os.path.exists(self.filePath + self.h5FileName) or self.overwriteHDF5:
            # save the new or changed spec file content to the hdf5 file
            # if it does not exist
            self.specFile.Save2HDF5(self.filePath + self.h5FileName)
        
    
    def getScanData(self,scanNum):
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
                data = xu.io.geth5_scan(self.filePath + self.h5FileName, scanNum)
                motors = []
            else:
                # read the data providing the motorNames
                motors, data = xu.io.geth5_scan(self.filePath + self.h5FileName, scanNum, *self.motorNames)
                        
            # convert the data array to float64 since lmfit works better
            # is there a smarter way to do so?
            dt = data.dtype
            dt = dt.descr
            for i, thisType in enumerate(dt):
                dt[i] = (dt[i][0], 'float64')                
            dt = dtype(dt)
            data = data.astype(dt)            
            
            # convert list of motors to recarray
            motors = rec.array(motors, names=self.motorNames)
        except:
            print('Scan #{0:.0f} not present in hdf5 file!'.format(scanNum))
            motors = []
            data   = []
                
        return motors, data
          
    
    def getClist(self):
        """Return the list of counters to evaluate as list even if they are 
        provided as Dict by the user.
        This method is only for backward compatibility to older versions.
            
        Returns:
            cList (List[str]): List of counter names to evaluate.
        
        """
        
        if isinstance(self.cList,dict):
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
        except:
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
    
    
    def resolveCounterName(self, col):
        """Replace all predefined counter definitions in a counter name.
        The function works recursively.

        Args:
            col (str) : Initial counter string.
            
        Returns:l
            colString (str): Resolved counter string.
        
        """
        
        reCall = False # boolean to stop recursive calls
        
        if col in self.cDef.keys(): 
            # its a predefinded counter so use the definition
            colString = self.cDef[col]  
            # recursive call if predefined counter must be resolved again
            reCall = True
        else: 
            # this counter is unknown so first check if we can replace any 
            # predefined counter, with its definition
            colString = col
                
            for findcDef in self.cDef.keys():
                # check for all predefined counters
                searchPattern = r'\b' + findcDef + r'\b'    
                if re.search(searchPattern,colString) != None:
                    # found a predefined counter 
                    # recursive call if predefined counter must be resolved again
                    reCall = True
                # replace the counter definition in the string
                (colString,_) = re.subn(searchPattern, '(' + self.cDef[findcDef] + ')', colString)
                #break

        if reCall:
            # do the recursive call
            colString = self.resolveCounterName(colString)
        
        return colString
        
    
    def colString2evalString(self, colString, colName):
        """Use regular expressions in order to generate an evaluateable string
        from the counter string in order to append the new counter to the 
        spec data.
        
        Args:
            colName (str)   : Name of the counter.
            colString (str) : Definition of the counter.
            
        Returns:
            evalString (str): Evaluateable string to add the new counter 
                              to the spec data.
        
        """
        
        # search for alphanumeric counter names in colString
        iterator = re.finditer('([0-9]*[a-zA-Z\_]+[0-9]*[a-zA-Z]*)*', colString)                  
        # these are keys which should not be replaced but evaluated        
        keys = list(self.mathKeys)
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
                    (colString,_) = re.subn(r'\b'+key.group()+r'\b', 'specData[\'' + key.group() + '\']', colString)
                    
        # generate the actual string for evaluation to append the new counter 
        # to the spec data array                     
        evalString = 'append_fields(data,\'' + colName + '\',data=(' + colString + '), dtypes=float, asrecarray=True)'        
        
        return evalString
        
    
    def addCustomCounters(self, data, scanNum):
        """Add custom counters to the spec data array.
        This is a stub for child classes.
        
        Args:
            data (ndarray) : Data array from the spec scan.
            scanNum (int)  : Scan number of the spec scan.
            
        Returns:
            data (ndarray): Data array from the spec scan.
        
        """
        
        return data
    
    
    def avgNbinScans(self,scanList,xGrid=array([])):
        """Averages data defined by the cunter list, cList, onto an optional 
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
        name= self.specFileName + "_{0:03d}".format(scanList[0])
        
        for i, scanNum in enumerate(scanList):
            # traverse the scan list
            try:
                # try to read the motors and data of this scan
                motors, specData = self.getScanData(scanNum)
            except:
                print('Scan #' + scanNum + ' not found, skipping')            
            
            # first add custom counters if defined
                                 
            
            # get the counters which should be evaluated
            cList = self.getClist()
            # process also the xCol as counter in order to allow for newly defined xCols
            cList.append(self.xCol) 
            data = array([])
            for col in cList:                 
                # traverse the counters in the cList
                if col in specData.dtype.names:
                    colName = col
                    colString = col
                    # append new col to data array
                    data = eval(self.colString2evalString(colString, colName))
                elif not col in self.customCounters:
                    # this counter is not in the spec file                      
                    colName = col
                    # replace colString with predefined counters recursively
                    colString = self.resolveCounterName(col)  
                    # append new col to data array
                    data = eval(self.colString2evalString(colString, colName))
                
            # remove xCol from cList for further treatment
            cList.remove(self.xCol)
            
            data = self.addCustomCounters(data,scanNum)             
            
            if i > 0:
                # this is not the first scan in the list so append the data to
                # the concatenated data array
                concatData = concatenate((concatData,data), axis=0)           
            else:
                concatData = data
                
                if len(xGrid) == 0:
                  # if no xGrid is given we use the xData of the first scan instead
                  xGrid =  concatData[self.xCol]    
        try:
            # bin the concatenated data to the xGrid
            # if a custom counter was calculated it might have a different length
            # than the spec counters which will result in an error while binning data
            # from a default spec counter and a custom counter.
            
            xGridReduced, _, _, _, _, _, _, _, _ = binData(concatData[self.xCol],concatData[self.xCol],xGrid)
            
            # create empty arrays for averages, std and errors
            avgData=recarray(shape(xGridReduced)[0],dtype=concatData.dtype)
            stdData=recarray(shape(xGridReduced)[0],dtype=concatData.dtype)
            errData=recarray(shape(xGridReduced)[0],dtype=concatData.dtype)
            
            if self.statisticType == 'poisson':
                binStat = 'sum'
            else: # gauss
                binStat = 'mean'
            
            for col in cList:
                # for all cols in the cList bin the data to the xGrid an calculate the averages, stds and errors
                
                    avgData[col], avgData[self.xCol], errData[col], errData[self.xCol], stdData[col], stdData[self.xCol], _, _, _ = binData(concatData[col],concatData[self.xCol],xGridReduced, statistic=binStat)
                
               
        except:
            print('xCol and yCol must have the same length --> probably you try plotting a custom counter together with a spec counter')
            
        return avgData, stdData, errData, name

    def plotScans(self,scanList, ylims=[], xlims=[], figSize=[], xGrid=[], yErr='std', xErr = 'std', norm2one=False, labelText='', titleText='', skipPlot=False, gridOn=True, yText='', xText=''):
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
            
        Returns:
            y2plot (OrderedDict)    : y-data which was plotted.
            x2plot (ndarray)        : x-data which was plotted.
            yerr2plot (OrderedDict) : y-error which was plotted.
            xerr2plot (ndarray)     : x-error which was plotted.
            name (str)              : Name of the data set.
        
        """
        
        # initialize the y-data as ordered dict in order to allow for multiple 
        # counters at the same time
        y2plot    = collections.OrderedDict()
        yerr2plot = collections.OrderedDict()
        
        # get the averaged data, stds and errors for the scan list and the xGrid
        avgData, stdData, errData, name = self.avgNbinScans(scanList, xGrid=xGrid)
        
        # set the error data
        if xErr == 'std':
            xErrData = stdData
        elif xErr == 'err':
            xErrData = errData
        else:
            xErrData = stdData
            xErrData[:] = 0
            
        if yErr == 'std':
            yErrData = stdData
        elif yErr == 'err':
            yErrData = errData
        else:
            yErrData = zeros_like(stdData)
        
        # set x-data and errors
        x2plot    = avgData[self.xCol]
        xerr2plot = xErrData[self.xCol]       
        
        # plot all keys in the clist
        cList = self.getClist()        
        for col in cList:
            # traverse the counter list
            
            # save the counter data and errors in the ordered dictionary
            y2plot[col]    = avgData[col]
            yerr2plot[col] = yErrData[col]
                        
            if norm2one:
                # normalize the y-data to 1 for t < t0
                # just makes sense for delay scans
                beforeZero = y2plot[col][x2plot <= self.t0]
                y2plot[col]     = y2plot[col]/mean(beforeZero)
                yerr2plot[col]  = yerr2plot[col]/mean(beforeZero)
            
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
                errorbar(x2plot,y2plot[col],fmt='-o',label=lt, xerr=xerr2plot, yerr=yerr2plot[col])
        
        if not skipPlot:
            # add a legend, labels, title and set the limits and grid
            legend(frameon=True,loc=0,numpoints=1)
            xlabel(self.xCol)
            if xlims:
                xlim(xlims)
            if ylims:
                ylim(ylims)
            if len(titleText) > 0:
                title(titleText)
            else:
                title(name)
            if len(xText) > 0:
                xlabel(xText)
                
            if len(yText) > 0:
                ylabel(yText)
                
            if gridOn:
                grid(True)   
        
        return y2plot, x2plot, yerr2plot, xerr2plot, name
            
    
    def plotScanSequence(self,scanSequence, ylims=[], xlims=[], figSize=[], xGrid=[], yErr='std', xErr = 'std',norm2one=False, sequenceType='', labelText='', titleText='', skipPlot=False, gridOn=True, yText='',xText=''):
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
                                          energy, theta, text] - default is fluence.
            labelText (Optional[str])   : Label of the plot - default is none. 
            titleText (Optional[str])   : Title of the figure - default is none. 
            skipPlot (Optional[bool])   : Skip plotting, just return data
                                          default is False.
            gridOn (Optional[bool])     : Add grid to plot - default is True.
            yText (Optional[str])       : y-Label of the plot - defaults is none.
            xText (Optional[str])       : x-Label of the plot - defaults is none.
            
        Returns:
            sequenceData (OrderedDict) : Dictionary of the averaged scan data.
            parameters (ndarray)       : Parameters of the sequence.
            names (List[str])          : List of names of each data set.
            labelTexts (List[str])     : List of labels for each data set.
        
        """
        
        # initialize the return data
        sequenceData= collections.OrderedDict()  
        names       = []
        labelTexts  = []
        parameters  = zeros([len(scanSequence),1])
        
#        pb = ProgressBar(len(scanSequence), title='Read Data', key='scanSequence')        
#        for i in pb:
#            scanList = scanSequence[i,0]
#            parameter = scanSequence[i,1]
        for i, (scanList, parameter) in enumerate(scanSequence):
            # traverse the scan sequence
            
            parameters[i] = parameter
            
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
                elif sequenceType == 'none':
                    #no parameter for single scans 
                    lt = ''
                elif sequenceType == 'text':
                    #parameter is a string
                    lt = parameter
                else:
                    #no sequence type is given --> enumerate
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
                                                            labelText=lt,
                                                            titleText=titleText, 
                                                            skipPlot=skipPlot,
                                                            gridOn=gridOn,
                                                            yText=yText,
                                                            xText=xText
                                                            )
                                                            
            if self.xCol not in sequenceData.keys():
                # if the xCol is not in the return data dict - add the key
                sequenceData[self.xCol]         = []
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
            
            
    def exportScanSequence(self,scanSequence,path,fileName, yErr='std', xErr = 'std', xGrid=[], norm2one=False):
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
                                                    xGrid = xGrid,
                                                    yErr = yErr,
                                                    xErr = xErr,
                                                    norm2one = norm2one,
                                                    skipPlot = True)
        
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
            savetxt('%s/%s_%s.dat' % (path,fileName,"".join(x for x in labelText if x.isalnum())), r_[saveData].T, delimiter = '\t', header=header)
            
    def fitScans(self,scans,mod,pars,ylims=[],xlims=[],figSize=[], xGrid=[], yErr='std', xErr = 'std', norm2one=False, sequenceType='text', labelText='', titleText='', yText='', xText='', select='', fitReport=0, showSingle=False, weights=False, fitMethod='leastsq', offsetT0 = False, plotSeparate = False, gridOn = True):
        scanSequence = [[scans, '']]
        return self.fitScanSequence(scanSequence,mod,pars,ylims,xlims,figSize, xGrid, yErr, xErr, norm2one, 'none', labelText, titleText, yText, xText, select, fitReport, showSingle, weights, fitMethod, offsetT0, plotSeparate, gridOn)
        
    
    def fitScanSequence(self,scanSequence,mod,pars,ylims=[],xlims=[],figSize=[], xGrid=[], yErr='std', xErr = 'std', norm2one=False, sequenceType='', labelText='', titleText='', yText='', xText='', select='', fitReport=0, showSingle=False, weights=False, fitMethod='leastsq', offsetT0 = False, plotSeparate = False, gridOn = True):
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
        res = {} # initialize the results dict

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
            
            # add some more results
            res[counter]['chisqr'] = []
            res[counter]['redchi'] = []
            res[counter]['CoM']    = []
            res[counter]['int']    = []
            res[counter]['fit']    = []
        
        
        # get the sequence data and parameters
        sequenceData, parameters, names, labelTexts = self.plotScanSequence(
                                            scanSequence, 
                                            ylims = ylims, 
                                            xlims = xlims, 
                                            figSize = figSize, 
                                            xGrid = xGrid, 
                                            yErr = yErr, 
                                            xErr = xErr,
                                            norm2one = norm2one, 
                                            sequenceType = sequenceType, 
                                            labelText = labelText, 
                                            titleText = titleText, 
                                            skipPlot=True)        
        
        # this is the number of different counters        
        numSubplots = len(self.getClist()) 
        
        # fitting and plotting the data
        l = 1 # counter for singlePlots
        
#        pb = ProgressBar(len(parameters), title='Fit Data', key='parameters')        
#        for i in pb:
#            parameter = parameters[i]
        for i, parameter in enumerate(parameters):
            # traverse all parameters of the sequence           
            lt          = labelTexts[i]         
            name        = names[i]
            
            x2plot = sequenceData[self.xCol][i]
            xerr2plot = sequenceData[self.xCol + 'Err'][i]
            
            if fitReport > 0:
                # plot for basics and full fit reporting
                print('')
                print('='*10 + ' Parameter: ' + lt + ' ' + '='*15)   
            
            j = 0 # counter for counters ;)
            k = 1 # counter for subplots
            for counter in sequenceData:
                # traverse all counters in the sequence
                
                # plot only y counters - next is the coresp. error                
                if j >= 2 and j%2 == 0:
                    
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
                        sel = ones_like(y2plot, dtype=bool)
                    else:
                        sel = eval(select)            
                    
                    # execute the select statement
                    y2plot = y2plot[sel]
                    x2plot = x2plot[sel]
                    yerr2plot = yerr2plot[sel]
                    xerr2plot = xerr2plot[sel]
                    
                    # remove nans
                    y2plot = y2plot[~isnan(y2plot)]
                    x2plot = x2plot[~isnan(y2plot)]
                    yerr2plot = yerr2plot[~isnan(y2plot)]
                    xerr2plot = xerr2plot[~isnan(y2plot)]        
                                
                    # do the fitting with or without weighting the data
                    if weights:
                        out  = _mod.fit(y2plot, _pars, x=x2plot, weights=1/yerr2plot, method=fitMethod)
                    else:
                        out  = _mod.fit(y2plot, _pars, x=x2plot, method=fitMethod)                         
                    
                    if fitReport > 0:
                        # for basic and full fit reporting
                        print('')
                        print('-'*10 + ' ' + counter + ': ' + '-'*15)  
                        for key in out.best_values:
                            print('{:>12}:  {:>10.4f} '.format(key,out.best_values[key]))
                    
                    # set the x-offset for delay scans - offset parameter in 
                    # the fit must be called 't0'
                    if offsetT0:
                        offsetX = out.best_values['t0']
                    else:
                        offsetX = 0
                        
                    figure(mainFigNum) # select the main figure                            
                    
                    if plotSeparate:
                        # use subplot for separate plotting
                        subplot( (numSubplots+numSubplots%2)/2,2,k)
                    
                    # plot the fit and the data as errorbars
                    x2plotFit = linspace(min(x2plot), max(x2plot), 1000)
                    plt = plot(x2plotFit-offsetX, out.eval(x=x2plotFit), '-', lw=2, alpha=1)
                    errorbar(x2plot-offsetX,y2plot,fmt='o', xerr=xerr2plot, yerr=yerr2plot, label=_lt, alpha=0.25, color=plt[0].get_color())
                    
                    if len(parameters) > 5:
                        # move the legend outside the plot for more than 
                        # 5 sequence parameters
                        legend(bbox_to_anchor=(0., 1.08, 1, .102), frameon=True,
                                       loc=3,numpoints=1,ncol=3, mode="expand", 
                                       borderaxespad=0.)
                    else:
                        legend(frameon=True,loc=0,numpoints=1)
                    
                    # set the axis limits, title, labels and gird
                    if xlims:
                        xlim(xlims)
                    if ylims:
                        ylim(ylims)
                    if len(titleText) > 0:
                        if isinstance(titleText, (list, tuple)):             
                            title(titleText[k-1])
                        else:
                            title(titleText)
                    else:
                        title(name)
                        
                    if len(xText) > 0:
                        xlabel(xText)
                        
                    if len(yText) > 0:
                        if isinstance(yText, (list, tuple)):             
                            ylabel(yText[k-1])
                        else:
                            ylabel(yText)
                    
                    if gridOn:
                        grid(True)  
                    
                    # show the single fits and residuals
                    if showSingle:
                        figure(mainFigNum+l, figsize=figSize)
#                        gs = mpl.gridspec.GridSpec(2*numSubplots*len(parameters),1, height_ratios=[1,3], hspace=0.1)
                        gs = mpl.gridspec.GridSpec(2,1, height_ratios=[1,3], hspace=0.1)
                        ax1 = subplot(gs[0])
                        markerline, stemlines, baseline = stem(x2plot-offsetX, out.residual, markerfmt=' ')
                        setp(stemlines, 'color', plt[0].get_color(), 'linewidth', 2, alpha=0.5)
                        setp(baseline, 'color','k', 'linewidth', 0)
                        
                        ax1.xaxis.tick_top()
                        ax1.yaxis.set_major_locator(MaxNLocator(3))
                        ylabel('Residuals')
                        if xlims:
                            xlim(xlims)
                        if ylims:
                            ylim(ylims)
                        
                        if len(xText) > 0:
                            xlabel(xText)
                            
                        if gridOn:
                            grid(True)                     
                        
                        if len(titleText) > 0:
                            if isinstance(titleText, (list, tuple)):             
                                title(titleText[k-1])
                            else:
                                title(titleText)
                        else:
                            title(name)
                        #print(i*k+k+numSubplots*len(parameters))
                        #ax2 = subplot(gs[i*numSubplots+k-1+numSubplots*len(parameters)])
                        ax2 = subplot(gs[1])
                        x2plotFit = linspace(min(x2plot), max(x2plot), 1000)
                        ax2.plot(x2plotFit-offsetX, out.eval(x=x2plotFit), '-', lw=2, alpha=1, color=plt[0].get_color())
                        ax2.errorbar(x2plot-offsetX,y2plot,fmt='o', xerr=xerr2plot, yerr=yerr2plot, label=_lt, alpha=0.25, color=plt[0].get_color())
                        legend(frameon=True,loc=0,numpoints=1)
                        xlabel('Delay [ps]')
                        ylabel('Rel. Change')
                        
                        if xlims:
                            xlim(xlims)
                        if ylims:
                            ylim(ylims)
                        
                        if len(xText) > 0:
                            xlabel(xText)
                        
                        if len(yText) > 0:
                            if isinstance(yText, (list, tuple)):             
                                ylabel(yText[k-1])
                            else:
                                ylabel(yText)
                    
                        if gridOn:
                            grid(True)                           
                        show()
                        
                        l += 1
                    if fitReport > 1:
                        # for full fit reporting
                        print('_'*40)
                        print((out.fit_report(pars)))
                    
                    # add the fit results to the returns
                    for pname, par in _pars.items():
                        res[counter][pname] = append(res[counter][pname], out.best_values[pname])
                        
                    res[counter]['chisqr'] = append(res[counter]['chisqr'], out.chisqr)
                    res[counter]['redchi'] = append(res[counter]['redchi'], out.chisqr)
                    res[counter]['CoM']    = append(res[counter]['CoM'], sum(y2plot*x2plot)/sum(y2plot))
                    res[counter]['int']    = append(res[counter]['int'], sum(y2plot))
                    res[counter]['fit']    = append(res[counter]['fit'], out)
                    
                    k += 1
                    
                j += 1
                # end if
            # end for             
            
        figure(mainFigNum) # set as active figure
                 
        return res, parameters, sequenceData


# sub classes of evalData.spec

class pilatusXPP(spec):
    """Inherit from spec and add capabilities to read Pilatus images from the 
    BESSY II XPP beamline with its specific goniometer setup.
    
    Attributes:
        overwriteHdf5 (bool)       : Boolean to force overwriting the HDF5 file
                                     default False.
        pilatus 
        (Pilatus100k[ImageReader]) : Instance of the Pilatus100k class
                                     to read the actual detector frames.
        hxrd (HXRD[xrayutilities]) : Instance of the HXRD class of the 
                                     xrayutilities.
        gridder 
        (gidder[xrayutilities])    : Instance of the gridder class of the 
                                     xrayutilities.
        normalizer 
        (IntensityNormalizer[xrayutilities])
                                   : Instance of the IntensityNormalizerr class 
                                     of the xrayutilities.
        delta (List[float])        : Offset angles of the goniometer axis: 
                                     Theta, Psi, Chi, Two_Theta
                                     default is [0,0,0,0].
        motorNames (List[str])     : List of goniometer motor names - default 
                                     is ['Theta', 'Chi', 'Phi', 'Two_Theta']
        customCounters (List[str]) : List of custom counters - default is 
                                     ['qx', 'qy', 'qz', 'QxMap', 'QyMap', 'QzMap']
        plotLog (bool)             : Boolean if subplots of RMS are log or lin
    
    """
    
    # properties    
    overwriteHDF5 = False
    pilatus       = ''
    hxrd          = ''
    gridder       = ''
    normalizer    = ''
    delta         = [0,0,0,0]
    motorNames    = ['Theta', 'Chi', 'Phi', 'Two Theta']
    customCounters= ['qx', 'qy', 'qz', 'QxMap', 'QyMap', 'QzMap']
    plotLog       = True
       
    
    def addCustomCounters(self,data,scanNum):
        """Add custom counters to the spec data array.
        Here we add the Qx, Qy, Qz maps and axises.
        
        Args:
            data (ndarray) : Data array from the spec scan.
            scanNum (int)  : Scan number of the spec scan.
            
        Returns:
            data (ndarray): Data array from the spec scan.
        
        """
        
        cList = self.getClist() # get the current counter list        
        cList.append(self.xCol) # process also the xCol        
        
        #check if any custom counter is in cList + xCol
        if set(cList) & set(self.customCounters):
            
            # calculate the Q data for the current scan number            
            Qmap, qx, qy, qz = self.convPilatusScan2Q(scanNum)
            
            # do the integration along the different axises
            QxMap = trapz(trapz(Qmap, qy, axis=1), qz, axis=1)#sum(sum(Qmap, axis=1),axis=1)
            QyMap = trapz(trapz(Qmap, qx, axis=0), qz, axis=1)#sum(sum(Qmap, axis=0),axis=1)
            QzMap = trapz(trapz(Qmap, qx, axis=0), qy, axis=0)#sum(sum(Qmap, axis=0),axis=0)
                       
            for col in set(cList) & set(self.customCounters):
                # append the custom counters to data array
                data = append_fields(data, col , data=eval(col) , dtypes=float, asrecarray=True)
            
        return data    
    
    
    def writePilatusData2HDF5(self, scanNum, childName, data, dataName):   
        """Write Pilatus data for a given scan number to the HDF5 file.
        
        Args:
            scanNum (int)   : Scan number of the spec scan.
            childName (str) : Name of the child where to save the data to.
            data (ndarray)  : Data array from the Pilatus data
            dataName (str)  : Name of the dataset.
        
        """
        
        # open the HDF5 file
        with xu.io.helper.xu_h5open(self.filePath + self.h5FileName, mode='a') as h5:
        
            h5g = h5.get(list(h5.keys())[0]) # get the root
            
            scan = h5g.get("scan_%d" % scanNum) # get the current scan   
            try:
                # try to create the new subgroup for the Pilatus data
                scan.create_group(childName)
            except:
                void
            
            g5 = scan[childName] # this is the new group
            
            try:
                # add the data to the group
                g5.create_dataset(dataName, data=data, compression="gzip", compression_opts=9)
            except:
                void
                
            h5.flush() # write the data to the file
                    
    
    def readPilatusDataFromHDF5(self, scanNum, childName, dataName):   
        """Read Pilatus data for a given scan number from the HDF5 file.
        
        Args:
            scanNum (int)   : Scan number of the spec scan.
            childName (str) : Name of the child where to save the data to.
            dataName (str)  : Name of the dataset.
            
        Returns:
            data (ndarray): Data array from the spec scan.
        
        """
        
        # open the HDF5 file
        with xu.io.helper.xu_h5open(self.filePath + self.h5FileName, mode='a') as h5:
            
            h5g = h5.get(list(h5.keys())[0]) # get the root                      
            
            try:
                scan = h5g.get("scan_%d" % scanNum) # get the current scan 
                # access the child if a childName is given
                if len(childName) == 0:
                    g5 = scan
                else:               
                    g5 = scan[childName]
                    
                data =  g5[dataName][:] # get the actual dataset
            except:
                # if no data is available return False
                data = False
            
        return data
    
    
    def readPilatusScan(self, scanNum):
        """Read the complete data of a Pilatus scan including the frames, 
        motors, and spec data.
        
        Args:
            scanNum (int)   : Scan number of the spec scan.
            
        Returns:
            frames (ndarray): Data array from the Pilatus data.
            motors (ndarray): Data array from the spec motors.
            data (ndarray)  : Data array from the spec scan.
        
        """
        
        # this is the file path to access the Pilatus images
        formatString = self.filePath + '/pilatus/S{0:0>5d}/{1}_{0:.0f}_{2:.0f}.tif'
        
        # update the spec file
        if self.updateBeforeRead:
            self.updateSpec()
        
        # check if pilatus images are already stored in hdf5 file               
        frames = self.readPilatusDataFromHDF5(scanNum, 'PilatusRaw', 'frames')
        
        if any(frames) and not self.overwriteHDF5:
            # if the data is present in the HDF5 file and we don't want to 
            # overwrite, read also the other datasets
            motors   = self.readPilatusDataFromHDF5(scanNum, 'PilatusRaw', 'motors')
            data   = self.readPilatusDataFromHDF5(scanNum, '', 'data')
            #print('Scan #{0:.0f} read from HDF5.'.format(scanNum))
        elif os.path.isfile(formatString.format(scanNum,self.name,1)):
            # data is not present in the HDF5 file but there are Pilatus images
            # on the disk, so read them and save them
        
            #print('Scan #{0:.0f} read from .tiff and saved to HDF5.'.format(scanNum))      
            
            # get the motors and data from the spec scan
            motors, data = self.getScanData(scanNum)
            
            numPoints = len(data) #  number of points in the scan
           
            # initilize the frames array
            frames = zeros([numPoints,self.pilatus.nop1,self.pilatus.nop2], dtype=int32)
            
            for i in range(1, numPoints, 1):
                # traverse all points in the scan
                pfile = formatString.format(scanNum,self.name,i) # format the pilatus image path
                img = self.pilatus.readImage(pfile) # read the image
                frames[i,:,:] = img # save the image in the return array
                
            # write the frames and motors to the HDF5 file
            self.writePilatusData2HDF5(scanNum, 'PilatusRaw', motors  , 'motors')
            self.writePilatusData2HDF5(scanNum, 'PilatusRaw', frames, 'frames')
            
        else:
            # no pilatus imagers for this scna
            print('Scan #{0:.0f} includes no Pilatus images!'.format(scanNum))
            frames = []
            motors = []
            data = []
        
        # if a normalizer is set to the normalization here after reading the data        
        if self.normalizer and any(frames):
            frames = self.normalizer(data, ccd=frames)
        
        return frames, motors, data
    
    
    def writeAllPilatusScans2HDF5(self):
        """Use this function with caution. It might take some time.
        Reads all scans from the spec file and save the Pilatus data, 
        if present, to the HDF5 file.
        Currently it allways overwrite the whole hdf5 file
        
        """
        
        # update the spec file in order to have the specFile object at hand
        self.updateSpec()
        
        for i , scan in enumerate(self.specFile.scan_list):
            # iterate over all scan in the specFile
            self.readPilatusScan(i+1) # read (and write) the pilatus data
        
    
    def convPilatusScan2Q(self, scanNum):
        """Convert the Pilatus data for a given scan number to q-space.
        
        Args:
            scanNum (int)   : Scan number of the spec scan.
            
        Returns:
            data (ndarray)   : Pilatus data in q-space.
            xaxis (ndarray)  : qx qxis.
            yaxis (ndarray)  : qy qxis.
            zaxis (ndarray)  : qz qxis.
        
        """
        
        # read the frames, motors and data        
        frames, motors, data = self.readPilatusScan(scanNum)   
        
        # convert the data to q-space using the HXRD instance            
        qx, qy, qz = self.hxrd.Ang2Q.area(motors[0] , motors[1] , motors[2] , motors[3], delta=self.delta)    
    
        # convert data to rectangular grid in reciprocal space using the gridder
        self.gridder(qx, qy, qz, frames[:,:,:])
        
        data = (self.gridder.data)
        
        return data, self.gridder.xaxis, self.gridder.yaxis, self.gridder.zaxis
        
        
    def plotPilatusScanQ(self, scanNum):
        """Plot the Pilatus data for a given scan number in q-space.
        
        Args:
            scanNum (int)   : Scan number of the spec scan.
        
        """
        
        if self.plotLog:
            scaleType = 'log'
            scaleFunc = lambda x: log10(x)
        else:
            scaleType = 'linear'
            scaleFunc = lambda x: x
        
        
        from matplotlib import gridspec        
        
        # get the data to plot
        data, xaxis, yaxis, zaxis = self.convPilatusScan2Q(scanNum)
        
        # do the plotting
        fig = figure()
        # qy qx Map
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[3,1],
                               height_ratios=[1,3]
                               )
        
        
        subplot(gs[2])
        
        #z = sum(data,axis=2)
        z = trapz(data,zaxis, axis=2)
        
        x = yaxis
        y = xaxis
        contourf(x,y,scaleFunc(z))
        xlabel(r'$Q_y$',size=18)
        ylabel(r'$Q_x$',size=18)
        xlim([min(x),max(x)])
        ylim([min(y),max(y)])
        grid()
        
        ax = subplot(gs[3])        
#        temp = sum(z,axis=1)
        temp = trapz(z,yaxis,axis=1)
        plot(temp,y, '-')
        ax.set_xscale(scaleType)
            
        ylim([min(y),max(y)])
        grid()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ylabel(r'$Q_x$',size=18, )
        
        ax = subplot(gs[0])
#        temp = sum(z,axis=0)
        temp = trapz(z,xaxis, axis=0)
        plot(x,temp, '-')
        ax.set_yscale(scaleType)
        
        xlim([min(x),max(x)])
        xlabel(r'$Q_y$',size=18)
        grid()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        
        
        gs.tight_layout(fig)
        show()
        
        #####################################################################
        fig=figure()
        # qz qx Map
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[3,1],
                               height_ratios=[1,3]
                               )
        
        subplot(gs[2])
#        z = sum(data,axis=1)
        z = trapz(data,yaxis, axis=1)
        
        x = zaxis
        y = xaxis
        contourf(x,y,scaleFunc(z))
        xlabel(r'$Q_z$',size=18)
        ylabel(r'$Q_x$',size=18)
        xlim([min(x),max(x)])
        ylim([min(y),max(y)])
        grid()
        
        ax = subplot(gs[3])
#        temp = sum(z,axis=1)
        temp = trapz(z,zaxis, axis=1)          
        plot(temp,y)
        ax.set_xscale(scaleType)
            
            
        ylim([min(y),max(y)])
        grid()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ylabel(r'$Q_x$',size=18, )
        
        ax = subplot(gs[0])  
#        temp = sum(z,axis=0)
        temp = trapz(z,xaxis, axis=0)
        plot(x,temp)
        ax.set_yscale(scaleType)
        
        xlim([min(x),max(x)])
        xlabel(r'$Q_z$',size=18)
        grid()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        
        gs.tight_layout(fig)
        show()
        
        
        #####################################################################
        fig=figure()
        # qz qy Map
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[3,1],
                               height_ratios=[1,3]
                               )
        
        subplot(gs[2])
#        z = sum(data,axis=0)
        z = trapz(data,xaxis, axis=0)
        x = zaxis
        y = yaxis
        contourf(x,y,scaleFunc(z))
        xlabel(r'$Q_z$',size=18)
        ylabel(r'$Q_y$',size=18)
        xlim([min(x),max(x)])
        ylim([min(y),max(y)])
        grid()
        
        ax = subplot(gs[3])
#        temp = sum(z,axis=1)
        temp = trapz(z,zaxis, axis=1)         
        plot(temp,y)
        ax.set_xscale(scaleType)
            
        
        ylim([min(y),max(y)])
        grid()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ylabel(r'$Q_y$',size=18, )
        
        ax = subplot(gs[0])
#        temp = sum(z,axis=0)
        temp = trapz(z,yaxis, axis=0)           
        plot(x,temp)
        ax.set_yscale(scaleType)
            
        
        xlim([min(x),max(x)])
        xlabel(r'$Q_z$',size=18)
        grid()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        
        gs.tight_layout(fig)
        show()
        
    
#    def plotPilatusScanSequenceQ(self, scanSequence, figSize=[]):
#        """Plot the Pilatus data for a given scan number in q-space.
#        
#        Args:
#            scanNum (int)   : Scan number of the spec scan.
#        
#        """
#        
#        allData = list()        
#        
#        for scanNum, parameter in scanSequence:
#            if isinstance(scanNum, (list, tuple, ndarray)):
#                scanNum = scanNum[0]
#            
#            # format the parameter as label text of this plot if no label text 
#            data, xaxis, yaxis, zaxis = self.convPilatusScan2Q(scanNum)
#            
#            temp = [xaxis, sum(sum(data,2),1), yaxis, sum(sum(data,2),0), zaxis, sum(sum(data,0),0)]
#            
#            allData.append(temp)
#        
#        figure(figsize = figSize)
#        for i, thisData in enumerate(allData):
#            subplot(3,1,1)
#            semilogy(thisData[0], thisData[1])
#            xlabel(r'$Q_x$')  
#            ylabel('Intensity')
#            grid(True)
#            
#            subplot(3,1,2)
#            semilogy(thisData[2], thisData[3])
#            xlabel(r'$Q_y$')
#            ylabel('Intensity')
#            grid(True)
#            
#            subplot(3,1,3)
#            semilogy(thisData[4], thisData[5])
#            xlabel(r'$Q_z$')
#            ylabel('Intensity')
#            grid(True)
#            
#        show()
#        
#        return allData


# helper functions
      
      
def edges4grid(grid):
    """Creates a vector of the corresponding edges for a grid vector. """
    binwidth = diff(grid);
    edges    = hstack([grid[0]-binwidth[0]/2, grid[0:-1]+binwidth/2, grid[-1]+binwidth[-1]/2]);
    
    return edges, binwidth


def binData(y,x,X,statistic='mean'):
    """Bin data y(x) on new grid X using a statistic type. """
        
    y = y.flatten(1)
    x = x.flatten(1)
    X = sort(X.flatten(1))
    
    # create bins for the grid
    edges, _ = edges4grid(X);    
    
    if array_equal(x,X): 
        # no binning since the new grid is the same as the old one
        Y = y
        bins = ones_like(Y)        
        n    = ones_like(Y)
    else:    
        # do the binning and get the Y results 
        Y, _ , bins = binned_statistic(x,y,statistic,edges)
        bins = bins.astype(int_)
        
        n = bincount(bins[bins > 0],minlength=len(X)+1)  
        n = n[1:len(X)+1]
    
    
    if array_equal(x,X) and statistic is not 'sum': 
        
        Ystd = zeros_like(Y)
        Xstd = zeros_like(X)
        Yerr = zeros_like(Y)
        Xerr = zeros_like(X)
    else:    
        # calculate the std of X and Y
        if statistic == 'sum':
            Ystd = sqrt(Y)                   
            Yerr = Ystd
        else:
            Ystd, _ , _ = binned_statistic(x,y,std,edges)        
            Yerr        = Ystd/sqrt(n)
        
        Xstd, _ , _ = binned_statistic(x,x,std,edges)        
        Xerr        = Xstd/sqrt(n)
    
    
    #remove NaNs
    Y    = Y[n > 0]
    X    = X[n > 0]
    Yerr = Yerr[n > 0]
    Xerr = Xerr[n > 0]
    Ystd = Ystd[n > 0]
    Xstd = Xstd[n > 0]       
    
    return Y, X, Yerr, Xerr, Ystd, Xstd, edges, bins, n


# xrayutilities child classes

from xrayutilities.io.imagereader import ImageReader
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

        ImageReader.__init__(self, 195, 487, hdrlen=4096, dtype=int32,
                             byte_swap=False, **keyargs)
    
    