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
import os
from xrayutilities.io import SPECFile, SPECScan
from xrayutilities.io.helper import xu_h5open, xu_open
from xrayutilities import config, utilities
import re
from .evalData import spec


class PalSpec(spec):
    
    def updateSpec(self):
        """Update the current spec file if already in memory.
        Otherwise read it and write its content to the hdf5 file.

        """
        try:
            # try if spec file object already exist
            self.specFile.Update()
        except Exception as e:
            # load the spec file from disc
            self.specFile = PalSpecFile(
                self.specFileName, path=self.filePath)
            self.specFile.Update()

        if not os.path.exists(os.path.join(self.hdf5Path, self.h5FileName)) or self.overwriteHDF5:
            # save the new or changed spec file content to the hdf5 file
            # if it does not exist
            self.specFile.Save2HDF5(os.path.join(
                self.hdf5Path, self.h5FileName))
    
    

# define some uesfull regular expressions
SPEC_time_format = re.compile(r"\d\d:\d\d:\d\d")
SPEC_multi_blank = re.compile(r"\s+")
SPEC_multi_blank2 = re.compile(r"\s\s+")
# denotes a numeric value
SPEC_int_value = re.compile(r"[+-]?\d+")
SPEC_num_value = re.compile(
    r"([+-]?\d*\.*\d*[eE]*[+-]*\d+|[+-]?[Ii][Nn][Ff]|[Nn][Aa][Nn])")
SPEC_dataline = re.compile(r"^[+-]*\d.*")

SPEC_scan = re.compile(r"^#RUN")
SPEC_initmoponames = re.compile(r"#MOT")
SPEC_initmopopos = re.compile(r"#VAL")
SPEC_datetime = re.compile(r"^#TIM")
SPEC_exptime = re.compile(r"^#T")
SPEC_nofcols = re.compile(r"^#N")
SPEC_colnames = re.compile(r"^#COL")
SPEC_MCAFormat = re.compile(r"^#@MCA")
SPEC_MCAChannels = re.compile(r"^#@CHANN")
SPEC_headerline = re.compile(r"^#")
SPEC_scanbroken = re.compile(r"#C[a-zA-Z0-9: .]*Scan aborted")
SPEC_scanresumed = re.compile(r"#C[a-zA-Z0-9: .]*Scan resumed")
SPEC_commentline = re.compile(r"#ATT")
SPEC_newheader = re.compile(r"^#E")
SPEC_errorbm20 = re.compile(r"^MI:")
scan_status_flags = ["OK", "NODATA", "ABORTED", "CORRUPTED"]

class PalSpecFile(SPECFile):    
    
    
    def __init__(self, filename, path=''):
        """
        SPECFile init routine

        Parameters
        ----------
        filename :  str
            filename of the spec file
        path :      str, optional
            path to the specfile
        """
        
        self.path = path
        
        self.filename = filename
        
        # we keep that empty as it has to be updated by parse_folders()
        self.full_filename = ''
        

        # list holding scan objects
        self.scan_list = []
        self.fid = None
        self.last_offset = 0
        self.last_scan_nb = 0
        
        self.folder_format = '{:07d}'
        self.file_format = '{:07d}_meta.log'

        # initially parse the file
        self.init_motor_names_fh = []  # this list will hold the names of the
        # motors saved in initial motor positions given in the file header
        self.init_motor_names_sh = []  # this list will hold the names of the
        # motors saved in initial motor positions given in the scan header
        self.init_motor_names = []  # this list will hold the names of the
        # motors saved in initial motor positions from either the file or
        # scan header
        
        self.parse_folders()
        
    def parse_folders(self):
        while True:
            scan_nb = self.last_scan_nb + 1
            print('Look for scan number {:d}'.format(scan_nb))
            
            data_file = os.path.abspath(os.path.join(
                self.path,
                self.folder_format.format(scan_nb), 
                self.file_format.format(scan_nb)))
            
            
            if os.path.exists(data_file):
                self.full_filename = data_file
                print('Parsing Scan #{:d}'.format(scan_nb))
                self.Parse()
                
                # when parsing is done, we reset everything
                self.fid = None
                self.last_offset = 0
                self.init_motor_names_fh = []  # this list will hold the names of the
                # motors saved in initial motor positions given in the file header
                self.init_motor_names_sh = []  # this list will hold the names of the
                # motors saved in initial motor positions given in the scan header
                self.init_motor_names = []  # this list will hold the names of the
                
                # we remeber the last scan number
                self.last_scan_nb = scan_nb
            else:
                print('data file does not exists')
                break
            
    def Parse(self):
        """
        Parses the file from the starting at last_offset and adding found scans
        to the scan list.
        """
        import numpy
        with xu_open(self.full_filename) as self.fid:
            # move to the last read position in the file
            self.fid.seek(self.last_offset, 0)
            scan_started = False
            scan_has_mca = False
            # list with the motors from whome the initial
            # position is stored.
            init_motor_values = []

            if config.VERBOSITY >= config.DEBUG:
                print('XU.io.SPECFile: start parsing')

            for line in self.fid:
                linelength = len(line)
                line = line.decode('ascii', 'ignore')
                if config.VERBOSITY >= config.DEBUG:
                    print('parsing line: %s' % line)

                # remove trailing and leading blanks from the read line
                line = line.strip()

                # fill the list with the initial motor names in the header
                if SPEC_newheader.match(line):
                    self.init_motor_names_fh = []

                elif SPEC_initmoponames.match(line) and not scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found initial motor "
                              "names in file header")
                    line = SPEC_initmoponames.sub("", line)
                    line = line.strip()
                    self.init_motor_names_fh = self.init_motor_names_fh + \
                        SPEC_multi_blank2.split(line)

                # if the line marks the beginning of a new scan
                elif SPEC_scan.match(line) and not scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found scan")
                    line_list = SPEC_multi_blank.split(line)
                    scannr = int(line_list[1])
                    scancmd = "".join(" " + x + " " for x in line_list[2:])
                    scan_started = True
                    scan_has_mca = False
                    scan_header_offset = self.last_offset
                    scan_status = "OK"
                    # define some necessary variables which could be missing in
                    # the scan header
                    itime = numpy.nan
                    time = ''
                    if config.VERBOSITY >= config.INFO_ALL:
                        print("XU.io.SPECFile.Parse: processing scan nr. %d "
                              "..." % scannr)
                    # set the init_motor_names to the ones found in
                    # the file header
                    self.init_motor_names_sh = []
                    self.init_motor_names = self.init_motor_names_fh

                    # if the line contains the date and time information
                elif SPEC_datetime.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found date and time")
                    # fetch the time from the line data
                    time = SPEC_time_format.findall(line)[0]
                    line = SPEC_time_format.sub("", line)
                    line = SPEC_datetime.sub("", line)
                    date = SPEC_multi_blank.sub(" ", line).strip()

                # if the line contains the integration time
                elif SPEC_exptime.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found exposure time")
                    itime = float(SPEC_num_value.findall(line)[0])
                # read the initial motor names in the scan header if present
                elif SPEC_initmoponames.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found initial motor "
                              "names in scan header")
                    line = SPEC_initmoponames.sub("", line)
                    line = line.strip()
                    self.init_motor_names_sh = self.init_motor_names_sh + \
                        SPEC_multi_blank2.split(line)
                    self.init_motor_names = self.init_motor_names_sh
                # read the initial motor positions
                elif SPEC_initmopopos.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found initial motor "
                              "positions")
                    line = SPEC_initmopopos.sub("", line)
                    line = line.strip()
                    line_list = SPEC_multi_blank.split(line)
                    # sometimes initial motor position are simply empty and
                    # this should not lead to an error
                    try:
                        for value in line_list:
                            init_motor_values.append(float(value))
                    except ValueError:
                        pass

                # if the line contains the column names
                elif SPEC_colnames.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found column names")
                    line = SPEC_colnames.sub("", line)
                    line = line.strip()
                    col_names = SPEC_multi_blank.split(line)
                    nofcols = len(col_names)
                    # this is a fix in the case that blanks are allowed in
                    # motor and detector names (only a single balanks is
                    # supported meanwhile)
                    if len(col_names) > nofcols:
                        col_names = SPEC_multi_blank2.split(line)

                elif SPEC_MCAFormat.match(line) and scan_started:
                    mca_col_number = int(SPEC_num_value.findall(
                                         line)[0])
                    scan_has_mca = True

                elif SPEC_MCAChannels.match(line) and scan_started:
                    line_list = SPEC_num_value.findall(line)
                    mca_channels = int(line_list[0])
                    mca_start = int(line_list[1])
                    mca_stop = int(line_list[2])

                elif (SPEC_scanbroken.findall(line) != [] and
                      scan_started):
                    # this is the case when a scan is broken and no data has
                    # been written, but nevertheless a comment is in the file
                    # that tells us that the scan was aborted
                    scan_data_offset = self.last_offset
                    s = SPECScan("scan_%i" % (scannr), scannr, scancmd,
                                 date, time, itime, col_names,
                                 scan_header_offset, scan_data_offset,
                                 self.full_filename, self.init_motor_names,
                                 init_motor_values, "NODATA")

                    self.scan_list.append(s)

                    # reset control flags
                    scan_started = False
                    scan_has_mca = False
                    # reset initial motor positions flag
                    init_motor_values = []

                elif SPEC_dataline.match(line) and scan_started:
                    # this is now the real end of the header block. at this
                    # point we know that there is enough information about the
                    # scan

                    # save the data offset
                    scan_data_offset = self.last_offset

                    # create an SPECFile scan object and add it to the scan
                    # list the name of the group consists of the prefix scan
                    # and the number of the scan in the file - this shoule make
                    # it easier to find scans in the HDF5 file.
                    s = SPECScan("scan_%i" % (scannr), scannr, scancmd, date,
                                 time, itime, col_names, scan_header_offset,
                                 scan_data_offset, self.full_filename,
                                 self.init_motor_names, init_motor_values,
                                 scan_status)
                    if scan_has_mca:
                        s.SetMCAParams(mca_col_number, mca_channels, mca_start,
                                       mca_stop)

                    self.scan_list.append(s)

                    # reset control flags
                    scan_started = False
                    scan_has_mca = False
                    # reset initial motor positions flag
                    init_motor_values = []

                elif SPEC_scan.match(line) and scan_started:
                    # this should only be the case when there are two
                    # consecutive file headers in the data file without any
                    # data or abort notice of the first scan; first store
                    # current scan as aborted then start new scan parsing
                    s = SPECScan("scan_%i" % (scannr), scannr, scancmd,
                                 date, time, itime, col_names,
                                 scan_header_offset, None,
                                 self.full_filename, self.init_motor_names,
                                 init_motor_values, "NODATA")
                    self.scan_list.append(s)

                    # reset control flags
                    scan_started = False
                    scan_has_mca = False
                    # reset initial motor positions flag
                    init_motor_values = []

                    # start parsing of new scan
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found scan "
                              "(after aborted scan)")
                    line_list = SPEC_multi_blank.split(line)
                    scannr = int(line_list[1])
                    scancmd = "".join(" " + x + " " for x in line_list[2:])
                    scan_started = True
                    scan_has_mca = False
                    scan_header_offset = self.last_offset
                    scan_status = "OK"
                    self.init_motor_names_sh = []
                    self.init_motor_names = self.init_motor_names_fh

                # else:
                #     print('cannot read that shit: {:s}'.format(line))
                # store the position of the file pointer
                self.last_offset += linelength

            # if reading of the file is finished store the data offset of the
            # last scan as the last offset for the next parsing run of the file
            self.last_offset = self.scan_list[-1].doffset

    