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

import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from uncertainties import unumpy
from .helpers import bin_data


class Evaluation(object):
    """Evaluation

    Attributes:
        clist (List[str])       : List of counter names to evaluate.
        cdef (Dict{str:str})    : Dict of predefined counter names and
                                  definitions.
        xcol (str)              : spec counter or motor to plot as x-axis.
        t0 (float)              : approx. time zero for delay scans to
                                  determine the unpumped region of the data
                                  for normalization.
        custom_counters (List[str]): List of custom counters - default is []
        math_keys (List[str])    : List of keywords which are not replaced in
                                  counter names
        statistic_type  (str)    : 'gauss' for normal averaging,
                                  'poisson' for counting statistics
        propagate_errors  (bool) : whether to propagate errors or not

    """

    def __init__(self, source):
        # properties
        self.source = source
        self.clist = []
        self.cdef = {}
        self.xcol = ''
        self.t0 = 0
        self.custom_counters = []
        self.math_keys = ['mean', 'sum', 'diff', 'max', 'min', 'round', 'abs',
                          'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                          'pi', 'exp', 'log', 'log10', 'sqrt']
        self.statistic_type = 'gauss'
        self.propagate_errors = True
        self.apply_data_filter = False
        self.data_filters = ['evaluatable statement']

    def get_clist(self):
        """Return the list of counters to evaluate as list even if they are
        provided as Dict by the user.
        This method is only for backward compatibility to older versions.

        Returns:
            clist (List[str]): List of counter names to evaluate.

        """

        if isinstance(self.clist, dict):
            # the clist property is a dict, so retrun its keys as list
            clist = list(self.clist.keys())
        else:
            clist = list(self.clist)

        return clist

    def get_last_fig_number(self):
        """Return the last figure number of all opened figures for plotting
        data in the same figure during for-loops.

        Returns:
            fig_number (int): Last figure number of all opened figures.

        """

        try:
            # get the number of all opened figures
            fig_number = mpl._pylab_helpers.Gcf.get_active().num
        except Exception:
            # there are no figures open
            fig_number = 1

        return fig_number

    def get_next_fig_number(self):
        """Return the number of the next figure for plotting data in the
        same figure during for-loops.

        Returns:
            next_fig_number (int): Next figure number of all opened figures.

        """

        return self.get_last_fig_number() + 1

    def traverse_counters(self, clist, spec_cols=''):
        """Traverse all counters and replace all predefined counter definitions.
        Returns also the included spec base counters for error propagation.

        Args:
            clist    (list) : Initial counter list.
            spec_cols (list) : Counters in spec file.

        Returns:
            resolved_counters (list): Resolved counters.
            base_counters (list)    : Base counters.

        """

        resolved_counters = []
        base_counters = []

        for col_name in clist:
            col_string, res_base_counters = self.resolve_counter_name(col_name, spec_cols)

            resolved_counters.append(col_string)
            base_counters.extend(res_base_counters)

        # remove duplicates using list(set())
        return resolved_counters, list(set(base_counters))

    def resolve_counter_name(self, col_name, spec_cols=''):
        """Replace all predefined counter definitions in a counter name.
        The function works recursively.

        Args:
            col_name (str) : Initial counter string.

        Returns:l
            col_string (str): Resolved counter string.

        """

        recall = False  # boolean to stop recursive calls

        base_counters = []

        col_string = col_name

        for find_cdef in self.cdef.keys():
            # check for all predefined counters
            search_pattern = r'\b' + find_cdef + r'\b'
            if re.search(search_pattern, col_string) is not None:
                if self.cdef[find_cdef] in spec_cols:
                    # this counter definition is a base spec counter
                    base_counters.append(self.cdef[find_cdef])
                # found a predefined counter
                # recursive call if predefined counter must be resolved again
                recall = True
                # replace the counter definition in the string
                (col_string, _) = re.subn(search_pattern,
                                          '(' + self.cdef[find_cdef] + ')', col_string)

        if recall:
            # do the recursive call
            col_string, rec_base_counters = self.resolve_counter_name(col_string, spec_cols)
            base_counters.extend(rec_base_counters)

        for find_cdef in spec_cols:
            # check for all base spec counters
            search_pattern = r'\b' + find_cdef + r'\b'
            if re.search(search_pattern, col_string) is not None:
                base_counters.append(find_cdef)

        return col_string, base_counters

    def col_string_to_eval_string(self, col_string, array_name='spec_data'):
        """Use regular expressions in order to generate an evaluateable string
        from the counter string in order to append the new counter to the
        spec data.

        Args:
            col_string (str) : Definition of the counter.
            mode (int)      : Flag for different modes

        Returns:
            eval_string (str): Evaluateable string to add the new counter
                              to the spec data.

        """

        # search for alphanumeric counter names in col_string
        iterator = re.finditer(
            '([0-9]*[a-zA-Z\_]+[0-9]*[a-zA-Z]*)*', col_string)
        # these are keys which should not be replaced but evaluated
        math_keys = list(self.math_keys)
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
                    (col_string, _) = re.subn(r'\b'+key.group()+r'\b',
                                              array_name + '[\'' + key.group() + '\']', col_string)

        # add 'np.' prefix to numpy functions/math keys
        for mk in math_keys:
            if mk != '0x0001FFFF':
                (col_string, _) = re.subn(r'\b' + mk + r'\b', 'np.' + mk, col_string)
        return col_string

    def add_custom_counters(self, spec_data, scan_num, base_counters):
        """Add custom counters to the spec data array.
        This is a stub for child classes.

        Args:
            spec_data (ndarray) : Data array from the spec scan.
            scan_num (int)  : Scan number of the spec scan.
            base_counters list(str) : List of the base spec and custom counters
                                     from the clist and xcol.

        Returns:
            spec_data (ndarray): Updated data array from the spec scan.

        """
        return spec_data

    def filter_data(self, data):
        """filter_data

        Args:
            data (TYPE): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """
        res = []
        for data_filter in self.data_filters:
            name, _ = self.resolve_counter_name(data_filter)
            idx = eval(self.col_string_to_eval_string(name, array_name='data'))
            if len(res) == 0:
                res = idx
            else:
                res = np.logical_and(res, idx)

        data_list = []
        dtype_list = []
        for name in data.dtype.names:
            data_list.append(data[name][res])
            dtype_list.append((name,
                               data[name][res].dtype,
                               data[name][res].shape))
        return np.core.records.fromarrays(data_list, dtype=dtype_list)

    def get_scan_data(self, scan_num):
        """

        Args:
            scan_num (TYPE): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """
        data, meta = self.source.get_scan_data(scan_num)
        if self.apply_data_filter:
            data = self.filter_data(data)
        return data

    def get_scan_list_data(self, scan_list):
        """

        Args:
            scan_num (TYPE): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """
        data_list, meta_list = self.source.get_scan_list_data(scan_list)
        if self.apply_data_filter:
            for i, data in enumerate(data_list):
                data_list[i] = self.filter_data(data)
        return data_list

    def avg_N_bin_scans(self, scan_list, xgrid=np.array([]), binning=True):
        """Averages data defined by the counter list, clist, onto an optional
        xgrid. If no xgrid is given the x-axis data of the first scan in the
        list is used instead.

        Args:
            scan_list (List[int])      : List of scan numbers.
            xgrid (Optional[ndarray]) : Grid to bin the data to -
                                        default in empty so use the
                                        x-axis of the first scan.

        Returns:
            avg_data (ndarray) : Averaged data for the scan list.
            std_data (ndarray) : Standart derivation of the data for the scan list.
            err_data (ndarray) : Error of the data for the scan list.
            name (str)        : Name of the data set.

        """

        # generate the name of the data set from the spec file name and scan_list
        name = self.source.name + " #{0:04d}".format(scan_list[0])

        # get the counters which should be evaluated
        clist = self.get_clist()
        if not clist:
            raise Exception('No clist is defined. Do not know what to plot!')
            return
        # process also the xcol as counter in order to allow for newly defined xcols
        if not self.xcol:
            raise Exception('No xcol is defined. Do not know what to plot!')
            return
        if self.xcol not in clist:
            clist.append(self.xcol)

        spec_cols = []
        concat_data = np.array([])

        data_list = self.get_scan_list_data(scan_list)

        for i, (spec_data, scan_num) in enumerate(zip(data_list, scan_list)):
            # traverse the scan list and read data
            # try:
            #     # try to read the motors and data of this scan
            #     spec_data = self.get_scan_data(scan_num)
            # except Exception:
            #     raise
            #     print('Scan #' + scan_num + ' not found, skipping')

            if i == 0 or len(spec_cols) == 0:  # we need to evaluate this only once
                # these are the base spec counters which are present in the data
                # file plus custom counters
                spec_cols = list(
                    set(list(spec_data.dtype.names) + self.custom_counters))

                # resolve the clist and retrieve the resolves counters and the
                # necessary base spec counters for error propagation
                resolved_counters, base_counters = self.traverse_counters(
                    clist, spec_cols)

                # counter names and resolved strings for further calculations
                if self.statistic_type == 'poisson' or self.propagate_errors:
                    # for error propagation we just need the base spec counters
                    # and the xcol
                    col_names = base_counters[:]
                    col_strings = base_counters[:]
                    # add the xcol to both lists
                    col_names.append(self.xcol)
                    col_strings.append(resolved_counters[clist.index(self.xcol)])
                else:
                    # we need to average the resolved counters
                    col_names = clist[:]
                    col_strings = resolved_counters[:]

                # create the dtype of the return array
                dtypes = []
                for col_name in clist:
                    dtypes.append((col_name, '<f8'))

            # add custom counters if defined
            spec_data = self.add_custom_counters(spec_data, scan_num, base_counters)

            data = np.array([])
            # read data into data array
            for col_string, col_name in zip(col_strings, col_names):
                # traverse the counters in the clist and append to data if not
                # already present
                eval_string = self.col_string_to_eval_string(
                    col_string, array_name='spec_data')

                if len(data) == 0:
                    data = np.array(eval(eval_string), dtype=[(col_name, float)])
                elif col_name not in data.dtype.names:
                    data = eval('np.lib.recfunctions.append_fields(data,\''
                                + col_name + '\',data=(' + eval_string
                                + '), dtypes=float, asrecarray=True, usemask=True)')

            if i > 0:
                # this is not the first scan in the list so append the data to
                # the concatenated data array
                concat_data = np.concatenate((concat_data, data), axis=0)
            else:
                concat_data = data

                if len(xgrid) == 0:
                    # if no xgrid is given we use the xData of the first scan instead
                    xgrid = concat_data[self.xcol]

        # remove xcol from clist and resolved counters for further treatment
        del resolved_counters[clist.index(self.xcol)]
        clist.remove(self.xcol)

        try:
            # bin the concatenated data to the xgrid
            # if a custom counter was calculated it might have a different length
            # than the spec counters which will result in an error while binning data
            # from a default spec counter and a custom counter.
            if binning:
                xgrid_reduced, _, _, _, _, _, _, _, _ = bin_data(
                    concat_data[self.xcol], concat_data[self.xcol], xgrid)
            else:
                xgrid_reduced = xgrid
            # create empty arrays for averages, std and errors
            avg_data = np.recarray(np.shape(xgrid_reduced)[0], dtype=dtypes)
            std_data = np.recarray(np.shape(xgrid_reduced)[0], dtype=dtypes)
            err_data = np.recarray(np.shape(xgrid_reduced)[0], dtype=dtypes)

            if self.statistic_type == 'poisson':
                bin_stat = 'sum'
            else:  # gauss
                bin_stat = 'mean'

            if binning:
                if self.statistic_type == 'poisson' or self.propagate_errors:
                    # propagate errors using the uncertainties package

                    # create empty dict for uncertainties data arrays
                    unc_data_err = {}
                    unc_data_std = {}

                    for col in base_counters:
                        # for all cols in the clist bin the data to the xgrid an calculate
                        # the averages, stds and errors
                        y, avg_data[self.xcol], yerr, err_data[self.xcol], ystd, \
                            std_data[self.xcol], _, _, _ = bin_data(concat_data[col],
                                                                    concat_data[self.xcol],
                                                                    xgrid_reduced,
                                                                    statistic=bin_stat)
                        # add spec base counters to uncData arrays
                        unc_data_std[col] = unumpy.uarray(y, ystd)
                        unc_data_err[col] = unumpy.uarray(y, yerr)

                    for col_name, col_string in zip(clist, resolved_counters):
                        eval_string = self.col_string_to_eval_string(
                            col_string, array_name='unc_data_err')
                        temp = eval(eval_string)

                        avg_data[col_name] = unumpy.nominal_values(temp)
                        err_data[col_name] = unumpy.std_devs(temp)

                        eval_string = self.col_string_to_eval_string(
                            col_string, array_name='unc_data_std')
                        temp = eval(eval_string)
                        std_data[col_name] = unumpy.std_devs(temp)
                else:
                    # no error propagation but averaging of individual scans
                    for col in clist:
                        # for all cols in the clist bin the data to the xgrid an calculate
                        # the averages, stds and errors
                        avg_data[col], avg_data[self.xcol], err_data[col], \
                            err_data[self.xcol], std_data[col], std_data[self.xcol], _, _, \
                            _ = bin_data(concat_data[col],
                                         concat_data[self.xcol],
                                         xgrid_reduced,
                                         statistic=bin_stat)
            else:
                for col_name, col_string in zip(clist, resolved_counters):
                    eval_string = self.col_string_to_eval_string(
                        col_string, array_name='spec_data')
                    temp = eval(eval_string)
                    avg_data[col_name] = temp
                    avg_data[self.xcol] = concat_data[self.xcol]
                    err_data[col_name] = 0
                    err_data[self.xcol] = 0
                    std_data[col_name] = 0
                    std_data[self.xcol] = 0

        except Exception:
            raise
            print('xcol and ycol must have the same length --> probably you try plotting a custom'
                  ' counter together with a spec counter')

        return avg_data, std_data, err_data, name

    def plot_scans(self, scan_list, ylims=[], xlims=[], fig_size=[], xgrid=[],
                   yerr='std', xerr='std', norm2one=False, binning=True,
                   label_text='', title_text='', skip_plot=False, grid_on=True,
                   ytext='', xtext='', fmt='-o'):
        """Plot a list of scans from the spec file.
        Various plot parameters are provided.
        The plotted data are returned.

        Args:
            scan_list (List[int])        : List of scan numbers.
            ylims (Optional[ndarray])   : ylim for the plot.
            xlims (Optional[ndarray])   : xlim for the plot.
            fig_size (Optional[ndarray]) : Figure size of the figure.
            xgrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            yerr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xerr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.
            label_text (Optional[str])   : Label of the plot - default is none.
            title_text (Optional[str])   : Title of the figure - default is none.
            skip_plot (Optional[bool])   : Skip plotting, just return data
                                          default is False.
            grid_on (Optional[bool])     : Add grid to plot - default is True.
            ytext (Optional[str])       : y-Label of the plot - defaults is none.
            xtext (Optional[str])       : x-Label of the plot - defaults is none.
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

        # get the averaged data, stds and errors for the scan list and the xgrid
        avg_data, std_data, err_data, name = self.avg_N_bin_scans(
            scan_list, xgrid=xgrid, binning=binning)

        # set the error data
        if xerr == 'std':
            xerr_data = std_data
        elif xerr == 'err':
            xerr_data = err_data
        else:
            xerr_data = np.zeros_like(std_data)

        if yerr == 'std':
            yerr_data = std_data
        elif yerr == 'err':
            yerr_data = err_data
        else:
            yerr_data = np.zeros_like(std_data)

        # set x-data and errors
        x2plot = avg_data[self.xcol]
        xerr2plot = xerr_data[self.xcol]

        # plot all keys in the clist
        clist = self.get_clist()
        for col in clist:
            # traverse the counter list

            # save the counter data and errors in the ordered dictionary
            y2plot[col] = avg_data[col]
            yerr2plot[col] = yerr_data[col]

            if norm2one:
                # normalize the y-data to 1 for t < t0
                # just makes sense for delay scans
                before_zero = y2plot[col][x2plot <= self.t0]
                y2plot[col] = y2plot[col]/np.mean(before_zero)
                yerr2plot[col] = yerr2plot[col]/np.mean(before_zero)

            if len(label_text) == 0:
                # if no label_text is given use the counter name
                lt = col
            else:
                if len(clist) > 1:
                    # for multiple counters add the counter name to the label
                    lt = label_text + ' | ' + col
                else:
                    # for a single counter just use the label_text
                    lt = label_text

            if not skip_plot:
                # plot the errorbar for each counter
                if (xerr == 'none') & (yerr == 'none'):
                    plt.plot(x2plot, y2plot[col], fmt, label=lt)
                else:
                    plt.errorbar(
                        x2plot, y2plot[col], fmt=fmt, label=lt,
                        xerr=xerr2plot, yerr=yerr2plot[col])

        if not skip_plot:
            # add a legend, labels, title and set the limits and grid
            plt.legend(frameon=True, loc=0, numpoints=1)
            plt.xlabel(self.xcol)
            if xlims:
                plt.xlim(xlims)
            if ylims:
                plt.ylim(ylims)
            if len(title_text) > 0:
                plt.title(title_text)
            else:
                plt.title(name)
            if len(xtext) > 0:
                plt.xlabel(xtext)

            if len(ytext) > 0:
                plt.ylabel(ytext)

            if grid_on:
                plt.grid(True)

        return y2plot, x2plot, yerr2plot, xerr2plot, name

    def plot_mesh_scan(self, scan_num, skip_plot=False, grid_on=False, ytext='', xtext='',
                       levels=20, cbar=True):
        """Plot a single mesh scan from the spec file.
        Various plot parameters are provided.
        The plotted data are returned.

        Args:
            scan_num (int)               : Scan number of the spec scan.
            skip_plot (Optional[bool])   : Skip plotting, just return data
                                          default is False.
            grid_on (Optional[bool])     : Add grid to plot - default is False.
            ytext (Optional[str])       : y-Label of the plot - defaults is none.
            xtext (Optional[str])       : x-Label of the plot - defaults is none.
            levels (Optional[int])      : levels of contour plot - defaults is 20.
            cbar (Optional[bool])       : Add colorbar to plot - default is True.

        Returns:
            xx, yy, zz              : x,y,z data which was plotted

        """

        from matplotlib.mlab import griddata
        from matplotlib import gridspec

        # read data from spec file
        try:
            # try to read data of this scan
            spec_data = self.get_scan_data(scan_num)
        except Exception:
            print('Scan #' + int(scan_num) + ' not found, skipping')

        dt = spec_data.dtype
        dt = dt.descr

        xmotor = dt[0][0]
        ymotor = dt[1][0]

        X = spec_data[xmotor]
        Y = spec_data[ymotor]

        xx = np.sort(np.unique(X))
        yy = np.sort(np.unique(Y))

        clist = self.get_clist()

        if len(clist) > 1:
            print('WARNING: Only the first counter of the clist is plotted.')

        Z = spec_data[clist[0]]

        zz = griddata(X, Y, Z, xx, yy, interp='linear')

        if not skip_plot:

            if cbar:
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
            if grid_on:
                plt.grid(True)

            plt.subplot(gs[2+k])

            plt.contourf(xx, yy, zz, levels, cmap='viridis')

            plt.xlabel(xmotor)
            plt.ylabel(ymotor)

            if len(xtext) > 0:
                plt.xlabel(xtext)

            if len(ytext) > 0:
                plt.ylabel(ytext)

            if grid_on:
                plt.grid(True)

            if cbar:
                cb = plt.colorbar(cax=plt.subplot(
                    gs[0]), orientation='horizontal')
                cb.ax.xaxis.set_ticks_position('top')
                cb.ax.xaxis.set_label_position('top')

            ax4 = plt.subplot(gs[3+k])

            plt.plot(np.mean(zz, 1), yy)
            plt.plot(zz[:, np.argmax(np.mean(zz, 0))], yy)
            plt.ylim([np.min(yy), np.max(yy)])

            ax4.yaxis.tick_right()
            if grid_on:
                plt.grid(True)

        return xx, yy, zz

    def plot_scan_sequence(self, scan_sequence, ylims=[], xlims=[], fig_size=[],
                           xgrid=[], yerr='std', xerr='std', norm2one=False,
                           binning=True, sequence_type='', label_text='',
                           title_text='', skip_plot=False, grid_on=True, ytext='',
                           xtext='', fmt='-o'):
        """Plot a list of scans from the spec file.
        Various plot parameters are provided.
        The plotted data are returned.

        Args:
            scan_sequence (ndarray[List[int]
                          , int/str])   : Sequence of scan lists and parameters.
            ylims (Optional[ndarray])   : ylim for the plot.
            xlims (Optional[ndarray])   : xlim for the plot.
            fig_size (Optional[ndarray]) : Figure size of the figure.
            xgrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            yerr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xerr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.
            sequence_type (Optional[str]): Type of the sequence: [fluence, delay,
                                          energy, theta, position, voltage, none,
                                          text] - default is enumeration.
            label_text (Optional[str])   : Label of the plot - default is none.
            title_text (Optional[str])   : Title of the figure - default is none.
            skip_plot (Optional[bool])   : Skip plotting, just return data
                                          default is False.
            grid_on (Optional[bool])     : Add grid to plot - default is True.
            ytext (Optional[str])       : y-Label of the plot - defaults is none.
            xtext (Optional[str])       : x-Label of the plot - defaults is none.
            fmt (Optional[str])         : format string of the plot - defaults is -o.

        Returns:
            sequence_data (OrderedDict) : Dictionary of the averaged scan data.
            parameters (ndarray)       : Parameters of the sequence.
            names (List[str])          : List of names of each data set.
            label_texts (List[str])     : List of labels for each data set.

        """

        # initialize the return data
        sequence_data = collections.OrderedDict()
        names = []
        label_texts = []
        parameters = []

        for i, (scan_list, parameter) in enumerate(scan_sequence):
            # traverse the scan sequence

            parameters.append(parameter)
            # format the parameter as label text of this plot if no label text
            # is given
            if len(label_text) == 0:
                if sequence_type == 'fluence':
                    lt = str.format('{:.2f}  mJ/cm²', parameter)
                elif sequence_type == 'delay':
                    lt = str.format('{:.2f}  ps', parameter)
                elif sequence_type == 'energy':
                    lt = str.format('{:.2f}  eV', parameter)
                elif sequence_type == 'theta':
                    lt = str.format('{:.2f}  deg', parameter)
                elif sequence_type == 'temperature':
                    lt = str.format('{:.2f}  K', parameter)
                elif sequence_type == 'position':
                    lt = str.format('{:.2f}  mm', parameter)
                elif sequence_type == 'voltage':
                    lt = str.format('{:.2f}  V', parameter)
                elif sequence_type == 'current':
                    lt = str.format('{:.2f}  A', parameter)
                elif sequence_type == 'scans':
                    lt = str(scan_list)
                elif sequence_type == 'none':
                    # no parameter for single scans
                    lt = ''
                elif sequence_type == 'text':
                    # parameter is a string
                    lt = parameter
                else:
                    # no sequence type is given --> enumerate
                    lt = str.format('#{}', i+1)
            else:
                lt = label_text[i]

            # get the plot data for the scan list
            y2plot, x2plot, yerr2plot, xerr2plot, name = self.plot_scans(
                scan_list,
                ylims=ylims,
                xlims=xlims,
                fig_size=fig_size,
                xgrid=xgrid,
                yerr=yerr,
                xerr=xerr,
                norm2one=norm2one,
                binning=binning,
                label_text=lt,
                title_text=title_text,
                skip_plot=skip_plot,
                grid_on=grid_on,
                ytext=ytext,
                xtext=xtext,
                fmt=fmt
            )

            if self.xcol not in sequence_data.keys():
                # if the xcol is not in the return data dict - add the key
                sequence_data[self.xcol] = []
                sequence_data[self.xcol + 'Err'] = []

            # add the x-axis data to the return data dict
            sequence_data[self.xcol].append(x2plot)
            sequence_data[self.xcol + 'Err'].append(xerr2plot)

            for counter in y2plot:
                # traverse all counters in the data set
                if counter not in sequence_data.keys():
                    # if the counter is not in the return data dict - add the key
                    sequence_data[counter] = []
                    sequence_data[counter + 'Err'] = []

                # add the counter data to the return data dict
                sequence_data[counter].append(y2plot[counter])
                sequence_data[counter + 'Err'].append(yerr2plot[counter])

            # append names and labels to their lists
            names.append(name)
            label_texts.append(lt)

        return sequence_data, parameters, names, label_texts

    def export_scan_sequence(self, scan_sequence, path, fileName, yerr='std',
                             xerr='std', xgrid=[], norm2one=False, binning=True):
        """Exports spec data for each scan list in the sequence as individual file.

        Args:
            scan_sequence (ndarray[List[int]
                          , int/str])   : Sequence of scan lists and parameters.
            path (str)                  : Path of the file to export to.
            fileName (str)              : Name of the file to export to.
            yerr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xerr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            xgrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.

        """
        # get scan_sequence data without plotting
        sequence_data, parameters, names, label_texts = self.plot_scan_sequence(
            scan_sequence,
            xgrid=xgrid,
            yerr=yerr,
            xerr=xerr,
            norm2one=norm2one,
            binning=binning,
            skip_plot=True)

        for i, label_text in enumerate(label_texts):
            # travserse the sequence

            header = ''
            saveData = []
            for counter in sequence_data:
                # travserse all counters in the data

                # build the file header
                header = header + counter + '\t '
                # build the data matrix
                saveData.append(sequence_data[counter][i])

            # save data with header to text file
            np.savetxt('{:s}/{:s}_{:s}.dat'.format(path, fileName,
                                                   "".join(x for x in label_text if x.isalnum())),
                       np.r_[saveData].T, delimiter='\t', header=header)

    def fit_scans(self, scans, mod, pars, ylims=[], xlims=[], fig_size=[], xgrid=[],
                  yerr='std', xerr='std', norm2one=False, binning=True,
                  sequence_type='text', label_text='', title_text='', ytext='',
                  xtext='', select='', fit_report=0, show_single=False,
                  weights=False, fit_method='leastsq', offset_t0=False,
                  plot_separate=False, grid_on=True, fmt='o'):
        """Fit, plot, and return the data of scans.

            This is just a wrapper for the fit_scan_sequence method
        """
        scan_sequence = [[scans, '']]
        return self.fit_scan_sequence(scan_sequence, mod, pars, ylims, xlims, fig_size,
                                      xgrid, yerr, xerr, norm2one, binning,
                                      'none', label_text, title_text, ytext,
                                      xtext, select, fit_report, show_single,
                                      weights, fit_method, offset_t0, plot_separate,
                                      grid_on, fmt=fmt)

    def fit_scan_sequence(self, scan_sequence, mod, pars, ylims=[], xlims=[], fig_size=[],
                          xgrid=[], yerr='std', xerr='std', norm2one=False,
                          binning=True, sequence_type='', label_text='',
                          title_text='', ytext='', xtext='', select='',
                          fit_report=0, show_single=False, weights=False,
                          fit_method='leastsq', offset_t0=False,
                          plot_separate=False, grid_on=True,
                          last_res_as_par=False, sequence_data=[], fmt='o'):
        """Fit, plot, and return the data of a scan sequence.

        Args:
            scan_sequence (ndarray[List[int]
                          , int/str])   : Sequence of scan lists and parameters.
            mod (Model[lmfit])          : lmfit model for fitting the data.
            pars (Parameters[lmfit])    : lmfit parameters for fitting the data.
            ylims (Optional[ndarray])   : ylim for the plot.
            xlims (Optional[ndarray])   : xlim for the plot.
            fig_size (Optional[ndarray]) : Figure size of the figure.
            xgrid (Optional[ndarray])   : Grid to bin the data to -
                                          default in empty so use the
                                          x-axis of the first scan.
            yerr (Optional[ndarray])    : Type of the errors in y: [err, std, none]
                                          default is 'std'.
            xerr (Optional[ndarray])    : Type of the errors in x: [err, std, none]
                                          default is 'std'.
            norm2one (Optional[bool])   : Norm transient data to 1 for t < t0
                                          default is False.
            sequence_type (Optional[str]): Type of the sequence: [fluence, delay,
                                          energy, theta] - default is fluence.
            label_text (Optional[str])   : Label of the plot - default is none.
            title_text (Optional[str])   : Title of the figure - default is none.
            ytext (Optional[str])       : y-Label of the plot - defaults is none.
            xtext (Optional[str])       : x-Label of the plot - defaults is none.
            select (Optional[str])      : String to evaluate as select statement
                                          for the fit region - default is none
            fit_report (Optional[int])   : Set the fit reporting level:
                                          [0: none, 1: basic, 2: full]
                                          default 0.
            show_single (Optional[bool]) : Plot each fit seperately - default False.
            weights (Optional[bool])    : Use weights for fitting - default False.
            fit_method (Optional[str])   : Method to use for fitting; refer to
                                          lmfit - default is 'leastsq'.
            offset_t0 (Optional[bool])   : Offset time scans by the fitted
                                          t0 parameter - default False.
            plot_separate (Optional[bool]):A single plot for each counter
                                          default False.
            grid_on (Optional[bool])     : Add grid to plot - default is True.
            last_res_as_par (Optional[bool]): Use the last fit result as start
                                           values for next fit - default is False.
            sequence_data (Optional[ndarray]): actual exp. data are externally given.
                                              default is empty
            fmt (Optional[str])         : format string of the plot - defaults is -o.


        Returns:
            res (Dict[ndarray])        : Fit results.
            parameters (ndarray)       : Parameters of the sequence.
            sequence_data (OrderedDict) : Dictionary of the averaged scan data.equenceData

        """

        # get the last open figure number
        main_fig_num = self.get_last_fig_number()

        if not fig_size:
            # use default figure size if none is given
            fig_size = mpl.rcParams['figure.figsize']

        # initialization of returns
        res = {}  # initialize the results dict

        for i, counter in enumerate(self.get_clist()):
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

        if len(sequence_data) > 0:
            # get only the parameters
            _, parameters, names, label_texts = self.plot_scan_sequence(
                scan_sequence,
                ylims=ylims,
                xlims=xlims,
                fig_size=fig_size,
                xgrid=xgrid,
                yerr=yerr,
                xerr=xerr,
                norm2one=norm2one,
                binning=True,
                sequence_type=sequence_type,
                label_text=label_text,
                title_text=title_text,
                skip_plot=True)
        else:
            # get the sequence data and parameters
            sequence_data, parameters, names, label_texts = self.plot_scan_sequence(
                scan_sequence,
                ylims=ylims,
                xlims=xlims,
                fig_size=fig_size,
                xgrid=xgrid,
                yerr=yerr,
                xerr=xerr,
                norm2one=norm2one,
                binning=True,
                sequence_type=sequence_type,
                label_text=label_text,
                title_text=title_text,
                skip_plot=True)

        # this is the number of different counters
        num_sub_plots = len(self.get_clist())

        # fitting and plotting the data
        l_plot = 1  # counter for single plots

        for i, parameter in enumerate(parameters):
            # traverse all parameters of the sequence
            lt = label_texts[i]
            name = names[i]

            x2plot = sequence_data[self.xcol][i]
            xerr2plot = sequence_data[self.xcol + 'Err'][i]

            if fit_report > 0:
                # plot for basics and full fit reporting
                print('')
                print('='*10 + ' Parameter: ' + lt + ' ' + '='*15)

            j = 0  # counter for counters ;)
            k = 1  # counter for subplots
            for counter in sequence_data:
                # traverse all counters in the sequence

                # plot only y counters - next is the coresp. error
                if j >= 2 and j % 2 == 0:

                    # add the counter name to the label for not seperate plots
                    if sequence_type == 'none':
                        _lt = counter
                    else:
                        if plot_separate or num_sub_plots == 1:
                            _lt = lt
                        else:
                            _lt = lt + ' | ' + counter

                    # get the fit models and fit parameters if they are lists/tupels
                    if isinstance(mod, (list, tuple)):
                        _mod = mod[k-1]
                    else:
                        _mod = mod

                    if last_res_as_par and i > 0:
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
                    y2plot = sequence_data[counter][i]
                    yerr2plot = sequence_data[counter + 'Err'][i]

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
                                       weights=1/yerr2plot, method=fit_method,
                                       nan_policy='propagate')
                    else:
                        out = _mod.fit(y2plot, _pars, x=x2plot,
                                       method=fit_method, nan_policy='propagate')

                    if fit_report > 0:
                        # for basic and full fit reporting
                        print('')
                        print('-'*10 + ' ' + counter + ': ' + '-'*15)
                        for key in out.best_values:
                            print('{:>12}:  {:>10.4e} '.format(
                                key, out.best_values[key]))

                    # set the x-offset for delay scans - offset parameter in
                    # the fit must be called 't0'
                    if offset_t0:
                        offsetX = out.best_values['t0']
                    else:
                        offsetX = 0

                    plt.figure(main_fig_num)  # select the main figure

                    if plot_separate:
                        # use subplot for separate plotting
                        plt.subplot((num_sub_plots+num_sub_plots % 2)/2, 2, k)

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
                    if len(title_text) > 0:
                        if isinstance(title_text, (list, tuple)):
                            plt.title(title_text[k-1])
                        else:
                            plt.title(title_text)
                    else:
                        plt.title(name)

                    if len(xtext) > 0:
                        plt.xlabel(xtext)

                    if len(ytext) > 0:
                        if isinstance(ytext, (list, tuple)):
                            plt.ylabel(ytext[k-1])
                        else:
                            plt.ylabel(ytext)

                    if grid_on:
                        plt.grid(True)

                    # show the single fits and residuals
                    if show_single:
                        plt.figure(main_fig_num+l_plot, figsize=fig_size)
                        gs = mpl.gridspec.GridSpec(
                            2, 1, height_ratios=[1, 3], hspace=0.1)
                        ax1 = plt.subplot(gs[0])
                        markerline, stemlines, baseline = plt.stem(
                            x2plot-offsetX, out.residual, markerfmt=' ',
                            use_line_collection=True)
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

                        if len(xtext) > 0:
                            plt.xlabel(xtext)

                        if grid_on:
                            plt.grid(True)

                        if len(title_text) > 0:
                            if isinstance(title_text, (list, tuple)):
                                plt.title(title_text[k-1])
                            else:
                                plt.title(title_text)
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

                        if len(xtext) > 0:
                            plt.xlabel(xtext)

                        if len(ytext) > 0:
                            if isinstance(ytext, (list, tuple)):
                                plt.ylabel(ytext[k-1])
                            else:
                                plt.ylabel(ytext)

                        if grid_on:
                            plt.grid(True)

                        l_plot += 1
                    if fit_report > 1:
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

        plt.figure(main_fig_num)  # set as active figure

        return res, parameters, sequence_data
