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

from . import config
import logging

import numpy as np
import collections
import matplotlib.pyplot as plt
from uncertainties import unumpy
from tabulate import tabulate
from .helpers import bin_data, traverse_counters, resolve_counter_name, col_string_to_eval_string

__all__ = ['Evaluation']

__docformat__ = 'restructuredtext'


class Evaluation(object):
    """Evaluation

    Main class for evaluating data.
    The raw data is accessed via a ``Source`` object.
    The evaluation allows to bin data, calculate errors and propagate them.
    There is also an interface to ``lmfit`` for easy batch-fitting.

    Args:
        source (Source): raw data source.

    Attributes:
        log (logging.logger): logger instance from logging.
        clist (list[str]): list of counter names to evaluate.
        cdef (dict{str:str}): dict of predefined counter names and
            definitions.
        xcol (str): counter or motor for x-axis.
        t0 (float): approx. time zero for delay scans to determine the
            unpumped region of the data for normalization.
        custom_counters (list[str]): list of custom counters - default is []
        math_keys (list[str]): list of keywords which are evaluated as numpy
            functions.
        ignore_keys (list[str]): list of keywords which should not be
            evaluated.
        statistic_type (str): 'gauss' for normal averaging, 'poisson' for
            counting statistics.
        propagate_errors (bool): propagate errors for dependent counters.

    """

    def __init__(self, source):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(config.LOG_LEVEL)
        self.source = source
        self.clist = []
        self.cdef = {}
        self.xcol = ''
        self.t0 = 0
        self.custom_counters = []
        self.math_keys = ['mean', 'sum', 'diff', 'max', 'min', 'round', 'abs',
                          'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                          'pi', 'exp', 'log', 'log10', 'sqrt', 'sign']
        self.ignore_keys = []
        self.statistic_type = 'gauss'
        self.propagate_errors = True
        self.apply_data_filter = False
        self.data_filters = ['evaluatable statement']

    def add_custom_counters(self, source_data, scan_num, source_counters):
        """Add custom counters to the source data array.
        This is a stub for child classes.

        Args:
            source_data (ndarray): data array from the source scan.
            scan_num (int): scan number of the source scan.
            source_counters list(str): List of the source counters and custom
                counters from the clist and xcol.

        Returns:
            source_data (ndarray): Updated data array from the source scan.

        """
        return source_data

    def filter_data(self, data):
        """filter_data

        Apply data filter to data.

        Args:
            data (ndarray): input data.

        Returns:
            ndarray: output data.

        """
        res = []
        for data_filter in self.data_filters:
            name, _ = resolve_counter_name(self.cdef, data_filter)
            idx = eval(col_string_to_eval_string(
                name, self.math_keys, self.ignore_keys, array_name='data'))
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
        """get_scan_data

        Get the data for a scan from the source and applying data filters if
        enabled.

        Args:
            scan_num (uint): scan number.

        Returns:
            ndarray: scan data array.

        """
        data, meta = self.source.get_scan_data(scan_num)
        if self.apply_data_filter:
            data = self.filter_data(data)
        return data

    def get_scan_list_data(self, scan_list):
        """get_scan_list_data

        Return a list of data sets for a given list of scan numbers.

        Args:
            scan_list (list[uint]): list of scan numbers.

        Returns:
            list[ndarray]: list of scan data arrays.

        """
        data_list, meta_list = self.source.get_scan_list_data(scan_list)
        if self.apply_data_filter:
            for i, data in enumerate(data_list):
                data_list[i] = self.filter_data(data)
        return data_list

    def avg_N_bin_scans(self, scan_list, xgrid=np.array([]), binning=True):
        """avg_N_bin_scans

        Averages data defined by the counter list, clist, onto an optional
        xgrid. If no xgrid is given the x-axis data of the first scan in the
        list is used instead.

        Args:
            scan_list (list[int]): list of scan numbers.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            binning (bool, optional): enable binning of data - default is True

        Returns:
            (tuple):
            - *avg_data (ndarray)* - averaged data for the scan list.
            - *std_data (ndarray)* - standard derivation of the data for the scan list.
            - *err_data (ndarray)* - error of the data for the scan list.
            - *name (str)* - name of the data set.

        """

        # generate the name of the data set from the source file name and scan_list
        name = self.source.name + " #{0:04d}".format(scan_list[0])

        # get the counters which should be evaluated
        if not self.clist:
            raise Exception('No clist is defined. Do not know what to plot!')
            return
        # process also the xcol as counter in order to allow for newly defined xcols
        if not self.xcol:
            raise Exception('No xcol is defined. Do not know what to plot!')
            return
        if self.xcol not in self.clist:
            self.clist.append(self.xcol)

        source_cols = []
        concat_data = np.array([])

        data_list = self.get_scan_list_data(scan_list)

        for i, (source_data, scan_num) in enumerate(zip(data_list, scan_list)):
            if i == 0 or len(source_cols) == 0:  # we need to evaluate this only once
                # these are the base source counters which are present in the data
                # file plus custom counters
                source_cols = list(
                    set(list(source_data.dtype.names) + self.custom_counters))

                # resolve the clist and retrieve the resolves counters and the
                # necessary base source counters for error propagation
                resolved_counters, source_counters = traverse_counters(self.clist,
                                                                       self.cdef,
                                                                       source_cols)

                # counter names and resolved strings for further calculations
                if self.statistic_type == 'poisson' or self.propagate_errors:
                    # for error propagation we just need the base source counters
                    # and the xcol
                    col_names = source_counters[:]
                    col_strings = source_counters[:]
                    # add the xcol to both lists
                    col_names.append(self.xcol)
                    col_strings.append(resolved_counters[self.clist.index(self.xcol)])
                else:
                    # we need to average the resolved counters
                    col_names = self.clist[:]
                    col_strings = resolved_counters[:]

                # create the dtype of the return array
                dtypes = []
                for col_name in self.clist:
                    dtypes.append((col_name, '<f8'))

            # add custom counters if defined
            source_data = self.add_custom_counters(source_data, scan_num, source_counters)

            data = np.array([])
            # read data into data array
            for col_string, col_name in zip(col_strings, col_names):
                # traverse the counters in the clist and append to data if not
                # already present
                eval_string = col_string_to_eval_string(
                    col_string, self.math_keys, self.ignore_keys, array_name='source_data')

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
        del resolved_counters[self.clist.index(self.xcol)]
        self.clist.remove(self.xcol)

        try:
            # bin the concatenated data to the xgrid
            # if a custom counter was calculated it might have a different length
            # than the source counters which will result in an error while binning data
            # from a default source counter and a custom counter.
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

                    for col in source_counters:
                        # for all cols in the clist bin the data to the xgrid an calculate
                        # the averages, stds and errors
                        y, avg_data[self.xcol], yerr, err_data[self.xcol], ystd, \
                            std_data[self.xcol], _, _, _ = bin_data(concat_data[col],
                                                                    concat_data[self.xcol],
                                                                    xgrid_reduced,
                                                                    statistic=bin_stat)
                        # add source base counters to uncData arrays
                        # the uncertainty package cannot handle masked arrays
                        # e.g. for divisions in the clist
                        # --> convert all base counter results to np.array()
                        unc_data_std[col] = unumpy.uarray(np.array(y),
                                                          np.array(ystd))
                        unc_data_err[col] = unumpy.uarray(np.array(y),
                                                          np.array(yerr))

                    for col_name, col_string in zip(self.clist, resolved_counters):
                        eval_string = col_string_to_eval_string(
                            col_string, self.math_keys, self.ignore_keys, array_name='unc_data_err'
                            )
                        temp = eval(eval_string)

                        avg_data[col_name] = unumpy.nominal_values(temp)
                        err_data[col_name] = unumpy.std_devs(temp)

                        eval_string = col_string_to_eval_string(
                            col_string, self.math_keys, self.ignore_keys, array_name='unc_data_std'
                            )
                        temp = eval(eval_string)
                        std_data[col_name] = unumpy.std_devs(temp)
                else:
                    # no error propagation but averaging of individual scans
                    for col in self.clist:
                        # for all cols in the clist bin the data to the xgrid an calculate
                        # the averages, stds and errors
                        avg_data[col], avg_data[self.xcol], err_data[col], \
                            err_data[self.xcol], std_data[col], std_data[self.xcol], _, _, \
                            _ = bin_data(concat_data[col],
                                         concat_data[self.xcol],
                                         xgrid_reduced,
                                         statistic=bin_stat)
            else:
                # no binning
                for col_name, col_string in zip(self.clist, resolved_counters):
                    eval_string = col_string_to_eval_string(
                        col_string, self.math_keys, self.ignore_keys, array_name='source_data')
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
                  ' counter together with a source counter')

        return avg_data, std_data, err_data, name

    def eval_scans(self, scan_list, xgrid=[], yerr='std', xerr='std', norm2one=False,
                   binning=True):
        """eval_scans

        Evaluate a list of scans for a given set of external parameters.

        Args:
            scan_list (list[int]): list of scan numbers.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none]
                default is 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none]
                default is 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0
                default is False.
            binning (bool, optional): enable binning of data - default is True
        Returns:
             (tuple):
            - *y2plot (OrderedDict)* - evaluated y-data.
            - *x2plot (ndarray)* -evaluated x-data.
            - *yerr2plot (OrderedDict)* - evaluated y-error.
            - *xerr2plot (ndarray)* - evaluated x-error.
            - *name (str)* - name of the data set.

        """
        # initialize the y-data as ordered dict in order to allow for multiple
        # counters at the same time
        y2plot = collections.OrderedDict()
        yerr2plot = collections.OrderedDict()

        # get the averaged data, stds and errors for the scan list and the xgrid
        avg_data, std_data, err_data, name = self.avg_N_bin_scans(
            scan_list, xgrid=xgrid, binning=binning)

        # set x-data and errors
        x2plot = avg_data[self.xcol]
        # set the error data
        if xerr == 'std':
            xerr2plot = std_data[self.xcol]
        elif xerr == 'err':
            xerr2plot = err_data[self.xcol]
        else:
            xerr2plot = None

        # plot all keys in the clist
        for col in self.clist:
            # traverse the counter list

            # save the counter data and errors in the ordered dictionary
            y2plot[col] = avg_data[col]
            if yerr == 'std':
                yerr2plot[col] = std_data[col]
            elif yerr == 'err':
                yerr2plot[col] = err_data[col]
            else:
                yerr2plot[col] = None

            if norm2one:
                # normalize the y-data to 1 for t < t0
                # e.g. for delay scans
                before_zero = y2plot[col][x2plot <= self.t0]
                y2plot[col] = y2plot[col]/np.mean(before_zero)
                if yerr2plot[col] is not None:
                    yerr2plot[col] = yerr2plot[col]/np.mean(before_zero)

        return y2plot, x2plot, yerr2plot, xerr2plot, name

    def eval_scan_sequence(self, scan_sequence, xgrid=[], yerr='std', xerr='std', norm2one=False,
                           binning=True):
        """eval_scan_sequence

        Evaluate a sequence of scans for a given set of external parameters.

        Args:
            scan_sequence (list[
                list/tuple[list[int],
                int/str]]): sequence of scan lists and parameters.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none]
                default is 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none]
                default is 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0
                default is False.
            binning (bool, optional): enable binning of data - default is True
        Returns:
             (tuple):
            - *sequence_data (OrderedDict)* - dictionary of the averaged scan data.
            - *parameters (list[str, float])* - parameters of the sequence.
            - *names (list[str])* - list of names of each data set.

        """
        # initialize the return data
        sequence_data = collections.OrderedDict()
        names = []
        parameters = []

        for i, (scan_list, parameter) in enumerate(scan_sequence):
            # traverse the scan sequence

            parameters.append(parameter)
            # get the data for the current scan list
            y2plot, x2plot, yerr2plot, xerr2plot, name = self.eval_scans(
                scan_list, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one,
                binning=binning,
            )
            # create a list of all counters from the scan and append the xcol
            sequence_counters = list(y2plot.keys())
            sequence_counters.append(self.xcol)
            for counter in sequence_counters:
                # traverse all counters in the data set
                if counter not in sequence_data.keys():
                    # if the counter is not in the return data dict - add the key
                    sequence_data[counter] = []
                    sequence_data[counter + 'Err'] = []

                # add the counter data to the return data dict
                try:
                    sequence_data[counter].append(y2plot[counter])
                    sequence_data[counter + 'Err'].append(yerr2plot[counter])
                except KeyError:
                    sequence_data[counter].append(x2plot)
                    sequence_data[counter + 'Err'].append(xerr2plot)

            names.append(name)

        return sequence_data, parameters, names

    def _plot_scans(self, y2plot, x2plot, yerr2plot, xerr2plot, name, label_text='', fmt='-o',
                    plot_separate=False, **kwargs):
        """_plot_scans

        Internal plotting function for a given data set.

        Args:
            y2plot (OrderedDict): y-data to plot.
            x2plot (ndarray): x-data to plot.
            yerr2plot (OrderedDict): y-error to plot.
            xerr2plot (ndarray): x-error which was plot.
            name (str): name of the data set.
            label_text (str, optional): label of the plot - default is none.
            fmt (str, optional): format string of the plot - defaults is -o.
            plot_separate (bool, optional): use separate subplots for different
                counters. Defaults to False.

        Returns:
            plots (list[PlotObjects]): list of matplotlib plot objects.

        """
        plots = []
        # plot all keys in the clist
        for i, counter in enumerate(self.clist):
            # iterate the counter list

            if plot_separate:
                # use subplot for separate plotting
                plt.subplot(1, len(self.clist), i+1)

            if len(label_text) == 0:
                # if no label_text is given use the counter name
                lt = counter
            else:
                if len(self.clist) > 1:
                    # for multiple counters add the counter name to the label
                    lt = label_text + ' | ' + counter
                else:
                    # for a single counter just use the label_text
                    lt = label_text

            # plot the errorbar for each counter
            if (xerr2plot is None) & (yerr2plot is None):
                plot = plt.plot(x2plot, y2plot[counter], fmt, label=lt, **kwargs)
            else:
                plot = plt.errorbar(x2plot, y2plot[counter], fmt=fmt, label=lt, xerr=xerr2plot,
                                    yerr=yerr2plot[counter], **kwargs)
            plots.append(plot)

            plt.xlabel(self.xcol)
            plt.title(name)

        return plots

    def plot_scans(self, scan_list, xgrid=np.array([]), yerr='std', xerr='std', norm2one=False,
                   binning=True, label_text='', fmt='-o', plot_separate=False, **kwargs):
        """plot_scans

        Plot a list of scans from the source file.

        Args:
            scan_list (list[int]): list of scan numbers.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none]
                default is 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none]
                default is 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0
                default is False.
            binning (bool, optional): enable binning of data - default is True
            label_text (str, optional): Label of the plot - default is none.
            fmt (str, optional): format string of the plot - defaults is -o.
            plot_separate (bool, optional): use separate subplots for different
                counters. Defaults to False.

        Returns:
             (tuple):
            - *y2plot (OrderedDict)* - y-data which was plotted.
            - *x2plot (ndarray)* - x-data which was plotted.
            - *yerr2plot (OrderedDict)* - y-error which was plotted.
            - *xerr2plot (ndarray)* - x-error which was plotted.
            - *name (str)* - Name of the data set.

        """

        y2plot, x2plot, yerr2plot, xerr2plot, name = \
            self.eval_scans(scan_list, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one,
                            binning=binning)

        _ = self._plot_scans(y2plot, x2plot, yerr2plot, xerr2plot, name, label_text=label_text,
                             fmt=fmt, plot_separate=plot_separate, **kwargs)
        plt.legend(frameon=True, loc=0, numpoints=1)

        return y2plot, x2plot, yerr2plot, xerr2plot, name

    def plot_scan_sequence(self, scan_sequence, xgrid=np.array([]), yerr='std', xerr='std',
                           norm2one=False, binning=True, label_format='', fmt='-o',
                           plot_separate=False, show_single=False, **kwargs):
        """plot_scan_sequence

        Plot a scan sequence from the source file.

        Args:
            scan_sequence (list[
                list/tuple[list[int],
                int/str]]): sequence of scan lists and parameters.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none]
                default is 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none]
                default is 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0
                default is False.
            binning (bool, optional): enable binning of data - default is True
            label_format (str, optional): format string for label text - default
                is empty.
            fmt (str, optional): format string of the plot - defaults is -o.
            plot_separate (bool, optional): use separate subplots for different
                counters. Defaults to False.
            show_single (bool, optional): show single figure for each sequence
                element.

        Returns:
             (tuple):
            - *sequence_data (OrderedDict)* - dictionary of the averaged scan data.
            - *parameters (list[str, float])* - parameters of the sequence.
            - *names (list[str])* - list of names of each data set.
            - *label_texts (list[str])* - list of labels for each data set.

        """

        sequence_data, parameters, names = self.eval_scan_sequence(
            scan_sequence, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one, binning=binning)

        label_texts = []
        for i, (scan_list, parameter) in enumerate(scan_sequence):
            # iterate the scan sequence
            if show_single:
                plt.figure()
            lt = '#{:d}'.format(i+1)
            if len(label_format) > 0:
                try:
                    lt = label_format.format(parameter)
                except ValueError:
                    self.log.warning('Could not apply \'label_format\' to parameter!')

            label_texts.append(lt)
            # extract clist und xcol from sequence_data
            _ = self._plot_scans({c: sequence_data[c][i] for c in self.clist},
                                 sequence_data[self.xcol][i],
                                 {c: sequence_data[c + 'Err'][i] for c in self.clist},
                                 sequence_data[self.xcol + 'Err'][i],
                                 names[i],
                                 label_text=lt,
                                 fmt=fmt,
                                 plot_separate=plot_separate,
                                 **kwargs)
            if show_single:
                plt.legend(frameon=True, loc=0, numpoints=1)
                plt.show()
            else:
                plt.legend(bbox_to_anchor=(0., 1.08, 1, .102), frameon=True,
                           loc=3, numpoints=1, ncol=3, mode="expand",
                           borderaxespad=0.)

        return sequence_data, parameters, names, label_texts

    def _fit_scans(self, y2plot, x2plot, yerr2plot, xerr2plot, mod, pars, select='', weights=False,
                   fit_method='leastsq', nan_policy='propagate'):
        """_fit_scans

        Internal method to fit a given data set.

        Args:
            y2plot (OrderedDict): y-data to plot.
            x2plot (ndarray): x-data to plot.
            yerr2plot (OrderedDict): y-error to plot.
            xerr2plot (ndarray): x-error which was plot.
            mod (lmfit.Model): fit model.
            pars (lmfit.parameters): fit parameters.
            select (str, optional): evaluatable string to select x-range.
                Defaults to empty string.
            weights (bool, optional): enable weighting by inverse of errors.
                Defaults to False.
            fit_method (str, optional): lmfit's fit method. Defaults to 'leastsq'.
            nan_policy (str, optional): lmfit's NaN policy. Defaults to 'propagate'.

        Returns:
            (tuple):
            - *res (dict)* - fit result dictionary.
            - *report (list[dict, report])* - list of lmfit's best value
                dictionary and fit report object
        """
        res = {}  # initialize the results dict
        report = []
        report_1 = []
        report_2 = {}

        for counter in y2plot:
            res[counter] = {}
            # get the fit models and fit parameters if they are lists/tuples

            # evaluate the select statement
            if select == '':
                # select all
                sel = np.ones_like(y2plot[counter], dtype=bool)
            else:
                sel = eval(select)

            # execute the select statement
            _y2plot = y2plot[counter][sel]
            _x2plot = x2plot[sel]
            _yerr2plot = yerr2plot[counter][sel]
            _xerr2plot = xerr2plot[sel]

            # remove nans
            _y2plot = _y2plot[~np.isnan(_y2plot)]
            _x2plot = _x2plot[~np.isnan(_y2plot)]
            _yerr2plot = _yerr2plot[~np.isnan(_y2plot)]
            _xerr2plot = _xerr2plot[~np.isnan(_y2plot)]

            # do the fitting with or without weighting the data
            if weights:
                out = mod.fit(_y2plot, pars, x=_x2plot, weights=1/_yerr2plot, method=fit_method,
                              nan_policy=nan_policy)
            else:
                out = mod.fit(_y2plot, pars, x=_x2plot, method=fit_method, nan_policy=nan_policy)

            best_values = list(out.best_values.values())
            best_values.insert(0, counter)
            report_1.append(best_values)

            report_2[counter] = out.fit_report()
            # add the fit results to the returns
            for pname, par in pars.items():
                res[counter][pname] = out.best_values[pname]
                res[counter][pname + 'Err'] = out.params[pname].stderr

            res[counter]['chisqr'] = out.chisqr
            res[counter]['redchi'] = out.redchi
            res[counter]['CoM'] = sum(_y2plot*_x2plot)/sum(_y2plot)
            res[counter]['int'] = np.trapz(_y2plot, x=_x2plot)
            res[counter]['fit'] = out

        report = [report_1, report_2]

        return res, report

    def _plot_fit_scans(self, y2plot, x2plot, yerr2plot, xerr2plot, name, res, offset_t0=False,
                        label_text='', fmt='o', plot_separate=False):
        """_plot_fit_scans

        Internal function plot scans and fits of a given data set and fit results.

        Args:
            y2plot (OrderedDict): y-data to plot.
            x2plot (ndarray): x-data to plot.
            yerr2plot (OrderedDict): y-error to plot.
            xerr2plot (ndarray): x-error which was plot.
            name (str): name of the data set.
            res (dict): fit results.
            offset_t0 (bool, optional): offset plot by t0 parameter of the fit
                results. Defaults to False.
            label_text (str, optional): label of the plot - default is none.
            fmt (str, optional): format string of the plot - defaults is -o.
            plot_separate (bool, optional): use separate subplots for different
                counters. Defaults to False.

        """
        plots = self._plot_scans(y2plot, x2plot, yerr2plot, xerr2plot, name, label_text=label_text,
                                 fmt=fmt, alpha=0.25, plot_separate=plot_separate)

        # set the x-offset for delay scans - offset parameter in
        # the fit must be called 't0'
        offsetX = 0
        if offset_t0:
            try:
                offsetX = res['t0']
            except KeyError:
                self.log.warning('No parameter \'t0\' present in model!')
        else:
            offsetX = 0

        for i, counter in enumerate(y2plot):
            if plot_separate:
                # use subplot for separate plotting
                plt.subplot(1, len(self.clist), i+1)
            # plot the fit and the data as errorbars
            x2plotFit = np.linspace(
                np.min(x2plot), np.max(x2plot), 10000)
            plt.plot(x2plotFit-offsetX, res[counter]['fit'].eval(x=x2plotFit), '-', lw=2, alpha=1,
                     color=plots[i][0].get_color())

    def fit_scans(self, scan_list, mod, pars, xgrid=[], yerr='std', xerr='std', norm2one=False,
                  binning=True, label_text='', fmt='o', select='', fit_report=0, weights=False,
                  fit_method='leastsq', nan_policy='propagate', offset_t0=False,
                  plot_separate=False):
        """fit_scans

        Evaluate, fit, and plot the results of a given list of scans from the
        source file.

        Args:
            scan_list (list[int]): list of scan numbers.
            mod (lmfit.Model): fit model.
            pars (lmfit.parameters): fit parameters.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none]
                default is 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none]
                default is 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0
                default is False.
            binning (bool, optional): enable binning of data - default is True
            label_text (str, optional): label of the plot - default is none.
            fmt (str, optional): format string of the plot - defaults is -o.
            select (str, optional): evaluatable string to select x-range.
                Defaults to empty string.
            fit_report (uint, optional): Default is 0 - no report. 1 - fit
                results. 2 - fit results and correlations.
            weights (bool, optional): enable weighting by inverse of errors.
                Defaults to False.
            fit_method (str, optional): lmfit's fit method. Defaults to 'leastsq'.
            nan_policy (str, optional): lmfit's NaN policy. Defaults to 'propagate'.
            offset_t0 (bool, optional): offset plot by t0 parameter of the fit
                results. Defaults to False.
            plot_separate (bool, optional): use separate subplots for different
                counters. Defaults to False.

        Returns:
             (tuple):
            - *res (dict)* - fit result dictionary.
            - *y2plot (OrderedDict)* - y-data which was fitted and plotted.
            - *x2plot (ndarray)* - x-data which was fitted and plotted.
            - *yerr2plot (OrderedDict)* - y-error which was fitted and plotted.
            - *xerr2plot (ndarray)* - x-error which was fitted and plotted.
            - *name (str)* - Name of the data set.

        """
        # get the data for the scan list
        y2plot, x2plot, yerr2plot, xerr2plot, name = \
            self.eval_scans(scan_list, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one,
                            binning=binning)

        # fit the model and parameters to the data
        res, report = self._fit_scans(y2plot, x2plot, yerr2plot, xerr2plot, mod, pars, select,
                                      weights, fit_method=fit_method, nan_policy=nan_policy)

        # plot the data and fit
        self._plot_fit_scans(y2plot, x2plot, yerr2plot, xerr2plot, name, res, offset_t0=offset_t0,
                             label_text=label_text, fmt=fmt, plot_separate=plot_separate)

        plt.legend(frameon=True, loc=0, numpoints=1)

        # print the fit report
        if fit_report > 0:
            print(tabulate(report[0], headers=['counter', *mod.param_names],
                           tablefmt="fancy_grid"))
        if fit_report > 1:
            for counter in y2plot:
                head_len = int(len(counter)/2)
                if np.mod(len(counter), 2) != 0:
                    fix = 1
                else:
                    fix = 0

                print('\n' + '='*(39-head_len-fix) + ' {:} '.format(counter) + '='*(39-head_len))
                print(report[1][counter])

        return res, y2plot, x2plot, yerr2plot, xerr2plot, name

    def fit_scan_sequence(self, scan_sequence, mod, pars, xgrid=[], yerr='std', xerr='std',
                          norm2one=False, binning=True, label_format='', fmt='o', select='',
                          fit_report=0, weights=False, fit_method='leastsq',
                          nan_policy='propagate', last_res_as_par=False, offset_t0=False,
                          plot_separate=False, show_single=False):
        """fit_scan_sequence

        Args:
            scan_sequence (list[
                list/tuple[list[int],
                int/str]]): sequence of scan lists and parameters.
            mod (lmfit.Model): fit model.
            pars (lmfit.parameters): fit parameters.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none]
                default is 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none]
                default is 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0
                default is False.
            binning (bool, optional): enable binning of data - default is True
            label_format (str, optional): format string for label text - default
                is empty.
            fmt (str, optional): format string of the plot - defaults is -o.
            select (str, optional): evaluatable string to select x-range.
                Defaults to empty string.
            fit_report (uint, optional): Default is 0 - no report. 1 - fit
                results. 2 - fit results and correlations.
            weights (bool, optional): enable weighting by inverse of errors.
                Defaults to False.
            fit_method (str, optional): lmfit's fit method. Defaults to 'leastsq'.
            nan_policy (str, optional): lmfit's NaN policy. Defaults to 'propagate'.
            last_res_as_par (bool, optional): use last fit result as start value
                for next fit. Defaults to False.
            offset_t0 (bool, optional): offset plot by t0 parameter of the fit
                results. Defaults to False.
            plot_separate (bool, optional): use separate subplots for different
                counters. Defaults to False.
            show_single (bool, optional): show single figure for each sequence
                element.
        Returns:
            (tuple):
            - *res (dict)* - fit result dictionary.
            - *sequence_data (OrderedDict)* - dictionary of the averaged scan data.
            - *parameters (list[str, float])* - parameters of the sequence.

        """
        # load data
        sequence_data, parameters, names = self.eval_scan_sequence(
            scan_sequence, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one, binning=binning)

        res = {}
        report_1 = []
        report_2 = []
        label_texts = []
        for counter in self.clist:
            res[counter] = {}

        for i, ((scan_list, parameter), name) in enumerate(zip(scan_sequence, names)):
            if show_single:
                plt.figure()
            # get the fit models and fit parameters if they are lists/tupels
            if isinstance(mod, (list, tuple)):
                _mod = mod[i]
            else:
                _mod = mod

            if last_res_as_par and i > 0:
                # use last results as start values for pars
                _pars = pars
                for pname, par in pars.items():
                    _pars[pname].value = res[counter][pname][i-1]
            else:
                if isinstance(pars, (list, tuple)):
                    _pars = pars[i]
                else:
                    _pars = pars

            lt = '#{:d}'.format(i+1)
            if len(label_format) > 0:
                try:
                    lt = label_format.format(parameter)
                except ValueError:
                    self.log.warning('Could not apply \'label_format\' to parameter!')

            label_texts.append(lt)
            # extract clist und xcol from sequence_data
            y2plot = {c: sequence_data[c][i] for c in self.clist}
            yerr2plot = {c: sequence_data[c + 'Err'][i] for c in self.clist}
            x2plot = sequence_data[self.xcol][i]
            xerr2plot = sequence_data[self.xcol + 'Err'][i]
            # fit the model and parameters to the data
            _res, _report = self._fit_scans(y2plot, x2plot, yerr2plot, xerr2plot, _mod, _pars,
                                            select, weights, fit_method=fit_method,
                                            nan_policy=nan_policy)

            # plot the data and fit
            self._plot_fit_scans(y2plot, x2plot, yerr2plot, xerr2plot, name, _res,
                                 offset_t0=offset_t0, label_text=lt, fmt=fmt,
                                 plot_separate=plot_separate)

            if show_single:
                plt.legend(frameon=True, loc=0, numpoints=1)
                plt.show()
            else:
                plt.legend(bbox_to_anchor=(0., 1.08, 1, .102), frameon=True,
                           loc=3, numpoints=1, ncol=3, mode="expand",
                           borderaxespad=0.)

            # store the results
            for counter in self.clist:
                for key in _res[counter].keys():
                    try:
                        res[counter][key] = np.append(res[counter][key], _res[counter][key])
                    except KeyError:
                        res[counter][key] = np.array([_res[counter][key]])

            # store the the report
            report_1.append(['>> ' + lt + ' <<'])
            for rep in _report[0]:
                report_1.append(rep)
            report_2.append(_report[1])

        # print the basic fit report
        if fit_report > 0:
            print(tabulate(report_1, headers=['counter', *mod.param_names],
                           tablefmt="fancy_grid"))
        # print the advanced fit report
        if fit_report > 1:
            for i, lt in enumerate(label_texts):
                lt_len = int(len(str(lt))/2)
                fix = 1 if np.mod(len(lt), 2) != 0 else 0
                print('\n' + '_'*(39-lt_len-fix) + ' {:} '.format(lt) + '_'*(39-lt_len))
                for counter in self.clist:
                    head_len = int(len(counter)/2)
                    fix = 1 if np.mod(len(counter), 2) != 0 else 0

                    print('\n' + '='*(39-head_len-fix) + ' {:} '.format(counter)
                          + '='*(39-head_len))
                    print(report_2[i][counter])

        return res, parameters, sequence_data

    @property
    def clist(self):
        return self._clist

    @clist.setter
    def clist(self, clist):
        """clist

        In order to keep backwards compatibility and to catch some wrong user
        inputs, the given ``clist`` is converted to a ``list`` when a ``dict``
        or number is given.

        """
        if isinstance(clist, dict):
            # the clist property is a dict, so retrun its keys as list
            clist = list(clist.keys())
        else:
            clist = list(clist)
        self._clist = clist
