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
        sequence = self.sequence(scan_sequence, xgrid=xgrid, yerr=yerr, xerr=xerr,
                                 norm2one=norm2one, binning=binning)

        return sequence.data, sequence.parameters, sequence.names

    def plot_scans(self, scan_list, xgrid=np.array([]), yerr='std', xerr='std', norm2one=False,
                   binning=True, label_text='', fmt='-o', plot_separate=False, **kwargs):
        """plot_scans

        Old syntax (<v2.0.0) for plotting a list of scans from the source file.

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
        scans = self.scans(scan_list, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one,
                           binning=binning)

        scans.plot(label_text=label_text, fmt=fmt, plot_separate=plot_separate, **kwargs)

        return scans.y2plot, scans.x2plot, scans.yerr2plot, scans.xerr2plot, scans.name

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
            label_format (str, optional): format string to generate labels from
                parameters. Defaults to empty string.
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
        try:
            sequence_type = kwargs.pop('sequence_type')
        except KeyError:
            pass

        if len(sequence_type) > 0 and len(label_format) == 0:
            # use sequence_type to generate label_format for backwards compatibility
            if sequence_type == 'temperature':
                label_format = '{:d K}'
        elif len(sequence_type) > 0 and len(label_format) > 0:
            self.log.warning('sequence_type parameter is ignored for label_format')

        sequence = self.sequence(scan_sequence, xgrid=xgrid, yerr=yerr, xerr=xerr,
                                 norm2one=norm2one, binning=binning, label_format=label_format)

        sequence.plot(fmt=fmt, plot_separate=plot_separate, show_single=show_single, **kwargs)

        return sequence.data, sequence.parameters, sequence.names, sequence.label_texts

    def fit_scans(self, scan_list, mod, pars, xgrid=[], yerr='std', xerr='std', norm2one=False,
                  binning=True, label_text='', fmt='o', select='', fit_report=0, weights=False,
                  fit_method='leastsq', nan_policy='propagate', skip_plot=False, offset_t0=False,
                  plot_separate=False, **kwargs):
        """fit_scans

        Old syntax (<v2.0.0) to evaluate, fit, and plot the results of a given
        list of scans from the source file.

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
            skip_plot (bool, optional): Skip plotting. Defaults to False.
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
        scans = self.scans(scan_list, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one,
                           binning=binning)

        scans.fit(mod, pars, select, report=fit_report, weights=weights, fit_method=fit_method,
                  nan_policy=nan_policy)

        if not skip_plot:
            scans.plot(label_text=label_text, fmt=fmt, plot_separate=plot_separate,
                       offset_t0=offset_t0, **kwargs)

        return (scans.fit_result, scans.y2plot, scans.x2plot, scans.yerr2plot, scans.xerr2plot,
                scans.name)

    def fit_scan_sequence(self, scan_sequence, mod, pars, xgrid=[], yerr='std', xerr='std',
                          norm2one=False, binning=True, label_format='', fmt='o', select='',
                          fit_report=0, weights=False, fit_method='leastsq', nan_policy='propagate',
                          last_res_as_par=False, skip_plot=False, offset_t0=False,
                          plot_separate=False, show_single=False, **kwargs):
        """fit_scan_sequence

        Evaluate, fit, and plot the results of a given scan sequence from the
        source file.

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
            binning (bool, optional): enable binning of data. Defaults to True.
            label_format (str, optional): format string to generate labels from
                parameters. Defaults to empty string.
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
            skip_plot (bool, optional): Skip plotting. Defaults to False.
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
        try:
            sequence_type = kwargs.pop('sequence_type')
        except KeyError:
            pass

        if len(sequence_type) > 0 and len(label_format) == 0:
            # use sequence_type to generate label_format for backwards compatibility
            if sequence_type == 'temperature':
                label_format = '{:d K}'
            elif sequence_type == 'text':
                label_format = '{:s}'
        elif len(sequence_type) > 0 and len(label_format) > 0:
            self.log.warning('sequence_type parameter is ignored for label_format')

        sequence = self.sequence(scan_sequence, xgrid=xgrid, yerr=yerr, xerr=xerr,
                                 norm2one=norm2one, binning=binning, label_format=label_format)

        sequence.fit(mod, pars, select=select, report=fit_report, weights=weights,
                     fit_method=fit_method, nan_policy=nan_policy, last_res_as_par=last_res_as_par)

        if not skip_plot:
            sequence.plot(fmt=fmt, plot_separate=plot_separate, show_single=show_single,
                          offset_t0=offset_t0, **kwargs)

        return sequence.fit_results, sequence.parameters, sequence.data

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

    def scans(self, scan_list, xgrid=[], yerr='std', xerr='std', norm2one=False, binning=True):
        """scans

        Factory method to populate and return `Scans` object for further
        plotting and fitting. It internally calls the `eval_scans` method.

        Args:
            scan_list (list[int]): list of scan numbers.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none].
                Defaults to 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none].
                Defaults to 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0.
                Defaults to False.
            binning (bool, optional): enable binning of data. Defaults to True.

        Returns:
            Scans: object for plotting and fitting.

        """
        y2plot, x2plot, yerr2plot, xerr2plot, name = self.eval_scans(
            scan_list, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one, binning=binning)

        return Scans(name, self.xcol, y2plot, x2plot, yerr2plot, xerr2plot)

    def sequence(self, scan_sequence, xgrid=[], yerr='std', xerr='std', norm2one=False,
                 binning=True, label_format=''):
        """sequence

        Args:
            scan_sequence (list[
                list/tuple[list[int],
                int/str]]): sequence of scan lists and parameters.
            xgrid (ndarray, optional): grid to bin the data to - default is
                empty so use the x-axis of the first scan.
            yerr (ndarray, optional): type of the errors in y: [err, std, none].
                Defaults to 'std'.
            xerr (ndarray, optional): type of the errors in x: [err, std, none].
                Defaults to 'std'.
            norm2one (bool, optional): normalize transient data to 1 for t < t0
                Defaults to False.
            binning (bool, optional): enable binning of data. Defaults to True.
            label_format (str, optional): format string to generate labels from
                parameters. Defaults to empty string.

        Returns:
            Sequence: object for plotting and fitting.

        """
        scans_list = []
        parameters = []

        for i, (scanlist, parameter) in enumerate(scan_sequence):

            scans = self.scans(scanlist, xgrid=xgrid, yerr=yerr, xerr=xerr, norm2one=norm2one,
                               binning=binning)

            parameters.append(parameter)
            scans_list.append(scans)

        return Sequence(parameters, scans_list, label_format=label_format)


class Scans():
    """Scans

    This class holds the data of a list of scans and for a given set of counters.
    The data is stored as attributes and can be easily accessed. The `Scans`
    objetcs have no access to their parent `Evaluation` objects and are solely
    meant as containers for storing, fitting, and plotting evaluated data.

    Args:
        name (str): name of the scans data set.
        xcol (str): name of the x-data.
        y2plot (OrderedDict): y-data for diffent counters.
        x2plot (dict[ndarray]): x-data for the xcol.
        yerr2plot (OrderedDict): y-error for diffent counters.
        xerr2plot (dict[ndarray]): x-error for the xcol.

    """
    def __init__(self, name, xcol, y2plot, x2plot, yerr2plot, xerr2plot):
        self.log = logging.getLogger(__name__)
        self.name = name
        self.xcol = xcol
        self.clist = list(y2plot.keys())
        self.y2plot = y2plot
        self.x2plot = x2plot
        self.yerr2plot = yerr2plot
        self.xerr2plot = xerr2plot
        self.fit_result = []
        self.fit_report = []

    def fit(self, mod, pars, select='', report=0, weights=False, fit_method='leastsq',
            nan_policy='propagate'):
        """fit

        Fit the data set using `lmfit` models and parameters. Sets the
        `fit_result` and `fit_report` attributes.

        Args:
            mod (lmfit.Model): fit model.
            pars (lmfit.parameters): fit parameters.
            select (str, optional): evaluatable string to select x-range.
                Defaults to empty string.
            report (uint, optional): Default is 0 - print no report.
                1 - print fit results. 2 - print fit results and correlations.
            weights (bool, optional): enable weighting by inverse of errors.
                Defaults to False.
            fit_method (str, optional): lmfit's fit method. Defaults to 'leastsq'.
            nan_policy (str, optional): lmfit's NaN policy. Defaults to 'propagate'.

        Returns:
            Scans: current object.

        """
        res = {}  # initialize the results dict
        for counter in self.clist:
            res[counter] = {}
            # get the fit models and fit parameters if they are lists/tuples

            # evaluate the select statement
            if select == '':
                # select all
                sel = np.ones_like(self.y2plot[counter], dtype=bool)
            else:
                sel = eval(select)

            # execute the select statement
            _y2plot = self.y2plot[counter][sel]
            _x2plot = self.x2plot[sel]
            _yerr2plot = self.yerr2plot[counter][sel]
            _xerr2plot = self.xerr2plot[sel]

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

            # add the fit results to the returns
            for pname, par in pars.items():
                res[counter][pname] = out.best_values[pname]
                res[counter][pname + 'Err'] = out.params[pname].stderr

            res[counter]['chisqr'] = out.chisqr
            res[counter]['redchi'] = out.redchi
            res[counter]['CoM'] = np.sum(_y2plot*_x2plot)/np.sum(_y2plot)
            res[counter]['int'] = np.trapz(_y2plot, x=_x2plot)
            res[counter]['fit'] = out

        self.fit_result = res

        if report == 1:
            self.print_fit_report(full=False)
        elif report == 2:
            self.print_fit_report(full=True)

        return self

    def plot(self, label_text='', fmt='-o', plot_separate=False, offset_t0=False, **kwargs):
        """plot

        Plot the data set. If a `fit_result` is available, it is plotted on top
        of the data.

        Args:
            label_text (str, optional): label of the plot - default is none.
            fmt (str, optional): format string of the plot - defaults is -o.
            plot_separate (bool, optional): use separate subplots for different
                counters. Defaults to False.
            offset_t0 (bool, optional): offset plot by t0 parameter of the fit
                results. Defaults to False.

        Returns:
            Scans: current object.

        """
        offsetX = 0
        if len(self.fit_result) > 0:
            # fit result is available
            fmt = 'o'
            if offset_t0:
                try:
                    offsetX = self.fit_result['t0']
                except KeyError:
                    self.log.warning('No parameter \'t0\' present in model!')
            else:
                offsetX = 0

        # plot all keys in the clist
        for i, counter in enumerate(self.clist):
            # iterate the counter list
            title = self.name
            if plot_separate:
                # use subplot for separate plotting
                plt.subplot(1, len(self.clist), i+1)
                title += ' | ' + counter

            if len(label_text) == 0:
                # if no label_text is given use the counter name
                lt = counter
            else:
                if len(self.clist) > 1 and not plot_separate:
                    # for multiple counters add the counter name to the label
                    lt = label_text + ' | ' + counter
                else:
                    # for a single counter just use the label_text
                    lt = label_text

            # plot the data for each counter
            if (self.xerr2plot is None) & (self.yerr2plot is None):
                plot = plt.plot(self.x2plot-offsetX, self.y2plot[counter], fmt, label=lt, **kwargs)
            else:
                plot = plt.errorbar(self.x2plot-offsetX, self.y2plot[counter], fmt=fmt, label=lt,
                                    xerr=self.xerr2plot, yerr=self.yerr2plot[counter], **kwargs)

            if len(self.fit_result) > 0:
                # fit result is available
                x2plotFit = np.linspace(np.min(self.x2plot), np.max(self.x2plot), 10000)
                plt.plot(x2plotFit-offsetX, self.fit_result[counter]['fit'].eval(x=x2plotFit), '-',
                         lw=2, alpha=1, color=plot[0].get_color())

            plt.xlabel(self.xcol)
            plt.title(title)
            plt.legend(frameon=True, loc=0, numpoints=1)

        return self

    def print_fit_report(self, full=False):
        """print_fit_report

        _summary_

        Args:
            full (bool, optional): _description_. Defaults to False.
        """
        tables = []
        reports = []

        for counter in self.clist:
            fit = self.fit_result[counter]['fit']
            tables.append([counter, *fit.best_values.values()])
            reports.append(fit.fit_report())
            headers = ['counter', *fit.best_values.keys()]

        if full:
            # print full fit report including correlations
            for table, report in zip(tables, reports):
                print(tabulate([table], headers=headers, tablefmt="fancy_grid"))
                print(report)
        else:
            # print only tabulated fit results
            print(tabulate(tables, headers=headers, tablefmt="fancy_grid"))

    @property
    def fit_result(self):
        if len(self._fit_result) == 0:
            self.log.warning('No fit result available.\n'
                             'Call .fit() method in advance.')
        else:
            return self._fit_result

    @fit_result.setter
    def fit_result(self, res):
        self._fit_result = res


class Sequence():
    """Sequence

    This class holds the data of a sequence of scan lists and parameters for a
    given set of counters. The data is stored as attributes and can be easily
    accessed. The `Sequence` objetcs have no access to their parent `Evaluation`
    objects and are solely meant as containers for storing, fitting, and
    plotting evaluated data.

    Args:
        parameters (list[str]): list of parameters.
        scans_list (list[Scans]): list of `Scans` objects.
        label_format (str, optional): format string to generate labels from
            parameters. Defaults to empty string.

    """

    def __init__(self, parameters, scans_list, label_format=''):
        self.log = logging.getLogger(__name__)
        self.parameters = parameters
        self.scans_list = scans_list
        self.label_format = label_format
        self.xcol = scans_list[0].xcol
        self.clist = scans_list[0].clist

        if len(label_format) > 0:
            try:
                self.label_texts = []
                for parameter in self.parameters:
                    self.label_texts.append(label_format.format(parameter))
            except ValueError:
                self.log.warning('Could not apply \'label_format\' to parameter!')
        else:
            self.label_texts = ['#{:02d}'.format(i+1) for i in range(len(parameters))]

    def fit(self, mod, pars, select='', report=0, weights=False, fit_method='leastsq',
            nan_policy='propagate', last_res_as_par=False):
        """fit _summary_

        Args:
            mod (_type_): _description_
            pars (_type_): _description_
            select (str, optional): _description_. Defaults to ''.
            report (int, optional): _description_. Defaults to 0.
            weights (bool, optional): _description_. Defaults to False.
            fit_method (str, optional): _description_. Defaults to 'leastsq'.
            nan_policy (str, optional): _description_. Defaults to 'propagate'.
            last_res_as_par (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        for i, scans in enumerate(self.scans_list):
            if isinstance(mod, (list, tuple)):
                _mod = mod[i]
            else:
                _mod = mod

            if last_res_as_par and i > 0:
                # use last results as start values for pars
                _pars = pars
                for counter in self.clist:
                    for pname, _ in pars.items():
                        _pars[pname].value = last_scans.fit_result[counter][pname]
            else:
                if isinstance(pars, (list, tuple)):
                    _pars = pars[i]
                else:
                    _pars = pars

            scans.fit(_mod, _pars, select=select, report=0, weights=weights, fit_method=fit_method,
                      nan_policy=nan_policy)

            last_scans = scans # remember for last_res_as_par

        if report == 1:
            self.print_fit_report(full=False)
        elif report == 2:
            self.print_fit_report(full=True)

        #     # store the the report
        #     report_1.append(['>> ' + lt + ' <<'])
        #     for rep in _report[0]:
        #         report_1.append(rep)
        #     report_2.append(_report[1])

        # # print the basic fit report
        # if report > 0:
        #     print(tabulate(report_1, headers=['counter', *mod.param_names],
        #                    tablefmt="fancy_grid"))
        # # print the advanced fit report
        # if report > 1:
        #     for i, lt in enumerate(self.label_texts):
        #         lt_len = int(len(str(lt))/2)
        #         fix = 1 if np.mod(len(lt), 2) != 0 else 0
        #         print('\n' + '_'*(39-lt_len-fix) + ' {:} '.format(lt) + '_'*(39-lt_len))
        #         for counter in self.clist:
        #             head_len = int(len(counter)/2)
        #             fix = 1 if np.mod(len(counter), 2) != 0 else 0

        #             print('\n' + '='*(39-head_len-fix) + ' {:} '.format(counter)
        #                   + '='*(39-head_len))
        #             print(report_2[i][counter])
        return self

    def plot(self, fmt='-o', plot_separate=False, show_single=False, offset_t0=False, **kwargs):
        """plot _summary_

        Args:
            fmt (str, optional): _description_. Defaults to '-o'.
            plot_separate (bool, optional): _description_. Defaults to False.
            show_single (bool, optional): _description_. Defaults to False.
            offset_t0 (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        for i, (label, scans) in enumerate(zip(self.label_texts, self.scans_list)):
            if show_single:
                plt.figure()

            scans.plot(label_text=label, fmt=fmt, plot_separate=plot_separate, offset_t0=offset_t0,
                       **kwargs)

            if show_single:
                plt.legend(frameon=True, loc=0, numpoints=1)
                plt.show()

        return self

    def print_fit_report(self, full=False):
        """print_fit_report

        _summary_

        Args:
            full (bool, optional): _description_. Defaults to False.
        """
        tables = []
        reports = []

        for counter in self.clist:
            fit = self.fit_result[counter]['fit']
            tables.append([counter, *fit.best_values.values()])
            reports.append(fit.fit_report())
            headers = ['counter', *fit.best_values.keys()]

        if full:
            # print full fit report including correlations
            for table, report in zip(tables, reports):
                print(tabulate([table], headers=headers, tablefmt="fancy_grid"))
                print(report)
        else:
            # print only tabulated fit results
            print(tabulate(tables, headers=headers, tablefmt="fancy_grid"))

    @property
    def data(self):
        data = collections.OrderedDict()
        # create a list of all counters from the scan and append the xcol
        sequence_counters = self.clist + [self.xcol]
        for scans in self.scans_list:
            for counter in sequence_counters:
                # traverse all counters in the data set
                if counter not in data.keys():
                    # if the counter is not in the return data dict - add the key
                    data[counter] = []
                    data[counter + 'Err'] = []

                # add the counter data to the return data dict
                try:
                    data[counter].append(scans.y2plot[counter])
                    data[counter + 'Err'].append(scans.yerr2plot[counter])
                except KeyError:
                    data[counter].append(scans.x2plot)
                    data[counter + 'Err'].append(scans.xerr2plot)

        return data

    @property
    def names(self):
        return [scans.name for scans in self.scans_list]

    @property
    def fit_results(self):
        res = {}
        for counter in self.clist:
            res[counter] = {}
            for scans in self.scans_list:
                for key in scans.fit_result[counter].keys():
                    try:
                        res[counter][key] = np.append(
                            res[counter][key], scans.fit_result[counter][key])
                    except KeyError:
                        res[counter][key] = np.array([scans.fit_result[counter][key]])

        return res
