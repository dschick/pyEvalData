#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest


def test_evaluation(evaluation):
    data = evaluation
    assert data.cdef == {}
    assert data.clist == []
    cdef = {
        'Pumped_demod':  'Pumped-(Unpumped-mean(Unpumped))',
        'PumpedM_demod': 'PumpedM-(UnpumpedM-mean(UnpumpedM))',
        'M':             'Pumped_demod-PumpedM_demod',
    }
    data.cdef = cdef
    data.xcol = 'delay'
    data.clist = ['M']
    data1 = data.get_scan_data(1)
    assert data1['delay'][0] == pytest.approx(-0.998557475)
    y, x, yerr, xerr, name = data.plot_scans([1])
    assert y[data.clist[0]][0] == pytest.approx(0.02183873769)
