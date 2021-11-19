#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest


@pytest.mark.parametrize('mysource, sname, scan_num, scan_delay',
                         [
                          ('source_spec', 'example_file_spec.spec', 1, -0.998557475),
                          ('source_pal', 'pal_file', 40, -33)
                         ])
def test_source(mysource, sname, scan_num, scan_delay, request):

    s = request.getfixturevalue(mysource)
    assert s.name == sname
    assert s.get_scan(scan_num).meta['number'] == scan_num
    assert s.get_scan(scan_num).data['delay'][0] == pytest.approx(scan_delay)

# def test_evaluation(evaluation):
#     data = evaluation
#     assert data.cdef == {}
#     assert data.clist == []
#     cdef = {
#         'pumped_demod':  'pumped-(unpumped-mean(unpumped))',
#         'pumpedM_demod': 'pumpedM-(unpumpedM-mean(unpumpedM))',
#         'M':             'pumped_demod-pumpedM_demod',
#     }
#     data.cdef = cdef
#     data.clist = ['M']
#     data1 = data.get_scan_data(1)
#     assert data1['delay'][0] == pytest.approx(-0.998557475)
