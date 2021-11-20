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
