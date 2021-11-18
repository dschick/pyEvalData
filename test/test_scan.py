#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyEvalData.io import Scan


def test_scan():
    scan = Scan(1)
    assert scan.number == 1
