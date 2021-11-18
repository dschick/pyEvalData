#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pyEvalData as ped
import os

@pytest.fixture(scope='session')  # one source to rule'em all   # can be put to its own file if desired
def source(tmp_path_factory):

    temp_folder = tmp_path_factory.mktemp("pytest_data")
    print(temp_folder)
    source = ped.io.Spec(file_name='test_file_spec.spec',
                         file_path='test_data/',
                         use_nexus=True,
                         force_overwrite=False,
                         update_before_read=False,
                         read_and_forget=True,
                         nexus_file_path=temp_folder,
                         nexus_file_name='2021_11_Schick')
    return source

def test_source(source):
    assert source.name=='test_file_spec.spec'
    assert source.scan1.meta['number'] == 1
    print(source.scan1.data['delay'])

def test_source(source):
    assert source.name=='test_file_spec.spec'

def test_source(source):
    assert source.name=='test_file_spec.spec'

def test_source(source):
    assert source.name=='test_file_spec.spec'


#    print(source)
#    data, meta = spec.get_scan_data(1)
#    print(data)
