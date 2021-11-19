#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pyEvalData as ped


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    temp_folder = tmp_path_factory.mktemp("pytest_data")
    print(temp_folder)
    source = ped.io.Spec(file_name='test_file_spec.spec',
                         file_path='test/test_data/',
                         use_nexus=True,
                         force_overwrite=False,
                         update_before_read=False,
                         read_and_forget=True,
                         nexus_file_path=temp_folder,
                         nexus_file_name='test')
    return source


@pytest.fixture(scope='module')
def evaluation(source):
    data = ped.Evaluation(source)
    return data
