#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pyEvalData as ped


@pytest.fixture(scope='module')
def source_pal(tmp_path_factory):
    temp_folder = tmp_path_factory.mktemp("pytest_data")
    source_pal = ped.io.PalH5(name='pal_file',
                              file_name='{0:07d}',
                              file_path='example_data/example_files_pal',
                              use_nexus=True,
                              force_overwrite=False,
                              update_before_read=False,
                              read_and_forget=True,
                              nexus_file_path=temp_folder,
                              nexus_file_name='test_pal')
    return source_pal


@pytest.fixture(scope='module')
def source_spec(tmp_path_factory):
    temp_folder = tmp_path_factory.mktemp("pytest_data")
    source_spec = ped.io.Spec(file_name='example_file_spec.spec',
                              file_path='example_data/',
                              use_nexus=True,
                              force_overwrite=False,
                              update_before_read=False,
                              read_and_forget=True,
                              nexus_file_path=temp_folder,
                              nexus_file_name='test_spec')
    return source_spec


@pytest.fixture(scope='module')
def evaluation(source_spec):
    data = ped.Evaluation(source_spec)
    return data
