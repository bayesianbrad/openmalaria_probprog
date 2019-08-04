#!/bin/bash
python3 setup_amortized_rs.py build_ext --inplace
yes | cp -rf cython_code/amortized*.so amortized_rs.so
rm -rf build
rm -rf cython_code