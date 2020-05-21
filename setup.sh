#!/bin/bash

# Set up script for dependencies and required libraries
# 1. Install qahirah (cairo) wraper and dependencies

cd qahirah
sudo python3 setup.py install
cd ../pybidi
sudo python3 setup.py install
cd ../python_freetype
sudo python3 setup.py install
cd ..
