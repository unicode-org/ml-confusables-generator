#!/bin/bash

# Set up script for dependencies and required libraries
# 1. Install HarfBuzz wraper and dependencies

cd harfpy
sudo python3 setup.py install
cd ../qahirah
sudo python3 setup.py install
cd ../pybidi
sudo python3 setup.py install
cd ../python_freetype
sudo python3 setup.py install
cd ..
