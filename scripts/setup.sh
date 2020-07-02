#!/bin/bash

# Set up script for dependencies and required libraries
# 1. Install libfreetype6 libcairo2
# 2. Install qahirah (cairo) wraper and dependencies
# 3. Install other required python libraries

apt-get -y update
apt-get -y install libfreetype6 libcairo2 libsm6 libxext6 libfontconfig1 libxrender1 fontconfig

cd qahirah
python3 setup.py install
cd ../pybidi
python3 setup.py install
cd ../python_freetype
python3 setup.py install
cd ..

python3 -m pip install opencv-python tensorflow-addons==0.9.1