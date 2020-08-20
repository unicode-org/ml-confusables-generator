# Copyright 2020 Google LLC
#!/bin/bash

# Set up script for dependencies and required libraries
# 1. Install libfreetype6 libcairo2
# 2. Install qahirah (cairo) wraper and dependencies
# 3. Install other required python libraries
# 4. Install required fonts

apt-get -y update
apt-get -y install libfreetype6 libcairo2 libsm6 libxext6 libfontconfig1 libxrender1 fontconfig libgl1-mesa-glx wget

mkdir -p qahirah
mkdir -p pybidi
mkdir -p python_freetype

cd qahirah
python3 setup.py install
cd ../pybidi
python3 setup.py install
cd ../python_freetype
python3 setup.py install
cd ..

python3 -m pip install opencv-python tensorflow-addons==0.9.1 pandas easydict sklearn

apt-get install -y fonts-noto-cjk-extra

