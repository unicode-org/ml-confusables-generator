#!/bin/bash

# Example for installing and listing new fonts

# Install Noto CJK font using apt-get
apt-get install -y fonts-noto-cjk-extra

# Show font face related to noto
fc-list | grep Noto
