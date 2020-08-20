# Copyright 2020 Google LLC
#!/bin/bash

# Check if data already exists and download.
if [ ! -d "/tf/data/full_data" ]
then
    mkdir -p /tf/data
    cd /tf/data
    gdown --id 10HjA5EDUylON9x_pJjsILzYeim-Twhsi
    unzip -q full_data.zip -d .
    rm full_data.zip
else
    echo "Directory '/tf/data/full_data' already exists. Abort download."
fi
