# Copyright 2020 Google LLC
#!/bin/bash

if [ ! -d "/tf/ckpts/TripletTransferTF" ]
then
    mkdir -p /tf/ckpts/TripletTransferTF
    cd /tf/ckpts/TripletTransferTF
    gdown --id 1ecONfCiBgz640V8zcAzwlJ_1SbI5ltqy
    gdown --id 1WiKxLq3uQRgep_tg7YgUcjNAq-A3Kb3X
    gdown --id 12Fe2lVOT5yNV7wOja2vPvsb1iPbVbA53
else
    echo "Directory '/tf/ckpts/TripletTransferTF' already exists. Abort download."
fi
