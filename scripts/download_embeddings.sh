# Copyright 2020 Google LLC
#!/bin/bash

# Check if data already exists and download.
if [ ! -f "/tf/embeddings/full_data_triplet1.0_vec.tsv" ]
then
    mkdir -p /tf/embeddings
    cd /tf/embeddings
    gdown --id 1zOeQw4-Z7jmK41msitxBFnAUm1j1QHQm
    gdown --id 1lQHPw7Q8UxZUP2GLz8uMX6cu3ei_XfCq
else
    echo "Directory '/tf/embeddings/full_data_triplet1.0_vec.tsv' already exists. Abort download."
fi