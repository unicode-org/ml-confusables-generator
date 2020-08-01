# Copyright 2020 Google LLC
#!/bin/bash

docker run --rm -it -p 8888:8888 -v $PWD:/tf -w /tf tensorflow/tensorflow:2.1.1-jupyter
