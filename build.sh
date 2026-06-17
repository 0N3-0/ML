#!/bin/env sh

set -xe

gcc -DML_LOG_LEVEL=3 ml.c -Wall -Wextra -lm -O3 -o ml 
gcc -DML_LOG_LEVEL=3 train_visualizer.c -Wall -Wextra -lraylib -lm -O3 -o train_visualizer 
