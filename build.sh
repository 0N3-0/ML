#!/bin/env sh

set -xe

gcc ml.c -Wall -Wextra -lm -O3 -o ml 
gcc train_visualizer.c -Wall -Wextra -lraylib -lm -O3 -o train_visualizer 
