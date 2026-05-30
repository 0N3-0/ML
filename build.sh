#!/bin/env sh

set -xe

gcc ml.c -Wall -Wextra -lm -O3 -o ml 
gcc model_run.c -Wall -Wextra -lraylib -lm -O3 -o model_run 
