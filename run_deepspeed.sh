#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

OUTPUT_FOLDER='/proj/karst/results/moco/'$DTime'_ds'

python src/main_deepspeed.py \
--my-config "config_train.conf" \
--save_dir $OUTPUT_FOLDER

