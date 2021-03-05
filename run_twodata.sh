#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

OUTPUT_FOLDER='/proj/karst/results/moco/'$DTime'_moco_20x2.5x'

python src/main.py \
--my-config "config_train_twodata.conf" \
--save_dir $OUTPUT_FOLDER

