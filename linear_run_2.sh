#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

FOLDER=$1
MODEL=$2
AMOUNT=$3
dataset=$4
OUTPUT_FOLDER='/proj/karst/results/moco/'$FOLDER'/linear_classification_'$MODEL''

python src/main.py \
--my-config "config_linear.conf" \
--save_dir ''$OUTPUT_FOLDER'/dataset_'$dataset'_'$AMOUNT'' \
--load_checkpoint_dir '/proj/karst/results/moco/'$FOLDER'/moco_model_'$MODEL'.pt' \
--seed $dataset \
--trainingset_split $AMOUNT



