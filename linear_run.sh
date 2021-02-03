#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

FOLDER=$1
MODEL=$2
OUTPUT_FOLDER='/proj/karst/results/moco/'$FOLDER'/linear_classification_'$MODEL''

dataset=0
while [ $dataset -le 4 ]
do

python src/main.py \
--my-config "config_linear.conf" \
--save_dir ''$OUTPUT_FOLDER'/dataset_'$dataset'' \
--load_checkpoint_dir '/proj/karst/results/moco/'$FOLDER'/moco_model_'$MODEL'.pt' \
--seed $dataset

let dataset=dataset+1

done


