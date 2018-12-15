#!/usr/bin/env bash

MMAPS_FILE_NAME=""
MODELS_FILE_NAME=""
AIDA_FILE_NAME=""

#mkdir models
#mkdir -p data2/training_files/mmaps
#
#curl -L ${MMAPS_FILE_NAME} -o data/mmaps
#curl -L ${MODELS_FILE_NAME} -o data/models/conll-v0.1.pt
#curl -L ${AIDA_FILE_ANME} -0 data/AIDA_YAGO2-dataset.tsv

echo "Processing Conll Data"
python scripts/gen_conll_train.py -d data -a data/AIDA_YAGO2-dataset.tsv