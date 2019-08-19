#!/usr/bin/env bash

set -e

TRAIN_URL=http://data.allenai.org/downloads/missingfact/training_data.tar.gz
OUTDIR=data/input/

mkdir -p $OUTDIR
wget $TRAIN_URL
tar xvfz training_data.tar.gz -C $OUTDIR
