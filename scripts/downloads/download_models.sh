#!/usr/bin/env bash

set -e

MODELS_URL=http://data.allenai.org/downloads/missingfact/trained_models.tar.gz
OUTDIR=data/trained_models/

mkdir -p $OUTDIR
wget $MODELS_URL
tar xvfz trained_models.tar.gz -C $OUTDIR
