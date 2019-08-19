#!/usr/bin/env bash
set -e

GLOVE_URL=https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz
OUTDIR=data/glove

mkdir -p $OUTDIR
wget ${GLOVE_URL} -O ${OUTDIR}/glove.840B.300d.txt.gz
