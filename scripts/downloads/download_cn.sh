#!/usr/bin/env bash
set -e

CN_URL=http://data.allenai.org/downloads/missingfact/conceptnet.tar.gz
OUTDIR=data/conceptnet/

mkdir -p $OUTDIR
wget $CN_URL
tar xvfz conceptnet.tar.gz -C $OUTDIR
