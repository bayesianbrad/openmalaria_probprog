#!/bin/bash

# process arguments
while getopts "e:s:a:" opt
do
  case "$opt" in
    e ) envrion="$OPTARG";; # name of conda environment
    s ) script="$OPTARG" ;; # script to run
    a ) address="$OPTARG" ;; # address of process to amortize
  esac
done
# set up conda environment
eval "$(conda shell.bash hook)"
conda activate $envrion
PYTHONPATH=$(which python)

echo " python path $PYTHONPATH"

python $script --address $address \
--batchsize 1024 \
--model 'density_estimator' \
--trainon True \
--teston True \
--trainiterations 1000 \
--testiterations 1000 \
--ncores 1 \
--proposal 'Normalapproximator' \
--savemodel True \
--op "{'name': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.999)}"




