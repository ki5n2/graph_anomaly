#!/bin/bash

# python argparse source for experiments
experiments=(
"--random-seed 1 --epochs 150 --n-cluster 3 --learning-rate 0.0001"
"--dataset-name BZR --random-seed 1 --epochs 500 --n-cluster 3 --learning-rate 0.0001"
"--dataset-name DHFR --random-seed 1 --epochs 150 --n-cluster 3 --learning-rate 0.0001"
"--dataset-name AIDS --random-seed 1 --epochs 100 --n-cluster 3 --learning-rate 0.0001"
)

# default prefix of job name
DEFAULT_NAME=graph

# DEVICE SETTING
DEVICES=(
    "--partition=hgx --gres=gpu:hgx:1 "
    "--partition=gpu1 --gres=gpu:rtx3090:1 "
    "--partition=gpu2 --gres=gpu:a10:1 "
    "--partition=gpu3 --gres=gpu:a10:1 "
    "--partition=gpu4 --gres=gpu:rtxa6000:1 "
    "--partition=gpu5 --gres=gpu:rtxa6000:1 "
    "--partition=edu1 --gres=gpu:a10:1 "
    )

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# virutal environment directory
ENV=/home1/rldnjs16/ENTER/envs/graph/bin/python3

# file directory of experiment ".py"
EXECUTION_FILE=/home1/rldnjs16/graph_anomaly_detection/BERT_model14__.py

for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index ${DEVICES[1]} $RUN_SRC $ENV $EXECUTION_FILE ${experiments[$index]} 
    sleep 1
done
