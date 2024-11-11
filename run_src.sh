#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1
##
#SBATCH --job-name=experiment
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err
##
#SBATCH --gres=gpu:rtx3090:1

hostname
date

module add CUDA/11.3.0
module add ANACONDA/2020.11

for argv in "$*"
do
    $argv
done
