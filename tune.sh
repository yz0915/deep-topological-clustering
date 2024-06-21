#!/bin/bash
##SBATCH --mem=128G
#SBATCH --job-name=deep_top_tune
#SBATCH -t 120:00:00  # time requested in hour:minute:second
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=top_tune/%x-%j.out  # Save stdout to sout directory
#SBATCH --error=top_tune/%x-%j.err   # Save stderr to sout directory

python3 ./graph_autoencoder.py