#!/bin/sh

#SBATCH --job-name=tamim_project1_test16
#SBATCH --partition=GPU 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=128gb


module load pytorch

python test.py