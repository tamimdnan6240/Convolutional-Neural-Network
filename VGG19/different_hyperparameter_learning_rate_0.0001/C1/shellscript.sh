#!/bin/sh

#SBATCH --job-name=tamim_project1_test1
#SBATCH --partition=Leo
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:00:00
#SBATCH --mem=128gb



module load pytorch

python C1.py