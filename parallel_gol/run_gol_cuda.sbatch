#!/bin/bash
#SBATCH --account=cis431_531
#SBATCH --partition=gpu
#SBATCH --job-name=gol_cuda
#SBATCH --output=output/gol_cuda_%A.out
#SBATCH --error=output/gol_cuda_%A.err
#SBATCH --time=00:10:00
#SBATCH --mem=16000M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load cuda

N=4096
STEPS=100

./gol_cuda $N $STEPS 0
