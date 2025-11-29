#!/bin/bash
#SBATCH --account=cis431_531
#SBATCH --partition=compute
#SBATCH --job-name=gol_omp_scaling
#SBATCH --output=output/gol_omp_scaling_%A.out
#SBATCH --error=output/gol_omp_scaling_%A.err
#SBATCH --time=00:10:00
#SBATCH --mem=16000M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_ENV=true

N=4096
STEPS=100

for T in 1 2 4 8 16 32
do
    export OMP_NUM_THREADS=$T
    echo "Threads=$T"
    ./gol_omp $N $STEPS 0
    echo ""
done
