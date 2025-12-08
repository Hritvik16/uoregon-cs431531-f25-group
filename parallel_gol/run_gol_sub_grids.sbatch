#!/bin/bash
#SBATCH --account=cis431_531
#SBATCH --partition=compute
#SBATCH --job-name=gol_omp_scaling
#SBATCH --output=output/gol_omp_scaling_%A.out
#SBATCH --error=output/gol_omp_scaling_%A.err
#SBATCH --time=00:10:00
#SBATCH --mem=16000M
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_ENV=true

N=4096
STEPS=100

for T in 16
do
    export OMP_NUM_THREADS=4
    echo "Threads=4"
    srun ./gol_omp $N $STEPS 1 $T 1
    echo ""
done
