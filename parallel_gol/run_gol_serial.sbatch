#!/bin/bash
#SBATCH --account=cis431_531            ### your ‘charge’ account 
#SBATCH --partition=compute             ### queue to submit to
#SBATCH --job-name=gol_parallel               ### job name
#SBATCH --output=output/gol_parallel_8_CPU_1024_100_0.out   ### file in which to store job stdout
#SBATCH --error=output/gol_parallel_8_CPU_1024_100_0.err    ### file in which to store job stderr
#SBATCH --time=00:15:00                 ### wall-clock time limit, in minutes
#SBATCH --mem=16000M                    ### memory limit per node (K|M|G|T)
#SBATCH --nodes=1                       ### number of nodes to use
#SBATCH --ntasks-per-node=1             ### number of MPI tasks per node
#SBATCH --cpus-per-task=32              ### number of CPUs for each task         

# OpenMP environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES="cores"
export OMP_PROC_BIND='spread'
export OMP_DISPLAY_ENV='true'

# executable param1 param2 param3 :
./gol_omp 1024 100 0
