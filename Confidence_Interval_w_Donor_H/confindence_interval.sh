#!/bin/bash
#SBATCH --job-name=fit_job
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
python confindence_interval.py $1