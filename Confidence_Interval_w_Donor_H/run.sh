#!/bin/sh
#SBATCH --mail-user=Saeed.Ahmad@nationwidechildrens.org
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --job-name=20dis_inb
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
echo 'Loading modules'
module purge
module load miniforge3/24.3.0
conda activate Num_Sim
echo 'activated conda environment'
sbatch --time=10-0 --cpus-per-task=1 --wrap="python main.py"
conda deactivate
echo 'Ending modules'