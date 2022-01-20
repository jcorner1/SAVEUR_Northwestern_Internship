#!/bin/bash -l
#SBATCH -N 1     
#SBATCH --time=99:00:00  
#SBATCH --job-name=MRMS_data_agg
#SBATCH -n 1
#SBATCH --cpus-per-task=36

# Environment variables

#export MPLBACKEND=“agg”
conda activate data
cd /gpfs/fs1/home/ac.jcorner/Rainfall/MRMS
python MRMS_data_agg_script_2.py