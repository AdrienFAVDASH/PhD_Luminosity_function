#!/bin/bash
#SBATCH --job-name=gama_create_sim
#SBATCH --output=/mnt/lustre/users/astro/aa2439/logs/evfit_%A_%a.out
#SBATCH --mail-type=END,FAIL,SUSPEND
#SBATCH --mail-user=aa2439@sussex.ac.uk
#SBATCH --array=1-10
#SBATCH --partition=long
#SBATCH --time=0-01:00:00
#SBATCH --export=ALL,PYTHONPATH=/mnt/lustre/users/astro/aa2439
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

cd /mnt/lustre/users/astro/aa2439
conda init
conda activate evfit_env

python <<EOF
import os
taskid = int(os.environ['SLURM_ARRAY_TASK_ID'])
import lf_ev
lf_ev.sim_cat_GIII(taskid - 1)
EOF
