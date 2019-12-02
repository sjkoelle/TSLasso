#!/bin/bash
#SBATCH --job-name repca_varnoise      # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition largemem     # Slurm partition to use
#SBATCH --ntasks 16          # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-05:00        # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=10000     # Memory limit for each tasks (in MB)
#SBATCH -o myscript_repcavarnoise0%j.out    # File to which STDOUT will be written
#SBATCH -e myscript_repcavarnoise0%j.err    # File to which STDERR will be written
#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate py35lynch
python rigidethanolpca_multirun_tangent_var0.py
python rigidethanolpca_multirun_tangent_var0p001.py
python rigidethanolpca_multirun_tangent_var0p0001.py
python rigidethanolpca_multirun_tangent_var0p00001.py

