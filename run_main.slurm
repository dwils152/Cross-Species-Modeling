#!/bin/bash
#SBATCH --partition=Orion
#SBATCH --job-name=CrossSpeciesPrediction
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --time=500:00:00
#SBATCH --mem=16GB

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo ""

# Code starts here -----------------------------------------
export NXF_TEMP=/scratch/${USER}/tmp
export TMPDIR=/scratch/${USER}/tmp

nextflow run main.nf \
    --sample_sheet sample_sheet.csv \
    -resume



cd $SLURM_SUBMIT_DIR
echo "Hello World! I ran on compute node $(/bin/hostname -s)"
echo ""
echo "======================================================"
echo "End Time   : $(date)"
