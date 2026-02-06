#!/bin/bash
#SBATCH -J CLST # job name
#SBATCH -o log*.o%j # output and error file name (%j expands to jobID)
#SBATCH -n 96 # total number of tasks requested -> 2:96
#SBATCH --gres=gpu:1        # request a gpu
#SBATCH -N 2  # number of nodes you want to run on
#SBATCH -p gpu   # queue (partition)
#SBATCH -t 40:30:00 # run time (hh:mm:ss) - 12.0 hours in this example

#python3 run_train.py
python run_swe_train.py -c configs/swe_train.yml

