#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/Ori_tr_r15
#BSUB -J Ori_tr_r15
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py default
