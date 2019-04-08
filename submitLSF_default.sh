#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/Ori_2_tr
#BSUB -J Ori_2_tr
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py default
