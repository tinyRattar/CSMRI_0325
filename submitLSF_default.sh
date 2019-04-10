#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/dOri_tr
#BSUB -J dOri_tr
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py dOri_tr
