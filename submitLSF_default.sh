#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/sr_dccnn
#BSUB -J dccnn
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py default2
