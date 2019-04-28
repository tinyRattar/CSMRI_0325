#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/sr_dunet
#BSUB -J sr_dunet
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py default
