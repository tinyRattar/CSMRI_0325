#!/bin/sh
#BSUB -q newgpu
#BSUB -o log/3d_DCDS
#BSUB -J 3d_DCDS
#BSUB -R "select[ngpus >0] rusage[ngpus_excl_p=1]"

echo ======Job Start======
python main.py dcds
