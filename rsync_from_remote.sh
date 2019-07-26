remotePath=rat@49.52.10.27:/home/rat/CSMRI_0325

echo rsync from remote:${remotePath}

echo rsync network dir...
rsync -a ${remotePath}/network ./
echo rsync config dir...
rsync -a ${remotePath}/config ./
echo rsync core_ver2.py
rsync -a ${remotePath}/core_ver2.py ./
echo rsync dataProcess.py
rsync -a ${remotePath}/dataProcess.py ./
echo rsync util dir...
rsync -a ${remotePath}/util ./

echo DONE
