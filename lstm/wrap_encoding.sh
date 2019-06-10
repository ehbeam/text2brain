#!/bin/sh
for FILE in encode_*.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
