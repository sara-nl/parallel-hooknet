#!/bin/bash
#Set job requirements
#SBATCH -p gpu_rtx2080ti
#SBATCH -t 5-00:00:00
#SBATCH -N 4
#SBATCH -J BS40_4_node_1000

module purge
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0
module load mpicopy

echo "$TMPDIR"

mpicopy /nfs/examode/Breast/Radboudumc/hooknet-breastdata-annonymized

echo "Done copying"

clear
source init.sh

mpirun -map-by ppr:4:node -np 16 -x LD_LIBRARY_PATH -x PATH python3 main.py \
       --l2_lambda 0.0001 --horovod --batch_size 40 --run_name BS40_4_node_1000 \
       --steps_per_epoch_train 1000 --val_steps 1000 --epochs 200 \
       --train_data_path "$TMPDIR"/hooknet-breastdata-annonymized/training/WSIs \
       --train_annotations_path "$TMPDIR"/hooknet-breastdata-annonymized/training/XML \
       --valid_data_path "$TMPDIR"/hooknet-breastdata-annonymized/validation/WSIs \
       --valid_annotations_path "$TMPDIR"/hooknet-breastdata-annonymized/validation/XML

#python3 main.py --batch_size 12 \
#       --steps_per_epoch_train 1000 --val_steps 1000 --epochs 200 --run_name 1_gpu_1000 \
#       --train_data_path "$TMPDIR"/hooknet-breastdata-annonymized/training/WSIs \
#       --train_annotations_path "$TMPDIR"/hooknet-breastdata-annonymized/training/XML \
#       --valid_data_path "$TMPDIR"/hooknet-breastdata-annonymized/validation/WSIs \
#       --valid_annotations_path "$TMPDIR"/hooknet-breastdata-annonymized/validation/XML
