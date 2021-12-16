clear
source init.sh

#python3 main.py --batch_size 12 \
#       --steps_per_epoch_train 250 --val_steps 5 --epochs 200 --run_name debug --debug \
#       --train_data_path ~/hooknet/hooknet-breastdata-annonymized/training/WSIs \
#       --train_annotations_path ~/hooknet/hooknet-breastdata-annonymized/training/XML \
#       --valid_data_path ~/hooknet/hooknet-breastdata-annonymized/validation/WSIs \
#       --valid_annotations_path ~/hooknet/hooknet-breastdata-annonymized/validation/XML

mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python3 main.py --batch_size 2 --horovod \
       --steps_per_epoch_train 1000 --val_steps 5 --epochs 200 --run_name debug --debug \
       --train_data_path /nfs/examode/Breast/Radboudumc/hooknet-breastdata-annonymized/training/WSIs \
       --train_annotations_path /nfs/examode/Breast/Radboudumc/hooknet-breastdata-annonymized/training/XML \
       --valid_data_path /nfs/examode/Breast/Radboudumc/hooknet-breastdata-annonymized/validation/WSIs \
       --valid_annotations_path /nfs/examode/Breast/Radboudumc/hooknet-breastdata-annonymized/validation/XML

#python3 apply.py -w output/saved_model_test/model.h5 \
#                 -d output/saved_model_test \
#                 -i /nfs/examode/Breast/Radboudumc/hooknet-breastdata-annonymized/testing/WSIs \
#                 -m /nfs/examode/Breast/Radboudumc/hooknet-breastdata-annonymized/testing/XML

