cd ..
projname=q2a_function

CUDA_VISIBLE_DEVICES=2 python train.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --wdb_offline False \
 --cfg configs/q2a_vit_xlnet.yaml 