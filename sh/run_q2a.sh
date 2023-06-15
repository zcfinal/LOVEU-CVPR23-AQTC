cd ..
projname=q2a

CUDA_VISIBLE_DEVICES=2 python train.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --wdb_offline False \
 --cfg configs/q2a_vit_xlnet.yaml MODEL.ARCH 'q2a' MODEL.FUNCTION_CENTRIC False