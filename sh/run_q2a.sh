cd ..
projname=q2a

# CUDA_VISIBLE_DEVICES=2 python train.py \
#  --wdb_project cvpr_loveu2023 \
#  --wdb_name ${projname} \
#  --wdb_offline False \
#  --cfg configs/q2a_vit_xlnet.yaml MODEL.ARCH 'q2a' MODEL.FUNCTION_CENTRIC False


CUDA_VISIBLE_DEVICES=2 python inference.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --wdb_offline True \
 --cfg configs/q2a_vit_xlnet_test.yaml CKPT "/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/outputs/cvpr_loveu2023/q2a/" MODEL.ARCH 'q2a' MODEL.FUNCTION_CENTRIC False