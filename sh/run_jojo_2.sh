cd ..

projname=textcombine_videohard

CUDA_VISIBLE_DEVICES=1 python train.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --wdb_offline False \
 --cfg /data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/configs/videoclip.yaml SOLVER.LR 1e-4 MODEL.TEXTGROUNDING 'combine' MODEL.VIDEOGROUNDING 'hard'