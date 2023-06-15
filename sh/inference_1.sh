cd ..
projname=textcombine_videohard

CUDA_VISIBLE_DEVICES=4 python inference.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --cfg configs/videoclip_test.yaml CKPT "/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/outputs/cvpr_loveu2023/textcombine_videohard/" MODEL.TEXTGROUNDING 'combine' MODEL.VIDEOGROUNDING 'hard'