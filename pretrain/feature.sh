#S3D video feature extraction

#CUDA_VISIBLE_DEVICES=0 python encode_video.py --wdb_name video_encode_train --wdb_project cvpr_loveu2023_encode --cfg configs/S3D.yaml FOR.VIDEO True DATASET.SPLIT "video" DATASET.LABEL "/data/zclfe/cvpr_comp/data/all_data_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data"

CUDA_VISIBLE_DEVICES=0 python encode_video.py --wdb_name button_encode_train --wdb_project cvpr_loveu2023_encode --cfg configs/S3D.yaml FOR.BUTTON True DATASET.SPLIT "video" DATASET.LABEL "/data/zclfe/cvpr_comp/data/all_data_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data"
#CUDA_VISIBLE_DEVICES=0 python encode_video.py --wdb_name video_encode_train --wdb_project cvpr_loveu2023_encode --cfg configs/VedioCLIP.yaml FOR.VIDEO True DATASET.SPLIT "video" DATASET.LABEL "/data/zclfe/cvpr_comp/data/all_data_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data"