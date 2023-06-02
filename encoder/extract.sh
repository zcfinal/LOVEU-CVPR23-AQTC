# CUDA_VISIBLE_DEVICES=7 python main.py --wdb_name video_encode_train --wdb_project cvpr_loveu2023_encode  --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_train" 
# CUDA_VISIBLE_DEVICES=5 python main.py --wdb_name video_encode_test --wdb_project cvpr_loveu2023_encode  --cfg configs/vit_xlnet.yaml FOR.VIDEO True DATASET.SPLIT "test" DATASET.LABEL "test_with_gt_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_test"

# CUDA_VISIBLE_DEVICES=0 python main.py --wdb_name script_encode_train --wdb_project cvpr_loveu2023_encode  --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_train" &
# CUDA_VISIBLE_DEVICES=3 python main.py --wdb_name script_encode_test --wdb_project cvpr_loveu2023_encode  --cfg configs/vit_xlnet.yaml FOR.SCRIPT True DATASET.SPLIT "test" DATASET.LABEL "test_with_gt_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_test"

# CUDA_VISIBLE_DEVICES=0 python main.py --wdb_name para_encode_train --wdb_project cvpr_loveu2023_encode   --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_train" &
# CUDA_VISIBLE_DEVICES=3 python main.py --wdb_name para_encode_test --wdb_project cvpr_loveu2023_encode   --cfg configs/vit_xlnet.yaml FOR.PARA True DATASET.SPLIT "test" DATASET.LABEL "test_with_gt_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_test"

# CUDA_VISIBLE_DEVICES=0 python main.py --wdb_name qa_encode_train --wdb_project cvpr_loveu2023_encode --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_train" &
# CUDA_VISIBLE_DEVICES=3 python main.py --wdb_name qa_encode_test --wdb_project cvpr_loveu2023_encode --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "test" DATASET.LABEL "test_with_gt_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/assistq_test"


# CUDA_VISIBLE_DEVICES=2 python main.py --wdb_name qa_encode_test2023 --wdb_project cvpr_loveu2023_encode --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "video" DATASET.LABEL "test2023_without_gt_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/" SUFFIX "test"

CUDA_VISIBLE_DEVICES=2 python main.py --wdb_name machinename_encode_test2023 --wdb_project cvpr_loveu2023_encode --cfg configs/vit_xlnet.yaml FOR.MACHINE_NAME True DATASET.SPLIT "video" DATASET.LABEL "all_data_with_score.json" DATASET.ROOT "/data/zclfe/cvpr_comp/data/"