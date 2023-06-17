cd ..
seeds=(789 1024 2048)
dims=(386)

for seed in ${seeds[@]}
do
for dim in ${dims[@]}
do
projname=more_train_l4_stateinput_${dim}_seed_${seed}

CUDA_VISIBLE_DEVICES=0 python inference.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --wdb_offline True \
 --seed ${seed} \
 --cfg configs/videoclip_test.yaml CKPT "/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/outputs/cvpr_loveu2023/${projname}/" MODEL.DIM_STATE ${dim}
 done
 done