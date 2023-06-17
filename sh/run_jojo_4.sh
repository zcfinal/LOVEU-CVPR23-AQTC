cd ..
seeds=(789 1024 2048 5555 77 99 120 400)
dims=(386)

for seed in ${seeds[@]}
do
for dim in ${dims[@]}
do
projname=more_train_l4_stateinput_${dim}_seed_${seed}

CUDA_VISIBLE_DEVICES=2 python train.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --wdb_offline False \
 --seed ${seed} \
 --cfg /data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/configs/videoclip.yaml SOLVER.LR 1e-4 MODEL.DIM_STATE ${dim}
 done
 done