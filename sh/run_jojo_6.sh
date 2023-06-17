cd ..
seeds=(0 1 2 3 4 5 6 7 8 42 240)
dims=(1536)

for seed in ${seeds[@]}
do
for dim in ${dims[@]}
do
projname=more_train_l4_stateinput_${dim}_seed_${seed}

CUDA_VISIBLE_DEVICES=0 python train.py \
 --wdb_project cvpr_loveu2023 \
 --wdb_name ${projname} \
 --wdb_offline False \
 --seed ${seed} \
 --cfg /data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/configs/videoclip.yaml SOLVER.LR 1e-4 MODEL.DIM_STATE ${dim}
 done
 done