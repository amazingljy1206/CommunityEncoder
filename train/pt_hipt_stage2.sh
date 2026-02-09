#!/bin/bash
# HIPT stage2 pretrain (final release).
python pretrain_sample3.py \
    --pretrain_model_dir ../models/pretrain_model_v2/model_pt_stage1.pt \
    --pretrain_model_dir2 ../models/pretrain_model_v2/model_pt_stage2.pt \
    --current_model_dir ../models/pretrain_model_v2/model_pt_current.pt \
    --stage_ckpt_dir ../models/pretrain_model_v2/ckpt \
    --n_epoch 15 \
    --n_iteration 12 \
    --max_lr 5e-4 \
    --attr_ratio 0.3 \
    --queue_size 512
