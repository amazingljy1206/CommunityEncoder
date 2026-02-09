#!/bin/bash
# HIPT stage2 finetune + evaluation (final release)

MODELS_DIR="../models"
FT_MODELS_DIR="${MODELS_DIR}/finetune_models_v3"
PRETRAIN_HIPT_STAGE2="${MODELS_DIR}/pretrain_model_v2/model_pt_stage1.pt"
if [ ! -f "${PRETRAIN_HIPT_STAGE2}" ]; then
    PRETRAIN_HIPT_STAGE2="${MODELS_DIR}/pretrain_model_v2/model_pt_stage1.pt"
fi

# ============================== task: User Aff   ==============================
python finetune_user_aff.py \
--pretrain_model_dir ${PRETRAIN_HIPT_STAGE2} \
--model_dir ${FT_MODELS_DIR}/user_aff/epoch50_hipt_stage1.pth \
--conv_name hgt \
--prev_norm \
--last_norm \
--dropout 0.1 \

# ============================== task: Com Agg   ==============================
python finetune_com_agg.py \
--pretrain_model_dir ${PRETRAIN_HIPT_STAGE2} \
--model_dir ${FT_MODELS_DIR}/com_agg/epoch50_hipt_stage1.pth \
--conv_name hgt \
--prev_norm \
--last_norm \
--dropout 0.2 \

# ============================== task: Soc Dim   ==============================
python finetune_soc_dim.py \
--pretrain_model_dir ${PRETRAIN_HIPT_STAGE2} \
--model_dir ${FT_MODELS_DIR}/soc_dim/epoch50_hipt_stage1.pth \
--conv_name hgt \
--sample_depth 4 \
--sample_width 64 \
--prev_norm \
--last_norm \
--dropout 0.2 \

# ============================== task: Com Emo   ==============================
python finetune_com_emo.py \
--pretrain_model_dir ${PRETRAIN_HIPT_STAGE2} \
--model_dir ${FT_MODELS_DIR}/com_emo/epoch50_hipt_stage1.pth \
--conv_name hgt \
--prev_norm \
--last_norm \
--dropout 0.2 \

# ============================== task: Com Conf   ==============================
python finetune_com_conf.py \
--pretrain_model_dir ${PRETRAIN_HIPT_STAGE2} \
--model_dir ${FT_MODELS_DIR}/com_conf/epoch50_hipt_stage1.pth \
--conv_name hgt \
--sample_depth 4 \
--sample_width 64 \
--prev_norm \
--last_norm \
--dropout 0.2 \
