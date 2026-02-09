# CommunityEncoder Official Repository

This directory is the official public code release for **CommunityEncoder final version**.
Only the final pretrain/finetune pipeline code is kept.

## Included Code
- `train/pt_hipt_stage2.sh`
- `train/pretrain_sample3.py`
- `ft_inference_augmented/ft_pipeline_hipt_stage2.sh`
- `ft_inference_augmented/finetune_pipeline.py`
- `ft_inference_augmented/finetune_user_aff.py`
- `ft_inference_augmented/finetune_com_agg.py`
- `ft_inference_augmented/finetune_soc_dim.py`
- `ft_inference_augmented/finetune_com_emo.py`
- `ft_inference_augmented/finetune_com_conf.py`
- `GPT_GNN/` core modules
- `requirements.txt`

## Environment
```bash
pip install -r requirements.txt
```

## Run (Final Pipelines)

```bash
# Pretraining
cd train
bash pt_hipt_stage2.sh
```

```bash
# Finetune + downstream evaluation
cd ../ft_inference_augmented
bash ft_pipeline_hipt_stage2.sh
```

## Data Path Note
If you use the separated data package in `../data`, pass explicit `--data_dir` when needed, e.g.:

```bash
python finetune_user_aff.py --data_dir ../../data/sample/graph_sample_time.pk
python finetune_com_emo.py --data_dir ../../data/graph/graph_sub_label2.pk
```
