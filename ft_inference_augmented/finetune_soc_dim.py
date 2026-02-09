# -*- coding: utf-8 -*-
"""Wrapper for the unified finetune pipeline (soc_dim)."""

import sys

from finetune_pipeline import main


if __name__ == "__main__":
    main([  "--task", "soc_dim", 
            "--method", "baseline",
            "--pretrain_model_dir", "../models/model3_epoch10l_stage2.pt",
            "--model_dir", "../models/finetune_models/soc_dim/epoch50_hipt_stage2.pth"
            ] + sys.argv[1:])
