# -*- coding: utf-8 -*-
"""Wrapper for the unified finetune pipeline (com_conf)."""

import sys

from finetune_pipeline import main


if __name__ == "__main__":
    main([  "--task", "com_conf", 
            "--method", "baseline",
            "--pretrain_model_dir", "../models/model3_epoch10l_stage2.pt",
            "--model_dir", "../models/finetune_models/com_conf/epoch50_hipt_stage2.pth"
            ] + sys.argv[1:])
