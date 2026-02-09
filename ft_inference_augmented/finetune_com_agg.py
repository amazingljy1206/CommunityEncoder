# -*- coding: utf-8 -*-
"""Wrapper for the unified finetune pipeline (task3 v2)."""

import sys

from finetune_pipeline import main


if __name__ == "__main__":
    main([  "--task", "task3", 
            "--method", "v2"] + sys.argv[1:])
