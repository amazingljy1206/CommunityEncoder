# -*- coding: utf-8 -*-
"""Wrapper for the unified finetune pipeline (task1 baseline)."""

import sys

from finetune_pipeline import main


if __name__ == "__main__":
    main([  "--task", "task1", 
            "--method", "baseline"] + sys.argv[1:])
